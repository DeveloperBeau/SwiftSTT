@preconcurrency import Foundation
import OSLog
import SwiftSTTCore

private let engineLog = Logger(subsystem: "com.swiftstt", category: "WhisperCppEngine")

/// WhisperTranscriptionEngine backed by whisper.cpp.
///
/// Two timing modes, chosen via ``init(storage:audioFactory:timing:)``:
///
/// - ``TranscriptionTiming/onStop`` (the default) buffers PCM samples while
///   recording and runs `whisper_full` once on stop. This is record-then-
///   transcribe, and covers dictation, the original consumer.
/// - ``TranscriptionTiming/whileRecording(_:)`` cuts the incoming audio into
///   ``AudioWindow``s on silence (falling back to a duration limit) and runs
///   `whisper_full` once per window as capture proceeds, reconciling the
///   overlap at forced cuts so no word is lost or duplicated.
///
/// Subscribers to ``statusStream()`` receive the current status on
/// registration. Subscribers to ``segmentStream()`` receive only segments
/// emitted after they subscribe.
public actor WhisperCppEngine: WhisperTranscriptionEngine {

    /// Factory closure that vends an ``AudioInputProvider`` for each recording session.
    public typealias AudioCaptureFactory = @Sendable () -> any AudioInputProvider

    /// Transcribes one window of 16 kHz mono Float32 PCM into segments.
    typealias WindowTranscribing = @Sendable ([Float]) async throws -> [TranscriptionSegment]

    private let storage: WhisperModelStorage
    private let audioFactory: AudioCaptureFactory
    private let timing: TranscriptionTiming
    private let makeCutter: @Sendable (StreamingWindowPolicy) -> AudioWindowCutter

    private var statusContinuations: [UUID: AsyncStream<WhisperEngineStatus>.Continuation] = [:]
    private var segmentContinuations: [UUID: AsyncStream<TranscriptionSegment>.Continuation] = [:]
    private var currentStatus: WhisperEngineStatus = .idle

    private struct Loaded {
        let model: WhisperModel
        let context: WhisperCppContext
    }

    private var loaded: Loaded?
    private var isPreparing = false
    private var transcribeWindow: WindowTranscribing?

    private var audioInput: (any AudioInputProvider)?
    private var buffer: [Float] = []
    private var captureToken: UUID?
    private var sampleContinuation: AsyncStream<[Float]>.Continuation?
    private var captureTask: Task<Void, Never>?

    private var cutter: AudioWindowCutter?
    private var seams: SeamReconciler?
    private var decodeTask: Task<Void, Never>?

    /// Creates a new engine with the given storage, audio input factory, and timing mode.
    public init(
        storage: WhisperModelStorage = WhisperModelStorage(),
        audioFactory: @escaping AudioCaptureFactory = { AVMicrophoneInput() },
        timing: TranscriptionTiming = .onStop
    ) {
        self.storage = storage
        self.audioFactory = audioFactory
        self.timing = timing
        self.makeCutter = { AudioWindowCutter(policy: $0) }
    }

    /// Test seat: supplies the decode step and the cutter directly, so the
    /// streaming path can be exercised without loading a model and without a
    /// real voice activity detector. Not public.
    init(
        storage: WhisperModelStorage,
        audioFactory: @escaping AudioCaptureFactory,
        timing: TranscriptionTiming,
        transcribeWindow: @escaping WindowTranscribing,
        makeCutter: @escaping @Sendable (StreamingWindowPolicy) -> AudioWindowCutter = {
            AudioWindowCutter(policy: $0)
        }
    ) {
        self.storage = storage
        self.audioFactory = audioFactory
        self.timing = timing
        self.transcribeWindow = transcribeWindow
        self.makeCutter = makeCutter
    }

    /// Returns a stream of engine lifecycle status updates.
    ///
    /// The current status is replayed to every new subscriber immediately on registration.
    public nonisolated func statusStream() -> AsyncStream<WhisperEngineStatus> {
        AsyncStream { continuation in
            let id = UUID()
            Task { [weak self] in
                await self?.registerStatusContinuation(id: id, continuation: continuation)
            }
            continuation.onTermination = { _ in
                Task { [weak self] in
                    await self?.removeStatusContinuation(id: id)
                }
            }
        }
    }

    /// Returns a stream of transcription segments.
    ///
    /// Under ``TranscriptionTiming/onStop`` segments arrive in one burst when
    /// recording stops. Under ``TranscriptionTiming/whileRecording(_:)`` they
    /// arrive incrementally as each audio window is decoded during capture.
    public nonisolated func segmentStream() -> AsyncStream<TranscriptionSegment> {
        AsyncStream { continuation in
            let id = UUID()
            Task { [weak self] in
                await self?.registerSegmentContinuation(id: id, continuation: continuation)
            }
            continuation.onTermination = { _ in
                Task { [weak self] in
                    await self?.removeSegmentContinuation(id: id)
                }
            }
        }
    }

    private func registerStatusContinuation(
        id: UUID,
        continuation: AsyncStream<WhisperEngineStatus>.Continuation
    ) {
        statusContinuations[id] = continuation
        continuation.yield(currentStatus)
    }

    private func removeStatusContinuation(id: UUID) {
        statusContinuations.removeValue(forKey: id)
    }

    private func registerSegmentContinuation(
        id: UUID,
        continuation: AsyncStream<TranscriptionSegment>.Continuation
    ) {
        segmentContinuations[id] = continuation
    }

    private func removeSegmentContinuation(id: UUID) {
        segmentContinuations.removeValue(forKey: id)
    }

    private func emitStatus(_ status: WhisperEngineStatus) {
        currentStatus = status
        for (_, cont) in statusContinuations {
            cont.yield(status)
        }
    }

    private func emitSegment(_ segment: TranscriptionSegment) {
        for (_, cont) in segmentContinuations {
            cont.yield(segment)
        }
    }

    /// Attempts to load the persisted default model into memory.
    ///
    /// Emits ``WhisperEngineStatus/idle`` if no model is selected or not yet downloaded.
    /// Emits ``WhisperEngineStatus/ready`` when the model is successfully loaded.
    /// Emits ``WhisperEngineStatus/failed(_:)`` if loading fails.
    public func prepare() async {
        guard !isPreparing else { return }
        isPreparing = true
        defer { isPreparing = false }
        guard let model = storage.model else {
            emitStatus(.idle)
            return
        }
        let downloader = ModelDownloader()
        guard await downloader.isDownloaded(model) else {
            emitStatus(.idle)
            return
        }
        if let cached = loaded, cached.model == model {
            emitStatus(.ready)
            return
        }
        loaded = nil
        transcribeWindow = nil
        emitStatus(.preparing)
        do {
            let bundle = try await downloader.bundle(for: model)
            let context = try WhisperCppContext(
                ggmlModelURL: bundle.ggmlModelURL,
                coreMLEncoderURL: bundle.coreMLEncoderURL
            )
            loaded = Loaded(model: model, context: context)
            // Auto-detect language: the bundled/downloaded models are multilingual.
            transcribeWindow = { [context] samples in
                try await context.transcribe(samples: samples, options: DecodingOptions())
            }
            emitStatus(.ready)
        } catch {
            engineLog.error(
                "prepare failed: \(String(describing: error), privacy: .private)"
            )
            emitStatus(.failed("Couldn't prepare the dictation model."))
        }
    }

    /// Begins audio capture and buffering.
    ///
    /// Throws if no model is loaded.
    public func start() async throws {
        guard transcribeWindow != nil else {
            emitStatus(.failed("Models still loading. Please wait."))
            throw SwiftSTTError.modelLoadFailed(
                "models not loaded; call prepare() first"
            )
        }
        guard audioInput == nil else { return }

        let input = audioFactory()
        audioInput = input
        buffer.removeAll(keepingCapacity: true)
        let token = UUID()
        captureToken = token

        if case .whileRecording(let policy) = timing {
            cutter = makeCutter(policy)
            seams = SeamReconciler(overlapDuration: policy.overlapDuration)
            decodeTask = nil
        }

        let (capturedSamples, continuation) = AsyncStream<[Float]>.makeStream(
            bufferingPolicy: .unbounded
        )
        sampleContinuation = continuation
        captureTask = Task { [weak self] in
            for await batch in capturedSamples {
                await self?.appendSamples(batch, expectedToken: token)
            }
        }

        try await input.start(
            targetSampleRate: 16_000,
            bufferDurationSeconds: 0.1
        ) { @Sendable batch in
            continuation.yield(batch)
        }
        emitStatus(.listening)
    }

    /// Stops audio capture, runs transcription on the buffered samples, and emits segments.
    ///
    /// Idempotent: safe to call when not recording.
    public func stop() async {
        guard let input = audioInput else { return }  // truly idempotent: no-op
        audioInput = nil
        await input.stop()

        sampleContinuation?.finish()
        sampleContinuation = nil
        await captureTask?.value
        captureTask = nil
        // Cleared only after the drain above so the guard in appendSamples
        // does not reject this session's own already-queued backlog while
        // captureTask is still consuming it.
        captureToken = nil

        if cutter != nil {
            if let final = await cutter?.flush() {
                enqueueDecode(of: final)
            }
            await decodeTask?.value
            cutter = nil
            seams = nil
            decodeTask = nil
        } else if let transcribeWindow {
            let pcm = buffer
            buffer.removeAll(keepingCapacity: true)
            do {
                let segments = try await transcribeWindow(pcm)
                for segment in segments {
                    emitSegment(segment)
                }
            } catch {
                engineLog.error(
                    "transcribe failed: \(String(describing: error), privacy: .private)"
                )
            }
        } else {
            buffer.removeAll(keepingCapacity: true)
        }

        // Close segment streams for this recording session.
        for (_, cont) in segmentContinuations {
            cont.finish()
        }
        segmentContinuations.removeAll(keepingCapacity: true)

        emitStatus(.ready)
    }

    private func appendSamples(_ samples: [Float], expectedToken: UUID) async {
        guard captureToken == expectedToken else { return }
        switch timing {
        case .onStop:
            buffer.append(contentsOf: samples)
        case .whileRecording:
            if let window = await cutter?.ingest(samples) {
                enqueueDecode(of: window)
            }
        }
    }

    private func enqueueDecode(of window: AudioWindow) {
        let previous = decodeTask
        decodeTask = Task { [weak self] in
            await previous?.value
            await self?.decodeAndEmit(window)
        }
    }

    private func decodeAndEmit(_ window: AudioWindow) async {
        guard let transcribeWindow else { return }
        do {
            let decoded = try await transcribeWindow(window.samples)
            let emitted = seams?.reconcile(decoded, from: window) ?? decoded
            for segment in emitted { emitSegment(segment) }
        } catch {
            engineLog.error(
                "transcribe failed: \(String(describing: error), privacy: .private)"
            )
        }
    }
}
