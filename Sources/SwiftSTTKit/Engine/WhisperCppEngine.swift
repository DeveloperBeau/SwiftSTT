@preconcurrency import Foundation
import OSLog
import SwiftSTTCore

private let engineLog = Logger(subsystem: "com.swiftstt", category: "WhisperCppEngine")

/// WhisperTranscriptionEngine backed by whisper.cpp.
///
/// Buffers PCM samples while recording; on stop, runs `whisper_full` once
/// and emits segments. This is a record-then-transcribe engine. For live
/// word-by-word streaming the implementation would need to call
/// `whisper_full` on rolling windows. Current shape covers dictation,
/// which is the only consumer.
///
/// Subscribers to ``statusStream()`` receive the current status on
/// registration. Subscribers to ``segmentStream()`` receive only segments
/// emitted after they subscribe.
public actor WhisperCppEngine: WhisperTranscriptionEngine {

    /// Factory closure that vends an ``AudioInputProvider`` for each recording session.
    public typealias AudioCaptureFactory = @Sendable () -> any AudioInputProvider

    private let storage: WhisperModelStorage
    private let audioFactory: AudioCaptureFactory

    private var statusContinuations: [UUID: AsyncStream<WhisperEngineStatus>.Continuation] = [:]
    private var segmentContinuations: [UUID: AsyncStream<TranscriptionSegment>.Continuation] = [:]
    private var currentStatus: WhisperEngineStatus = .idle

    private struct Loaded {
        let model: WhisperModel
        let context: WhisperCppContext
    }

    private var loaded: Loaded?
    private var isPreparing = false

    private var audioInput: (any AudioInputProvider)?
    private var buffer: [Float] = []
    private var captureToken: UUID?

    /// Creates a new engine with the given storage and audio input factory.
    public init(
        storage: WhisperModelStorage = WhisperModelStorage(),
        audioFactory: @escaping AudioCaptureFactory = { AVMicrophoneInput() }
    ) {
        self.storage = storage
        self.audioFactory = audioFactory
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

    /// Returns a stream of transcription segments emitted after each recording stop.
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
        emitStatus(.preparing)
        do {
            let bundle = try await downloader.bundle(for: model)
            let context = try WhisperCppContext(
                ggmlModelURL: bundle.ggmlModelURL,
                coreMLEncoderURL: bundle.coreMLEncoderURL
            )
            loaded = Loaded(model: model, context: context)
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
        guard loaded != nil else {
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

        try await input.start(
            targetSampleRate: 16_000,
            bufferDurationSeconds: 0.1
        ) { @Sendable samples in
            Task { [weak self] in
                await self?.appendSamples(samples, expectedToken: token)
            }
        }
        emitStatus(.listening)
    }

    /// Stops audio capture, runs transcription on the buffered samples, and emits segments.
    ///
    /// Idempotent: safe to call when not recording.
    public func stop() async {
        guard let input = audioInput else { return }  // truly idempotent: no-op
        captureToken = nil
        audioInput = nil
        await input.stop()

        if let cached = loaded {
            let pcm = buffer
            buffer.removeAll(keepingCapacity: true)
            do {
                let segments = try await cached.context.transcribe(
                    samples: pcm,
                    // Auto-detect language: the bundled/downloaded models
                    // are multilingual.
                    options: DecodingOptions()
                )
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

    private func appendSamples(_ samples: [Float], expectedToken: UUID) {
        guard captureToken == expectedToken else { return }
        buffer.append(contentsOf: samples)
    }
}
