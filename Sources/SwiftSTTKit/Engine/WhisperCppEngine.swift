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

    /// One recording session's mutable state, held by the actor as
    /// ``currentSession`` and locally by ``stop()`` from the moment it is
    /// entered.
    ///
    /// Swift actors are reentrant across `await`: while a `stop()` call is
    /// suspended, a `start()` call for a *new* session can run to completion
    /// on the same actor. Bundling everything one recording session owns
    /// into a single reference-typed object, rather than a handful of flat
    /// actor properties, is what makes that safe. `stop()` snapshots its own
    /// `Session` into a local *before* its first suspension point and then
    /// only ever touches that local for the rest of the method — a
    /// reentrant `start()` installs a brand new `Session` in
    /// `currentSession`, and the two never share storage, so neither call
    /// can clobber the other's continuations, tasks, cutter or reconciler.
    ///
    /// `@unchecked Sendable`: every stored property is mutated only from
    /// code running on `WhisperCppEngine`'s own actor executor
    /// (`start()`, `stop()`, `appendSamples`, `enqueueDecode`,
    /// `decodeAndEmit`). The `Task` closures that capture a `Session` never
    /// read or write its properties directly; they only forward the
    /// reference into an actor-isolated call.
    private final class Session: @unchecked Sendable {
        let input: any AudioInputProvider
        var buffer: [Float] = []
        var sampleContinuation: AsyncStream<[Float]>.Continuation?
        var captureTask: Task<Void, Never>?
        var cutter: AudioWindowCutter?
        var seams: SeamReconciler?
        var decodeTask: Task<Void, Never>?

        init(input: any AudioInputProvider) {
            self.input = input
        }
    }

    private let storage: WhisperModelStorage
    private let audioFactory: AudioCaptureFactory
    private let timing: TranscriptionTiming
    private let makeCutter: @Sendable (StreamingWindowPolicy) -> AudioWindowCutter

    private var statusContinuations: [UUID: AsyncStream<WhisperEngineStatus>.Continuation] = [:]
    private var segmentContinuations: [UUID: AsyncStream<TranscriptionSegment>.Continuation] = [:]
    private var updateContinuations: [UUID: AsyncStream<TranscriptUpdate>.Continuation] = [:]
    private var currentStatus: WhisperEngineStatus = .idle

    private struct Loaded {
        let model: WhisperModel
        let context: WhisperCppContext
    }

    private var loaded: Loaded?
    private var isPreparing = false
    private var transcribeWindow: WindowTranscribing?

    private var currentSession: Session?

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
    ///
    /// > Important: this stream is append-only, so under
    /// > ``TranscriptionTiming/whileRecording(_:)`` it cannot express the one
    /// > thing streaming needs to say: that a window's last words were a guess
    /// > the next window has since corrected. Those words arrive here and stay.
    /// > Use ``transcriptStream()`` for a transcript that takes them back.
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

    /// Returns a stream of transcript updates, each appending segments and
    /// occasionally withdrawing the ones it supersedes.
    ///
    /// This is the whole transcript's stream, not just its additions. A window
    /// cut through speech is decoded without the audio that follows it, and
    /// what the model made of its final words is a guess. Emitting the guess
    /// immediately is what keeps streaming worth doing; withdrawing it when the
    /// next window disagrees is what keeps the transcript right. Apply each
    /// update with ``TranscriptUpdate/apply(to:)``.
    ///
    /// Under ``TranscriptionTiming/onStop`` nothing is ever withdrawn: the
    /// whole recording is decoded at once, with no window edges to guess at.
    public nonisolated func transcriptStream() -> AsyncStream<TranscriptUpdate> {
        AsyncStream { continuation in
            let id = UUID()
            Task { [weak self] in
                await self?.registerUpdateContinuation(id: id, continuation: continuation)
            }
            continuation.onTermination = { _ in
                Task { [weak self] in
                    await self?.removeUpdateContinuation(id: id)
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

    private func registerUpdateContinuation(
        id: UUID,
        continuation: AsyncStream<TranscriptUpdate>.Continuation
    ) {
        updateContinuations[id] = continuation
    }

    private func removeUpdateContinuation(id: UUID) {
        updateContinuations.removeValue(forKey: id)
    }

    private func emitStatus(_ status: WhisperEngineStatus) {
        currentStatus = status
        for (_, cont) in statusContinuations {
            cont.yield(status)
        }
    }

    private func emitSegment(_ segment: TranscriptionSegment) {
        emitUpdate(TranscriptUpdate(segments: [segment]))
    }

    /// Sends `update` to both streams. Subscribers to ``segmentStream()`` see
    /// only what it appends, having no way to be told about a retraction.
    private func emitUpdate(_ update: TranscriptUpdate) {
        for (_, cont) in updateContinuations {
            cont.yield(update)
        }
        for (_, cont) in segmentContinuations {
            for segment in update.segments { cont.yield(segment) }
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
        guard currentSession == nil else { return }

        let input = audioFactory()
        let session = Session(input: input)
        currentSession = session

        if case .whileRecording(let policy) = timing {
            session.cutter = makeCutter(policy)
            session.seams = SeamReconciler(overlapDuration: policy.overlapDuration)
        }

        let (capturedSamples, continuation) = AsyncStream<[Float]>.makeStream(
            bufferingPolicy: .unbounded
        )
        session.sampleContinuation = continuation
        session.captureTask = Task { [weak self] in
            for await batch in capturedSamples {
                await self?.appendSamples(batch, session: session)
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
        guard let session = currentSession else { return }  // truly idempotent: no-op
        currentSession = nil
        await session.input.stop()

        session.sampleContinuation?.finish()
        session.sampleContinuation = nil
        await session.captureTask?.value
        session.captureTask = nil

        if let cutter = session.cutter {
            if let final = await cutter.flush() {
                enqueueDecode(of: final, session: session)
            }
            await session.decodeTask?.value
            session.cutter = nil
            session.seams = nil
            session.decodeTask = nil
        } else if let transcribeWindow {
            let pcm = session.buffer
            session.buffer.removeAll(keepingCapacity: true)
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
            session.buffer.removeAll(keepingCapacity: true)
        }

        // Only finalize engine-wide state (segment streams, `.ready`) if no
        // reentrant start() installed a new session while this stop() was
        // suspended above — otherwise that session is still recording, and
        // finishing its subscribers' streams or reporting `.ready` over it
        // would be exactly the clobbering this method must not do.
        if currentSession == nil {
            for (_, cont) in updateContinuations {
                cont.finish()
            }
            updateContinuations.removeAll(keepingCapacity: true)
            for (_, cont) in segmentContinuations {
                cont.finish()
            }
            segmentContinuations.removeAll(keepingCapacity: true)
            emitStatus(.ready)
        }
    }

    private func appendSamples(_ samples: [Float], session: Session) async {
        // No `currentSession` guard here: `session` is always the one
        // `captureTask` was created for, its storage is never shared with
        // any other session, and `stop()` clears `currentSession` before
        // draining `captureTask` (so a reentrant `start()` can begin) —
        // guarding on identity here would reject this session's own
        // still-draining backlog. A provider that keeps calling `onChunk`
        // after `stop()` is already handled: `sampleContinuation.finish()`
        // makes any further yield into it a no-op.
        switch timing {
        case .onStop:
            session.buffer.append(contentsOf: samples)
        case .whileRecording:
            if let window = await session.cutter?.ingest(samples) {
                enqueueDecode(of: window, session: session)
            }
        }
    }

    private func enqueueDecode(of window: AudioWindow, session: Session) {
        // ponytail: unbounded decode queue. If decode is slower than realtime the
        // chain grows without limit. Add a drop-oldest cap when a model that slow is
        // actually used.
        let previous = session.decodeTask
        session.decodeTask = Task { [weak self] in
            await previous?.value
            await self?.decodeAndEmit(window, session: session)
        }
    }

    private func decodeAndEmit(_ window: AudioWindow, session: Session) async {
        guard let transcribeWindow else { return }
        do {
            let decoded = try await transcribeWindow(window.samples)
            let update =
                session.seams?.reconcile(decoded, from: window)
                ?? TranscriptUpdate(segments: decoded)
            emitUpdate(update)
        } catch {
            engineLog.error(
                "transcribe failed: \(String(describing: error), privacy: .private)"
            )
        }
    }
}
