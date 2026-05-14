@preconcurrency import Foundation
import OSLog
import SwiftWhisperCore

private let engineLog = Logger(subsystem: "com.swiftwhisper", category: "WhisperCppEngine")

/// TranscriptionEngine backed by whisper.cpp.
///
/// Buffers PCM samples while recording; on stop, runs `whisper_full` once
/// and emits segments. This is a record-then-transcribe engine — for live
/// word-by-word streaming the implementation would need to call
/// `whisper_full` on rolling windows. Current shape covers dictation,
/// which is the only consumer.
///
/// Subscribers to ``statusStream()`` receive the current status on
/// registration. Subscribers to ``segmentStream()`` receive only segments
/// emitted after they subscribe.
public actor WhisperCppEngine: TranscriptionEngine {

    /// Factory closure that vends an ``AudioInputProvider`` for each recording session.
    public typealias AudioCaptureFactory = @Sendable () -> any AudioInputProvider

    private let storage: DefaultModelStorage
    private let audioFactory: AudioCaptureFactory

    private var statusContinuations: [UUID: AsyncStream<EngineStatus>.Continuation] = [:]
    private var segmentContinuations: [UUID: AsyncStream<TranscriptionSegment>.Continuation] = [:]
    private var currentStatus: EngineStatus = .idle

    private struct Loaded {
        let model: WhisperModel
        let context: WhisperCppContext
    }

    private var loaded: Loaded?

    private var audioInput: (any AudioInputProvider)?
    private var buffer: [Float] = []
    private var captureToken: UUID?

    /// Creates a new engine with the given storage and audio input factory.
    public init(
        storage: DefaultModelStorage = DefaultModelStorage(),
        audioFactory: @escaping AudioCaptureFactory = { AVMicrophoneInput() }
    ) {
        self.storage = storage
        self.audioFactory = audioFactory
    }

    /// Returns a stream of engine lifecycle status updates.
    ///
    /// The current status is replayed to every new subscriber immediately on registration.
    public nonisolated func statusStream() -> AsyncStream<EngineStatus> {
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
        continuation: AsyncStream<EngineStatus>.Continuation
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

    private func emitStatus(_ status: EngineStatus) {
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
    /// Emits ``EngineStatus/idle`` if no model is selected or not yet downloaded.
    /// Full context loading is stubbed pending Task 6 (ggmlModelURL on ModelBundle).
    public func prepare() async {
        guard let model = storage.model else {
            emitStatus(.idle)
            return
        }
        let downloader = ModelDownloader()
        guard await downloader.isDownloaded(model) else {
            emitStatus(.idle)
            return
        }
        // TODO(Task 6): after ModelBundle exposes ggmlModelURL, instantiate
        // WhisperCppContext here. For now the engine cannot load a model;
        // we emit .idle (not .failed) so consumers don't render an error state.
        emitStatus(.idle)
    }

    /// Begins audio capture and buffering.
    ///
    /// Throws if no model is loaded.
    public func start() async throws {
        guard loaded != nil else {
            emitStatus(.failed("Models still loading. Please wait."))
            throw SwiftWhisperError.modelLoadFailed(
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
    /// Idempotent — safe to call when not recording.
    public func stop() async {
        guard let input = audioInput else {
            if loaded != nil { emitStatus(.ready) }
            return
        }
        captureToken = nil
        audioInput = nil
        await input.stop()

        guard let cached = loaded else {
            buffer.removeAll(keepingCapacity: true)
            emitStatus(.ready)
            return
        }
        let pcm = buffer
        buffer.removeAll(keepingCapacity: true)

        do {
            let segments = try await cached.context.transcribe(
                samples: pcm,
                options: DecodingOptions(language: "en")
            )
            for segment in segments {
                emitSegment(segment)
            }
        } catch {
            engineLog.error(
                "transcribe failed: \(String(describing: error), privacy: .private)"
            )
        }
        emitStatus(.ready)
    }

    private func appendSamples(_ samples: [Float], expectedToken: UUID) {
        guard captureToken == expectedToken else { return }
        buffer.append(contentsOf: samples)
    }
}
