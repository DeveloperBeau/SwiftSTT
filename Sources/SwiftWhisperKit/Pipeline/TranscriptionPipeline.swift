@preconcurrency import CoreML
import Foundation
import Synchronization
import SwiftWhisperCore

/// Wires audio capture through VAD, mel spectrogram, encoder, decoder, and
/// segment emission into a single streaming transcription actor.
///
/// Call ``start()`` to begin capturing audio and receiving transcription
/// segments. The returned `AsyncStream` yields a new ``TranscriptionSegment``
/// each time the configured ``SegmentEmissionMode`` confirms one.
///
/// ## Sliding window
///
/// The pipeline keeps a rolling 30-second mel buffer inside the injected
/// ``MelSpectrogram`` actor. After each successful decode the cursor advances
/// past the consumed audio so the next encode operates on fresh content. In
/// ``SegmentEmissionMode/timestampSegments`` mode the cursor moves to the
/// last segment's predicted end time. In ``SegmentEmissionMode/stableTokens``
/// mode (or when no timestamps are emitted) the cursor moves by 28 seconds,
/// keeping a 2-second overlap so the next window has context.
///
/// ```swift
/// let pipeline = TranscriptionPipeline(
///     audioInput: myProvider,
///     vad: myVAD,
///     melSpectrogram: mel,
///     encoder: encoder,
///     decoder: decoder,
///     tokenizer: tokenizer,
///     policy: LocalAgreementPolicy(),
///     emissionMode: .timestampSegments
/// )
///
/// let stream = try await pipeline.start()
/// for await segment in stream {
///     print(segment.text)
/// }
/// ```
public actor TranscriptionPipeline {

    /// Mel frames Whisper consumes per second of audio (10 ms hop).
    public static let framesPerSecond: Int = 100

    /// Whisper's encoder window in seconds.
    public static let windowDurationSeconds: Double = 30.0

    /// Cursor advance applied when no timestamp is available, leaving a
    /// 2-second overlap into the next window for context.
    public static let advanceFallbackSeconds: Double = 28.0

    private let audioInput: any AudioInputProvider
    private let vad: any VoiceActivityDetector
    private let melSpectrogram: MelSpectrogram
    private let encoder: WhisperEncoder
    private let decoder: WhisperDecoder
    private let tokenizer: WhisperTokenizer
    private let policy: LocalAgreementPolicy
    private let options: DecodingOptions
    private let emissionMode: SegmentEmissionMode

    private let targetSampleRate: Double
    private let bufferDurationSeconds: Double
    private let decodeIntervalSeconds: Double
    private let maxBufferedFrames: Int

    private var decodeTask: Task<Void, Never>?
    private var segmentContinuation: AsyncStream<TranscriptionSegment>.Continuation?
    private var isRunning = false

    /// Frames the pipeline has already retired from the rolling mel buffer.
    /// Used to compute the per-decode `windowOffsetSeconds`.
    private var melCursorFrames: Int = 0

    /// Shared box that the `@Sendable` audio callback and the actor can both
    /// access. Holds the chunk continuation behind a `Mutex` so the callback
    /// can yield without crossing the actor boundary, and `stop()` can finish
    /// it from the actor side.
    private var chunkBox: ChunkBox?

    public init(
        audioInput: any AudioInputProvider,
        vad: any VoiceActivityDetector,
        melSpectrogram: MelSpectrogram,
        encoder: WhisperEncoder,
        decoder: WhisperDecoder,
        tokenizer: WhisperTokenizer,
        policy: LocalAgreementPolicy,
        options: DecodingOptions = .default,
        emissionMode: SegmentEmissionMode = .stableTokens,
        targetSampleRate: Double = 16_000,
        bufferDurationSeconds: Double = 0.064,
        decodeIntervalSeconds: Double = 1.0,
        maxBufferedFrames: Int = 3_000
    ) {
        self.audioInput = audioInput
        self.vad = vad
        self.melSpectrogram = melSpectrogram
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.policy = policy
        self.options = options
        self.emissionMode = emissionMode
        self.targetSampleRate = targetSampleRate
        self.bufferDurationSeconds = bufferDurationSeconds
        self.decodeIntervalSeconds = decodeIntervalSeconds
        self.maxBufferedFrames = maxBufferedFrames
    }

    /// Opens the audio stream and returns transcription segments as the model
    /// confirms them.
    ///
    /// The stream finishes when ``stop()`` is called, the audio input ends, or
    /// an unrecoverable encoder/decoder error occurs.
    public func start() async throws(SwiftWhisperError) -> AsyncStream<TranscriptionSegment> {
        guard !isRunning else {
            return AsyncStream { $0.finish() }
        }

        let (segmentStream, segCont) = AsyncStream<TranscriptionSegment>.makeStream()
        self.segmentContinuation = segCont

        let (chunkStream, chunkCont) = AsyncStream<AudioChunk>.makeStream()
        let box = ChunkBox(continuation: chunkCont)
        self.chunkBox = box

        let startTime = Date().timeIntervalSince1970
        let rate = targetSampleRate

        try await audioInput.start(
            targetSampleRate: targetSampleRate,
            bufferDurationSeconds: bufferDurationSeconds
        ) { @Sendable samples in
            let timestamp = Date().timeIntervalSince1970 - startTime
            box.yield(
                AudioChunk(samples: samples, sampleRate: Int(rate), timestamp: timestamp)
            )
        }

        isRunning = true
        melCursorFrames = 0

        decodeTask = Task { [weak self] in
            guard let self else { return }
            await self.processAudioStream(chunkStream)
        }

        return segmentStream
    }

    /// Cancels capture and finishes the segment stream. Idempotent.
    public func stop() async {
        guard isRunning else { return }
        isRunning = false
        chunkBox?.finish()
        chunkBox = nil
        decodeTask?.cancel()
        decodeTask = nil
        await audioInput.stop()
        segmentContinuation?.finish()
        segmentContinuation = nil
    }

    // MARK: - Audio processing loop

    private func processAudioStream(_ stream: AsyncStream<AudioChunk>) async {
        var wasSpeech = false
        var lastDecodeTime = Date().timeIntervalSince1970

        for await chunk in stream {
            let speech = await vad.isSpeech(chunk: chunk)

            if !speech {
                let pendingFrames = await melSpectrogram.currentFrameCount()
                if wasSpeech && pendingFrames > 0 {
                    await runDecodeCycle()
                    await melSpectrogram.reset()
                    await policy.reset()
                    melCursorFrames = 0
                }
                wasSpeech = false
                continue
            }

            wasSpeech = true

            do {
                _ = try await melSpectrogram.process(chunk: chunk)
            } catch {
                segmentContinuation?.finish()
                return
            }

            let pendingFrames = await melSpectrogram.currentFrameCount()
            let now = Date().timeIntervalSince1970
            let elapsed = now - lastDecodeTime
            let shouldDecode = (elapsed >= decodeIntervalSeconds && pendingFrames > 0)
                || pendingFrames >= maxBufferedFrames

            if shouldDecode {
                lastDecodeTime = now
                await runDecodeCycle()
            }
        }

        let pendingFrames = await melSpectrogram.currentFrameCount()
        if pendingFrames > 0 {
            await runDecodeCycle()
        }

        segmentContinuation?.finish()
    }

    private func runDecodeCycle() async {
        let snapshot: MelSpectrogramResult
        do {
            snapshot = try await melSpectrogram.snapshot()
        } catch {
            segmentContinuation?.finish()
            return
        }
        guard snapshot.nFrames > 0 else { return }

        let windowOffsetSeconds = Double(melCursorFrames) / Double(Self.framesPerSecond)

        let encoderOutput: MLMultiArray
        do {
            encoderOutput = try await encoder.encode(spectrogram: snapshot)
        } catch {
            segmentContinuation?.finish()
            return
        }

        var effectiveOptions = options
        if emissionMode == .timestampSegments {
            effectiveOptions.withoutTimestamps = false
        }

        let tokens: [WhisperToken]
        do {
            tokens = try await decoder.decode(encoderOutput: encoderOutput, options: effectiveOptions)
        } catch {
            segmentContinuation?.finish()
            return
        }

        switch emissionMode {
        case .stableTokens:
            await emitStableTokens(
                tokens: tokens,
                windowOffsetSeconds: windowOffsetSeconds,
                snapshotFrames: snapshot.nFrames
            )
        case .timestampSegments:
            await emitTimestampSegments(
                tokens: tokens,
                windowOffsetSeconds: windowOffsetSeconds,
                snapshotFrames: snapshot.nFrames
            )
        }
    }

    private func emitStableTokens(
        tokens: [WhisperToken],
        windowOffsetSeconds: TimeInterval,
        snapshotFrames: Int
    ) async {
        let contentTokens = tokens.filter { !tokenizer.isSpecial(token: $0.id) }
        let stableTokens = await policy.ingest(tokens: contentTokens)

        if !stableTokens.isEmpty {
            let text = tokenizer.decode(tokens: stableTokens.map(\.id))
            let durationSeconds = Double(snapshotFrames) / Double(Self.framesPerSecond)
            let segment = TranscriptionSegment(
                text: text,
                start: windowOffsetSeconds,
                end: windowOffsetSeconds + durationSeconds
            )
            segmentContinuation?.yield(segment)
        }
        await advanceCursorFallback(snapshotFrames: snapshotFrames)
    }

    private func emitTimestampSegments(
        tokens: [WhisperToken],
        windowOffsetSeconds: TimeInterval,
        snapshotFrames: Int
    ) async {
        let segments = WhisperDecoder.parseSegments(
            tokens: tokens,
            tokenizer: tokenizer,
            windowOffsetSeconds: windowOffsetSeconds
        )

        if segments.isEmpty {
            await advanceCursorFallback(snapshotFrames: snapshotFrames)
            return
        }

        for segment in segments {
            segmentContinuation?.yield(segment)
        }

        let lastEnd = segments[segments.count - 1].end
        let absoluteEndFrame = Int((lastEnd * Double(Self.framesPerSecond)).rounded())
        let consumeFrames = max(0, absoluteEndFrame - melCursorFrames)
        await advance(framesConsumed: consumeFrames, snapshotFrames: snapshotFrames)
    }

    private func advanceCursorFallback(snapshotFrames: Int) async {
        let fallbackFrames = Int(Self.advanceFallbackSeconds * Double(Self.framesPerSecond))
        let consume = min(snapshotFrames, fallbackFrames)
        await advance(framesConsumed: consume, snapshotFrames: snapshotFrames)
    }

    private func advance(framesConsumed: Int, snapshotFrames: Int) async {
        let consume = max(0, min(framesConsumed, snapshotFrames))
        guard consume > 0 else { return }
        melCursorFrames += consume
        await melSpectrogram.advance(framesConsumed: consume)
    }
}

// MARK: - ChunkBox

/// Thread-safe wrapper around `AsyncStream.Continuation` so the audio callback
/// (which runs off-actor on a real-time thread) can yield chunks without
/// crossing the actor boundary.
private final class ChunkBox: @unchecked Sendable {

    private let continuation: Mutex<AsyncStream<AudioChunk>.Continuation?>

    init(continuation: AsyncStream<AudioChunk>.Continuation) {
        self.continuation = Mutex(continuation)
    }

    func yield(_ chunk: AudioChunk) {
        continuation.withLock { _ = $0?.yield(chunk) }
    }

    func finish() {
        continuation.withLock {
            $0?.finish()
            $0 = nil
        }
    }
}
