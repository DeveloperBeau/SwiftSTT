@preconcurrency import CoreML
import Foundation
import Synchronization
import SwiftWhisperCore

/// Wires audio capture through VAD, mel spectrogram, encoder, decoder, and
/// local agreement into a single streaming transcription actor.
///
/// Call ``start()`` to begin capturing audio and receiving transcription
/// segments. The returned `AsyncStream` yields a new ``TranscriptionSegment``
/// each time the local agreement policy confirms stable tokens.
///
/// ```swift
/// let pipeline = TranscriptionPipeline(
///     audioInput: myProvider,
///     vad: myVAD,
///     melSpectrogram: mel,
///     encoder: encoder,
///     decoder: decoder,
///     tokenizer: tokenizer,
///     policy: LocalAgreementPolicy()
/// )
///
/// let stream = try await pipeline.start()
/// for await segment in stream {
///     print(segment.text)
/// }
/// ```
public actor TranscriptionPipeline {

    private let audioInput: any AudioInputProvider
    private let vad: any VoiceActivityDetector
    private let melSpectrogram: MelSpectrogram
    private let encoder: WhisperEncoder
    private let decoder: WhisperDecoder
    private let tokenizer: WhisperTokenizer
    private let policy: LocalAgreementPolicy
    private let options: DecodingOptions

    private let targetSampleRate: Double
    private let bufferDurationSeconds: Double
    private let decodeIntervalSeconds: Double
    private let maxBufferedFrames: Int

    private var decodeTask: Task<Void, Never>?
    private var segmentContinuation: AsyncStream<TranscriptionSegment>.Continuation?
    private var isRunning = false

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
        self.targetSampleRate = targetSampleRate
        self.bufferDurationSeconds = bufferDurationSeconds
        self.decodeIntervalSeconds = decodeIntervalSeconds
        self.maxBufferedFrames = maxBufferedFrames
    }

    /// Opens the audio stream and returns transcription segments as the model
    /// confirms them through local agreement.
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
        let nMels = 80
        var accumulatedMel = [[Float]](repeating: [], count: nMels)
        var pendingFrameCount = 0
        var wasSpeech = false
        var lastDecodeTime = Date().timeIntervalSince1970
        var audioStartOffset: TimeInterval = 0

        for await chunk in stream {
            let speech = await vad.isSpeech(chunk: chunk)

            if !speech {
                if wasSpeech && pendingFrameCount > 0 {
                    await runDecodeCycle(
                        accumulatedMel: &accumulatedMel,
                        pendingFrameCount: &pendingFrameCount,
                        audioStartOffset: audioStartOffset,
                        nMels: nMels
                    )
                    await melSpectrogram.reset()
                    await policy.reset()
                    audioStartOffset = chunk.timestamp
                }
                wasSpeech = false
                continue
            }

            if !wasSpeech {
                audioStartOffset = chunk.timestamp
            }
            wasSpeech = true

            let melResult: MelSpectrogramResult
            do {
                melResult = try await melSpectrogram.process(chunk: chunk)
            } catch {
                segmentContinuation?.finish()
                return
            }

            if melResult.nFrames > 0 {
                appendMel(melResult, to: &accumulatedMel, nMels: nMels)
                pendingFrameCount += melResult.nFrames
            }

            let now = Date().timeIntervalSince1970
            let elapsed = now - lastDecodeTime
            let shouldDecode = (elapsed >= decodeIntervalSeconds && pendingFrameCount > 0)
                || pendingFrameCount >= maxBufferedFrames

            if shouldDecode {
                lastDecodeTime = now
                await runDecodeCycle(
                    accumulatedMel: &accumulatedMel,
                    pendingFrameCount: &pendingFrameCount,
                    audioStartOffset: audioStartOffset,
                    nMels: nMels
                )
            }
        }

        // Flush any remaining mel data when the stream ends
        if pendingFrameCount > 0 {
            await runDecodeCycle(
                accumulatedMel: &accumulatedMel,
                pendingFrameCount: &pendingFrameCount,
                audioStartOffset: audioStartOffset,
                nMels: nMels
            )
        }

        segmentContinuation?.finish()
    }

    private func runDecodeCycle(
        accumulatedMel: inout [[Float]],
        pendingFrameCount: inout Int,
        audioStartOffset: TimeInterval,
        nMels: Int
    ) async {
        guard pendingFrameCount > 0 else { return }

        let snapshot: MelSpectrogramResult
        do {
            snapshot = try flattenMel(accumulatedMel, nMels: nMels, nFrames: pendingFrameCount)
        } catch {
            segmentContinuation?.finish()
            return
        }

        let encoderOutput: MLMultiArray
        do {
            encoderOutput = try await encoder.encode(spectrogram: snapshot)
        } catch {
            segmentContinuation?.finish()
            return
        }

        let tokens: [WhisperToken]
        do {
            tokens = try await decoder.decode(encoderOutput: encoderOutput, options: options)
        } catch {
            segmentContinuation?.finish()
            return
        }

        let contentTokens = tokens.filter { !tokenizer.isSpecial(token: $0.id) }
        let stableTokens = await policy.ingest(tokens: contentTokens)

        if !stableTokens.isEmpty {
            let text = tokenizer.decode(tokens: stableTokens.map(\.id))
            let durationSeconds = Double(pendingFrameCount) * 0.01
            let segment = TranscriptionSegment(
                text: text,
                start: audioStartOffset,
                end: audioStartOffset + durationSeconds
            )
            segmentContinuation?.yield(segment)
        }

        for i in 0..<nMels {
            accumulatedMel[i].removeAll(keepingCapacity: true)
        }
        pendingFrameCount = 0
    }

    // MARK: - Mel accumulation helpers

    private func appendMel(
        _ result: MelSpectrogramResult,
        to accumulated: inout [[Float]],
        nMels: Int
    ) {
        for m in 0..<nMels {
            for t in 0..<result.nFrames {
                accumulated[m].append(result.frames[m * result.nFrames + t])
            }
        }
    }

    private func flattenMel(
        _ accumulated: [[Float]],
        nMels: Int,
        nFrames: Int
    ) throws(SwiftWhisperError) -> MelSpectrogramResult {
        var flat = [Float](repeating: 0, count: nMels * nFrames)
        for m in 0..<nMels {
            for t in 0..<nFrames {
                flat[m * nFrames + t] = accumulated[m][t]
            }
        }
        return try MelSpectrogramResult(frames: flat, nMels: nMels, nFrames: nFrames)
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
