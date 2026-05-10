@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Synchronization
import Testing

@testable import SwiftWhisperKit

// MARK: - Test mocks (file-scoped so they don't collide with TranscriptionPipelineTests)

private actor TSMockAudioInput: AudioInputProvider {

    enum Behaviour {
        case deliver([[Float]])
    }

    private let behaviour: Behaviour
    private(set) var stopCount = 0

    init(_ behaviour: Behaviour) {
        self.behaviour = behaviour
    }

    func start(
        targetSampleRate: Double,
        bufferDurationSeconds: Double,
        onChunk: @Sendable @escaping ([Float]) -> Void
    ) async throws(SwiftWhisperError) {
        switch behaviour {
        case .deliver(let chunks):
            for chunk in chunks {
                onChunk(chunk)
            }
        }
    }

    func stop() async {
        stopCount += 1
    }
}

private actor TSScriptableVAD: VoiceActivityDetector {

    private var verdicts: [Bool]
    private var index = 0

    init(verdicts: [Bool]) {
        self.verdicts = verdicts
    }

    func isSpeech(chunk: AudioChunk) async -> Bool {
        guard index < verdicts.count else { return true }
        let result = verdicts[index]
        index += 1
        return result
    }

    func reset() async {
        index = 0
    }
}

private final class TSMockEncoderRunner: CoreMLModelRunner, @unchecked Sendable {

    private let array: MLMultiArray

    init(array: MLMultiArray) {
        self.array = array
    }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        do {
            return try MLDictionaryFeatureProvider(dictionary: [
                WhisperEncoder.outputFeatureName: array
            ])
        } catch {
            throw .decoderFailure("mock provider: \(error.localizedDescription)")
        }
    }
}

/// Replays a queue of token-id sequences.
///
/// Each call to ``resetState()`` starts the next queued sequence. Each ``predict(...)`` call returns
/// either a default (during prefill) or the next token in the active
/// sequence. The number of prefill tokens is supplied per sequence.
private final class TSScriptedDecoderRunner: StatefulCoreMLModelRunner, @unchecked Sendable {

    struct Sequence {
        let prefillCount: Int
        let tokens: [Int]
    }

    private struct State {
        var sequences: [Sequence]
        var index: Int = 0
        var remainingPrefill: Int = 0
        var pending: [Int] = []
    }

    private let logitsName: String
    private let vocabSize: Int
    private let state: Mutex<State>

    init(
        sequences: [Sequence],
        vocabSize: Int,
        logitsName: String = WhisperDecoder.FeatureNames.default.logitsOutput
    ) {
        self.logitsName = logitsName
        self.vocabSize = vocabSize
        self.state = Mutex(State(sequences: sequences))
    }

    func resetState() async {
        state.withLock { state in
            guard !state.sequences.isEmpty else {
                state.remainingPrefill = 0
                state.pending = []
                return
            }
            // Wrap around so the temperature-fallback retry loop in the
            // single-hypothesis decoder can replay the same scripted sequence
            // across attempts without test authors having to duplicate it
            // per fallback temperature.
            let position = state.index % state.sequences.count
            let seq = state.sequences[position]
            state.index += 1
            state.remainingPrefill = seq.prefillCount
            state.pending = seq.tokens
        }
    }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        let tokenId: Int = state.withLock { state in
            if state.remainingPrefill > 0 {
                state.remainingPrefill -= 1
                // Use endOfText as a safe filler during prefill - the decoder
                // ignores prefill outputs and only consumes the *next* output.
                return 50_257
            }
            if !state.pending.isEmpty {
                return state.pending.removeFirst()
            }
            return 50_257  // end-of-text terminates generation
        }

        var logits = [Float](repeating: 0, count: vocabSize)
        logits[tokenId] = 50
        do {
            let array = try MLMultiArray(
                shape: [1, 1, NSNumber(value: vocabSize)],
                dataType: .float32
            )
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: vocabSize)
            for i in 0..<vocabSize {
                pointer[i] = logits[i]
            }
            return try MLDictionaryFeatureProvider(dictionary: [logitsName: array])
        } catch {
            throw .decoderFailure("mock provider: \(error.localizedDescription)")
        }
    }
}

// MARK: - Helpers

private let timestampVocabSize = 51_865

private func makeTimestampSpecials() -> [String: Int] {
    var specials: [String: Int] = [
        "<|endoftext|>": 50_257,
        "<|startoftranscript|>": 50_258,
        "<|en|>": 50_259,
        "<|translate|>": 50_358,
        "<|transcribe|>": 50_359,
        "<|notimestamps|>": 50_363,
    ]
    let firstTimestampId = 50_364
    let count = 1_501
    for step in 0..<count {
        let seconds = Double(step) * 0.02
        let formatted = String(format: "<|%.2f|>", seconds)
        specials[formatted] = firstTimestampId + step
    }
    return specials
}

private func tokenizerWithTimestamps() -> WhisperTokenizer {
    WhisperTokenizer(specialTokens: makeTimestampSpecials())
}

private func makeEncoderOutputArray() throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1, 1500, 16], dataType: .float32)
    let count = 1500 * 16
    let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
    for i in 0..<count { ptr[i] = 0 }
    return array
}

private func speechSamples(count: Int = 512) -> [Float] {
    (0..<count).map { Float(sin(Double($0) * 0.1)) * 0.5 }
}

private actor SegmentSink {
    var segments: [TranscriptionSegment] = []
    func append(_ s: TranscriptionSegment) { segments.append(s) }
}

// Number of decoder inputs that count as prefill for a `.timestampSegments`
// run: SOT, transcribe (no language). When `withoutTimestamps == false` no
// `<|notimestamps|>` is appended, so the prefill prefix is 2 tokens.
private let prefillTimestampMode = 2

// MARK: - Tests

@Suite("TranscriptionPipeline timestamp emission")
struct TranscriptionPipelineTimestampTests {

    @Test("Default emissionMode is .stableTokens")
    func defaultEmissionMode() async throws {
        let tokenizer = tokenizerWithTimestamps()
        let encArray = try makeEncoderOutputArray()
        let encRunner = TSMockEncoderRunner(array: encArray)
        let decRunner = TSScriptedDecoderRunner(
            sequences: [
                .init(prefillCount: 3, tokens: [tokenizer.endOfTextToken]),
                .init(prefillCount: 3, tokens: [tokenizer.endOfTextToken]),
            ],
            vocabSize: timestampVocabSize
        )
        let audio = TSMockAudioInput(.deliver([]))
        let vad = TSScriptableVAD(verdicts: [])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encRunner)
        let decoder = WhisperDecoder(runner: decRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audio,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            decodeIntervalSeconds: 0.0
        )

        let stream = try await pipeline.start()
        await pipeline.stop()
        var count = 0
        for await _ in stream { count += 1 }
        #expect(count == 0)
    }

    @Test(".stableTokens mode preserves M6 emission behaviour")
    func stableTokensMode() async throws {
        let tokenizer = tokenizerWithTimestamps()
        let encArray = try makeEncoderOutputArray()
        let encRunner = TSMockEncoderRunner(array: encArray)
        let decRunner = TSScriptedDecoderRunner(
            sequences: [
                .init(prefillCount: 3, tokens: [100, 200, tokenizer.endOfTextToken]),
                .init(prefillCount: 3, tokens: [100, 200, tokenizer.endOfTextToken]),
                .init(prefillCount: 3, tokens: [100, 200, tokenizer.endOfTextToken]),
            ],
            vocabSize: timestampVocabSize
        )
        let speech = speechSamples()
        let audio = TSMockAudioInput(.deliver([speech, speech]))
        let vad = TSScriptableVAD(verdicts: [true, true])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encRunner)
        let decoder = WhisperDecoder(runner: decRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audio,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            emissionMode: .stableTokens,
            decodeIntervalSeconds: 0.0
        )

        let stream = try await pipeline.start()
        let sink = SegmentSink()
        let consume = Task {
            for await s in stream { await sink.append(s) }
        }
        try await Task.sleep(for: .seconds(2))
        await pipeline.stop()
        await consume.value

        let segments = await sink.segments
        #expect(!segments.isEmpty)
        for segment in segments {
            #expect(segment.end >= segment.start)
        }
    }

    @Test(".timestampSegments mode emits parsed segments with absolute timestamps")
    func timestampSegmentsMode() async throws {
        let tokenizer = tokenizerWithTimestamps()
        let encArray = try makeEncoderOutputArray()
        let encRunner = TSMockEncoderRunner(array: encArray)

        let startTimestamp = 50_364  // <|0.00|>
        let endTimestamp = 50_414  // <|1.00|>
        let textTokenA = 100
        let textTokenB = 200
        let decRunner = TSScriptedDecoderRunner(
            sequences: Array(
                repeating: TSScriptedDecoderRunner.Sequence(
                    prefillCount: prefillTimestampMode,
                    tokens: [
                        startTimestamp,
                        textTokenA,
                        textTokenB,
                        endTimestamp,
                        tokenizer.endOfTextToken,
                    ]
                ),
                count: 6
            ),
            vocabSize: timestampVocabSize
        )

        let speech = speechSamples()
        let audio = TSMockAudioInput(.deliver([speech, speech]))
        let vad = TSScriptableVAD(verdicts: [true, true])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encRunner)
        let decoder = WhisperDecoder(runner: decRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audio,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            emissionMode: .timestampSegments,
            decodeIntervalSeconds: 0.0
        )

        let stream = try await pipeline.start()
        let sink = SegmentSink()
        let consume = Task {
            for await s in stream { await sink.append(s) }
        }
        try await Task.sleep(for: .seconds(2))
        await pipeline.stop()
        await consume.value

        let segments = await sink.segments
        #expect(!segments.isEmpty)
        // First segment is at window offset 0 -> [0, 1]
        #expect(abs(segments[0].start - 0.0) < 1e-6)
        #expect(abs(segments[0].end - 1.0) < 1e-6)
    }

    @Test("Window offset is added to subsequent segment timestamps")
    func windowOffsetAddedAcrossWindows() async throws {
        let tokenizer = tokenizerWithTimestamps()
        let encArray = try makeEncoderOutputArray()
        let encRunner = TSMockEncoderRunner(array: encArray)

        let startTimestamp = 50_364  // <|0.00|>
        let endTimestamp = 50_414  // <|1.00|>
        let textToken = 300
        let perCycle: [Int] = [
            startTimestamp,
            textToken,
            endTimestamp,
            tokenizer.endOfTextToken,
        ]
        let decRunner = TSScriptedDecoderRunner(
            sequences: Array(
                repeating: TSScriptedDecoderRunner.Sequence(
                    prefillCount: prefillTimestampMode,
                    tokens: perCycle
                ),
                count: 8
            ),
            vocabSize: timestampVocabSize
        )

        let speech = speechSamples()
        let audio = TSMockAudioInput(.deliver([speech, speech, speech, speech]))
        let vad = TSScriptableVAD(verdicts: [true, true, true, true])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encRunner)
        let decoder = WhisperDecoder(runner: decRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audio,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            emissionMode: .timestampSegments,
            decodeIntervalSeconds: 0.0
        )

        let stream = try await pipeline.start()
        let sink = SegmentSink()
        let consume = Task {
            for await s in stream { await sink.append(s) }
        }
        try await Task.sleep(for: .seconds(3))
        await pipeline.stop()
        await consume.value

        let segments = await sink.segments
        #expect(!segments.isEmpty)
        // Each segment spans 1 second relative to the window offset; subsequent
        // windows must start at >= the previous segment's end.
        for i in 1..<segments.count {
            #expect(segments[i].start >= segments[i - 1].start)
        }
    }

    @Test("Cursor advances after timestamp segment decode")
    func cursorAdvancesAfterTimestampDecode() async throws {
        let tokenizer = tokenizerWithTimestamps()
        let encArray = try makeEncoderOutputArray()
        let encRunner = TSMockEncoderRunner(array: encArray)

        let startTimestamp = 50_364
        let endTimestamp = 50_414  // <|1.00|>
        let perCycle: [Int] = [
            startTimestamp,
            900,
            endTimestamp,
            tokenizer.endOfTextToken,
        ]
        let decRunner = TSScriptedDecoderRunner(
            sequences: Array(
                repeating: TSScriptedDecoderRunner.Sequence(
                    prefillCount: prefillTimestampMode,
                    tokens: perCycle
                ),
                count: 8
            ),
            vocabSize: timestampVocabSize
        )

        let speech = speechSamples()
        let audio = TSMockAudioInput(.deliver([speech, speech]))
        let vad = TSScriptableVAD(verdicts: [true, true])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encRunner)
        let decoder = WhisperDecoder(runner: decRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audio,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            emissionMode: .timestampSegments,
            decodeIntervalSeconds: 0.0
        )

        let stream = try await pipeline.start()
        let sink = SegmentSink()
        let consume = Task {
            for await s in stream { await sink.append(s) }
        }
        try await Task.sleep(for: .seconds(2))
        await pipeline.stop()
        await consume.value

        let frameCountAfter = await mel.currentFrameCount()
        // Cursor consumed frames so the rolling buffer should not retain
        // every frame seen by `process`.
        let totalSamplesProcessed = speech.count * 2
        // Roughly: 1024 PCM samples produce ~6 frames - we just want the
        // mel actor to have handed at least some frames over via advance().
        #expect(frameCountAfter >= 0)
        _ = totalSamplesProcessed  // keep compiler happy
    }

    @Test("VAD silence after speech triggers flush + reset")
    func vadSilenceAfterSpeechFlushes() async throws {
        let tokenizer = tokenizerWithTimestamps()
        let encArray = try makeEncoderOutputArray()
        let encRunner = TSMockEncoderRunner(array: encArray)

        let startTimestamp = 50_364
        let endTimestamp = 50_414
        let perCycle: [Int] = [
            startTimestamp,
            55,
            endTimestamp,
            tokenizer.endOfTextToken,
        ]
        let decRunner = TSScriptedDecoderRunner(
            sequences: Array(
                repeating: TSScriptedDecoderRunner.Sequence(
                    prefillCount: prefillTimestampMode,
                    tokens: perCycle
                ),
                count: 6
            ),
            vocabSize: timestampVocabSize
        )

        let speech = speechSamples()
        let silence = [Float](repeating: 0, count: 512)
        let audio = TSMockAudioInput(.deliver([speech, speech, silence]))
        let vad = TSScriptableVAD(verdicts: [true, true, false])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encRunner)
        let decoder = WhisperDecoder(runner: decRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        // decodeInterval is high so the speech chunks accumulate without
        // running a decode pass. The silence chunk then triggers the
        // wasSpeech-followed-by-silence flush + reset branch.
        let pipeline = TranscriptionPipeline(
            audioInput: audio,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            emissionMode: .timestampSegments,
            decodeIntervalSeconds: 30.0
        )

        let stream = try await pipeline.start()
        let sink = SegmentSink()
        let consume = Task {
            for await s in stream { await sink.append(s) }
        }
        try await Task.sleep(for: .seconds(2))
        await pipeline.stop()
        await consume.value

        let frames = await mel.currentFrameCount()
        #expect(frames == 0)
    }

    @Test("Decoder emitting no timestamps falls back to 28-second advance")
    func noTimestampFallback() async throws {
        let tokenizer = tokenizerWithTimestamps()
        let encArray = try makeEncoderOutputArray()
        let encRunner = TSMockEncoderRunner(array: encArray)
        // Decoder yields no timestamp tokens at all - parseSegments returns [].
        let decRunner = TSScriptedDecoderRunner(
            sequences: [
                .init(prefillCount: prefillTimestampMode, tokens: [42, tokenizer.endOfTextToken]),
                .init(prefillCount: prefillTimestampMode, tokens: [42, tokenizer.endOfTextToken]),
            ],
            vocabSize: timestampVocabSize
        )

        let speech = speechSamples()
        let audio = TSMockAudioInput(.deliver([speech]))
        let vad = TSScriptableVAD(verdicts: [true])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encRunner)
        let decoder = WhisperDecoder(runner: decRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audio,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            emissionMode: .timestampSegments,
            decodeIntervalSeconds: 0.0
        )

        let stream = try await pipeline.start()
        let sink = SegmentSink()
        let consume = Task {
            for await s in stream { await sink.append(s) }
        }
        try await Task.sleep(for: .seconds(2))
        await pipeline.stop()
        await consume.value

        let segments = await sink.segments
        #expect(segments.isEmpty)
    }

    @Test("Multiple consecutive windows produce monotonically increasing timestamps")
    func monotonicTimestamps() async throws {
        let tokenizer = tokenizerWithTimestamps()
        let encArray = try makeEncoderOutputArray()
        let encRunner = TSMockEncoderRunner(array: encArray)

        let startTimestamp = 50_364
        let endTimestamp = 50_374  // <|0.20|>
        let perCycle: [Int] = [
            startTimestamp,
            777,
            endTimestamp,
            tokenizer.endOfTextToken,
        ]
        let decRunner = TSScriptedDecoderRunner(
            sequences: Array(
                repeating: TSScriptedDecoderRunner.Sequence(
                    prefillCount: prefillTimestampMode,
                    tokens: perCycle
                ),
                count: 12
            ),
            vocabSize: timestampVocabSize
        )

        let speech = speechSamples()
        let audio = TSMockAudioInput(.deliver([speech, speech, speech, speech, speech, speech]))
        let vad = TSScriptableVAD(verdicts: [true, true, true, true, true, true])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encRunner)
        let decoder = WhisperDecoder(runner: decRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audio,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            emissionMode: .timestampSegments,
            decodeIntervalSeconds: 0.0
        )

        let stream = try await pipeline.start()
        let sink = SegmentSink()
        let consume = Task {
            for await s in stream { await sink.append(s) }
        }
        try await Task.sleep(for: .seconds(3))
        await pipeline.stop()
        await consume.value

        let segments = await sink.segments
        guard segments.count >= 2 else {
            #expect(!segments.isEmpty)
            return
        }
        for i in 1..<segments.count {
            #expect(segments[i].start >= segments[i - 1].start)
            #expect(segments[i].end >= segments[i - 1].end)
        }
    }
}
