@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Synchronization
import Testing

@testable import SwiftWhisperKit

// MARK: - Mock audio input

/// Delivers canned buffers synchronously during `start()` then returns.
/// The chunk stream finishes when the pipeline calls `stop()`.
private actor MockAudioInput: AudioInputProvider {

    enum Behaviour {
        case deliver([[Float]])
        case throwError(SwiftWhisperError)
    }

    private let behaviour: Behaviour
    private(set) var startCount = 0
    private(set) var stopCount = 0

    init(_ behaviour: Behaviour) {
        self.behaviour = behaviour
    }

    func start(
        targetSampleRate: Double,
        bufferDurationSeconds: Double,
        onChunk: @Sendable @escaping ([Float]) -> Void
    ) async throws(SwiftWhisperError) {
        startCount += 1
        switch behaviour {
        case .throwError(let err):
            throw err
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

// MARK: - Scriptable VAD

private actor ScriptableVAD: VoiceActivityDetector {

    private var verdicts: [Bool]
    private var index = 0

    init(verdicts: [Bool]) {
        self.verdicts = verdicts
    }

    func isSpeech(chunk: AudioChunk) async -> Bool {
        guard index < verdicts.count else { return false }
        let result = verdicts[index]
        index += 1
        return result
    }

    func reset() async {
        index = 0
    }
}

// MARK: - Mock CoreML runner

private final class MockRunner: CoreMLModelRunner, @unchecked Sendable {

    private struct State {
        var callCount: Int = 0
    }

    enum Behaviour {
        case returnArray(MLMultiArray)
        case throwError(SwiftWhisperError)
    }

    private let behaviour: Behaviour
    private let state = Mutex<State>(State())

    init(_ behaviour: Behaviour) {
        self.behaviour = behaviour
    }

    var callCount: Int { state.withLock { $0.callCount } }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        state.withLock { $0.callCount += 1 }
        switch behaviour {
        case .throwError(let err):
            throw err
        case .returnArray(let array):
            do {
                return try MLDictionaryFeatureProvider(dictionary: [
                    WhisperEncoder.outputFeatureName: array
                ])
            } catch {
                throw .decoderFailure("mock provider: \(error.localizedDescription)")
            }
        }
    }
}

// MARK: - Resettable mock stateful runner

/// Resets its generation queue on each `resetState()` call so multiple decode
/// cycles produce the same token sequence. The agreement policy requires
/// identical output from successive decode passes.
private final class ResettableMockStatefulRunner: StatefulCoreMLModelRunner, @unchecked Sendable {

    private struct State {
        var resetCount: Int = 0
        var remainingPrefill: Int
        var generationQueue: [[Float]]
    }

    private let prefillCount: Int
    private let generationLogits: [[Float]]
    private let defaultLogits: [Float]
    private let logitsName: String
    private let state: Mutex<State>

    init(
        prefillCount: Int,
        generationLogits: [[Float]],
        defaultLogits: [Float],
        logitsName: String = WhisperDecoder.FeatureNames.default.logitsOutput
    ) {
        self.prefillCount = prefillCount
        self.generationLogits = generationLogits
        self.defaultLogits = defaultLogits
        self.logitsName = logitsName
        self.state = Mutex(
            State(
                resetCount: 0,
                remainingPrefill: prefillCount,
                generationQueue: generationLogits
            ))
    }

    var resetCount: Int { state.withLock { $0.resetCount } }

    func resetState() async {
        state.withLock {
            $0.resetCount += 1
            $0.remainingPrefill = prefillCount
            $0.generationQueue = generationLogits
        }
    }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        let logits: [Float] = state.withLock { state in
            if state.remainingPrefill > 0 {
                state.remainingPrefill -= 1
                return defaultLogits
            }
            if !state.generationQueue.isEmpty {
                return state.generationQueue.removeFirst()
            }
            return defaultLogits
        }

        do {
            let array = try MLMultiArray(
                shape: [1, 1, NSNumber(value: logits.count)],
                dataType: .float32
            )
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
            for i in 0..<logits.count {
                pointer[i] = logits[i]
            }
            return try MLDictionaryFeatureProvider(dictionary: [logitsName: array])
        } catch {
            throw .decoderFailure("mock provider: \(error.localizedDescription)")
        }
    }
}

// MARK: - Mock stateful runner (single-use)

private final class MockStatefulRunner: StatefulCoreMLModelRunner, @unchecked Sendable {

    enum Behaviour {
        case canned(prefillCount: Int, logitsPerCall: [[Float]], defaultLogits: [Float])
        case throwError(SwiftWhisperError)
    }

    private struct State {
        var resetCount: Int = 0
        var canned: [[Float]] = []
        var remainingPrefill: Int = 0
    }

    private let behaviour: Behaviour
    private let logitsName: String
    private let defaultLogits: [Float]
    private let state = Mutex<State>(State())

    init(
        behaviour: Behaviour,
        logitsName: String = WhisperDecoder.FeatureNames.default.logitsOutput
    ) {
        self.behaviour = behaviour
        self.logitsName = logitsName

        switch behaviour {
        case .canned(let prefill, let queue, let fallback):
            self.defaultLogits = fallback
            self.state.withLock {
                $0.canned = queue
                $0.remainingPrefill = prefill
            }
        case .throwError:
            self.defaultLogits = []
        }
    }

    func resetState() async {
        state.withLock { $0.resetCount += 1 }
    }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        let logits: [Float] = state.withLock { state in
            if state.remainingPrefill > 0 {
                state.remainingPrefill -= 1
                return defaultLogits
            }
            if !state.canned.isEmpty {
                return state.canned.removeFirst()
            }
            return defaultLogits
        }

        switch behaviour {
        case .throwError(let error):
            throw error
        case .canned:
            do {
                let array = try MLMultiArray(
                    shape: [1, 1, NSNumber(value: logits.count)],
                    dataType: .float32
                )
                let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: logits.count)
                for i in 0..<logits.count {
                    pointer[i] = logits[i]
                }
                return try MLDictionaryFeatureProvider(dictionary: [logitsName: array])
            } catch {
                throw .decoderFailure("mock provider: \(error.localizedDescription)")
            }
        }
    }
}

// MARK: - Helpers

private let vocabSize = 50_400

private let defaultSpecials: [String: Int] = [
    "<|endoftext|>": 50_257,
    "<|startoftranscript|>": 50_258,
    "<|en|>": 50_259,
    "<|transcribe|>": 50_359,
    "<|translate|>": 50_358,
    "<|notimestamps|>": 50_363,
]

private func makeTokenizer() -> WhisperTokenizer {
    WhisperTokenizer(specialTokens: defaultSpecials)
}

private func oneHotLogits(vocabSize: Int, hot: Int, value: Float = 10) -> [Float] {
    var out = [Float](repeating: 0, count: vocabSize)
    out[hot] = value
    return out
}

private func makeEncoderOutputArray() throws -> MLMultiArray {
    let array = try MLMultiArray(shape: [1, 1500, 16], dataType: .float32)
    let count = 1500 * 16
    let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
    for i in 0..<count { ptr[i] = 0 }
    return array
}

/// Generates enough PCM samples to produce at least one mel frame.
private func makeSpeechSamples(count: Int = 512) -> [Float] {
    (0..<count).map { Float(sin(Double($0) * 0.1)) * 0.5 }
}

/// Actor-isolated segment collector. Receives segments from a stream
/// concurrently and stores them for later assertion.
private actor SegmentCollector {
    var segments: [TranscriptionSegment] = []

    func append(_ segment: TranscriptionSegment) {
        segments.append(segment)
    }
}

// MARK: - Tests

@Suite("TranscriptionPipeline")
struct TranscriptionPipelineTests {

    @Test("Pipeline emits segments from canned audio with mock runners")
    func emitsSegments() async throws {
        let tokenizer = makeTokenizer()
        let encoderOutput = try makeEncoderOutputArray()
        let encoderRunner = MockRunner(.returnArray(encoderOutput))

        let decoderRunner = ResettableMockStatefulRunner(
            prefillCount: 3,
            generationLogits: [
                oneHotLogits(vocabSize: vocabSize, hot: 100),
                oneHotLogits(vocabSize: vocabSize, hot: 200),
                oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
            ],
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )

        let speech = makeSpeechSamples()
        let audioInput = MockAudioInput(.deliver([speech, speech]))
        let vad = ScriptableVAD(verdicts: [true, true])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encoderRunner)
        let decoder = WhisperDecoder(runner: decoderRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audioInput,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            decodeIntervalSeconds: 0.0
        )

        let stream = try await pipeline.start()
        let collector = SegmentCollector()

        let consumeTask = Task {
            for await segment in stream {
                await collector.append(segment)
            }
        }

        // Allow time for async processing, then stop to unblock the stream
        try await Task.sleep(for: .seconds(3))
        await pipeline.stop()
        await consumeTask.value

        let segments = await collector.segments
        #expect(!segments.isEmpty)
    }

    @Test("VAD silence skips encoding")
    func vadSilenceSkipsEncoding() async throws {
        let tokenizer = makeTokenizer()
        let encoderOutput = try makeEncoderOutputArray()
        let encoderRunner = MockRunner(.returnArray(encoderOutput))

        let decoderRunner = ResettableMockStatefulRunner(
            prefillCount: 3,
            generationLogits: [],
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )

        let silence = [Float](repeating: 0, count: 512)
        let audioInput = MockAudioInput(.deliver([silence, silence, silence]))
        let vad = ScriptableVAD(verdicts: [false, false, false])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encoderRunner)
        let decoder = WhisperDecoder(runner: decoderRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audioInput,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            decodeIntervalSeconds: 0.0
        )

        let stream = try await pipeline.start()
        let collector = SegmentCollector()

        let consumeTask = Task {
            for await segment in stream {
                await collector.append(segment)
            }
        }

        try await Task.sleep(for: .seconds(1))
        await pipeline.stop()
        await consumeTask.value

        let segments = await collector.segments
        #expect(segments.isEmpty)
        #expect(encoderRunner.callCount == 0)
    }

    @Test("stop() finishes the stream cleanly")
    func stopFinishesStream() async throws {
        let tokenizer = makeTokenizer()
        let encoderOutput = try makeEncoderOutputArray()
        let encoderRunner = MockRunner(.returnArray(encoderOutput))

        let decoderRunner = ResettableMockStatefulRunner(
            prefillCount: 3,
            generationLogits: [],
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )

        let audioInput = MockAudioInput(.deliver([]))
        let vad = ScriptableVAD(verdicts: [])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encoderRunner)
        let decoder = WhisperDecoder(runner: decoderRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audioInput,
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
        for await _ in stream {
            count += 1
        }
        #expect(count == 0)
        await #expect(audioInput.stopCount >= 1)
    }

    @Test("Encoder error terminates the stream")
    func encoderErrorTerminatesStream() async throws {
        let tokenizer = makeTokenizer()
        let encoderRunner = MockRunner(.throwError(.decoderFailure("encoder boom")))

        let decoderRunner = ResettableMockStatefulRunner(
            prefillCount: 3,
            generationLogits: [],
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )

        let speech = makeSpeechSamples()
        let audioInput = MockAudioInput(.deliver([speech]))
        let vad = ScriptableVAD(verdicts: [true])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encoderRunner)
        let decoder = WhisperDecoder(runner: decoderRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audioInput,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            decodeIntervalSeconds: 0.0
        )

        let stream = try await pipeline.start()
        let collector = SegmentCollector()

        let consumeTask = Task {
            for await segment in stream {
                await collector.append(segment)
            }
        }

        try await Task.sleep(for: .seconds(2))
        await pipeline.stop()
        await consumeTask.value

        let segments = await collector.segments
        #expect(segments.isEmpty)
    }

    @Test("LocalAgreement produces stable output across two decode runs")
    func localAgreementIntegration() async throws {
        let tokenizer = makeTokenizer()
        let encoderOutput = try makeEncoderOutputArray()
        let encoderRunner = MockRunner(.returnArray(encoderOutput))

        let decoderRunner = ResettableMockStatefulRunner(
            prefillCount: 3,
            generationLogits: [
                oneHotLogits(vocabSize: vocabSize, hot: 100),
                oneHotLogits(vocabSize: vocabSize, hot: 200),
                oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
            ],
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )

        let speech1 = makeSpeechSamples(count: 512)
        let speech2 = makeSpeechSamples(count: 512)
        let audioInput = MockAudioInput(.deliver([speech1, speech2]))
        let vad = ScriptableVAD(verdicts: [true, true])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encoderRunner)
        let decoder = WhisperDecoder(runner: decoderRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audioInput,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            decodeIntervalSeconds: 0.0
        )

        let stream = try await pipeline.start()
        let collector = SegmentCollector()

        let consumeTask = Task {
            for await segment in stream {
                await collector.append(segment)
            }
        }

        try await Task.sleep(for: .seconds(3))
        await pipeline.stop()
        await consumeTask.value

        let segments = await collector.segments
        // Two decode runs with the same tokens confirm through agreement
        #expect(!segments.isEmpty)
    }

    @Test("stop() mid-stream cleans up and stops audio input")
    func stopMidStreamCleansUp() async throws {
        let tokenizer = makeTokenizer()
        let encoderOutput = try makeEncoderOutputArray()
        let encoderRunner = MockRunner(.returnArray(encoderOutput))

        let decoderRunner = ResettableMockStatefulRunner(
            prefillCount: 3,
            generationLogits: [],
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )

        let audioInput = MockAudioInput(.deliver([]))
        let vad = ScriptableVAD(verdicts: [])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encoderRunner)
        let decoder = WhisperDecoder(runner: decoderRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audioInput,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            decodeIntervalSeconds: 0.0
        )

        _ = try await pipeline.start()
        await pipeline.stop()
        // Double stop is safe
        await pipeline.stop()
        await #expect(audioInput.stopCount >= 1)
    }

    @Test("Double start returns finished stream on second call")
    func doubleStart() async throws {
        let tokenizer = makeTokenizer()
        let encoderOutput = try makeEncoderOutputArray()
        let encoderRunner = MockRunner(.returnArray(encoderOutput))

        let decoderRunner = ResettableMockStatefulRunner(
            prefillCount: 3,
            generationLogits: [],
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )

        let audioInput = MockAudioInput(.deliver([]))
        let vad = ScriptableVAD(verdicts: [])
        let mel = try MelSpectrogram()
        let encoder = WhisperEncoder(runner: encoderRunner)
        let decoder = WhisperDecoder(runner: decoderRunner, tokenizer: tokenizer)
        let policy = LocalAgreementPolicy()

        let pipeline = TranscriptionPipeline(
            audioInput: audioInput,
            vad: vad,
            melSpectrogram: mel,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            decodeIntervalSeconds: 0.0
        )

        _ = try await pipeline.start()

        let secondStream = try await pipeline.start()
        var count = 0
        for await _ in secondStream {
            count += 1
        }
        #expect(count == 0)

        await pipeline.stop()
    }
}
