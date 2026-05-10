@preconcurrency import CoreML
import Foundation
import Synchronization
import Testing
@testable import SwiftWhisperKit
import SwiftWhisperCore

// MARK: - Mock runner

/// Mock that replays the same canned logits queue for every fresh attempt.
/// The decoder's temperature-fallback loop calls `resetState()` between
/// attempts, so the mock rewinds its queue each time.
private final class FallbackMockRunner: StatefulCoreMLModelRunner, @unchecked Sendable {

    private struct State {
        var canned: [[Float]] = []
        var resetCount: Int = 0
        var attemptCount: Int = 0
        var calls: Int = 0
    }

    private let logitsName: String
    private let tokenInputName: String
    private let cacheLengthName: String
    private let perAttemptQueues: [[[Float]]]
    private let defaultLogits: [Float]
    private let prefillCount: Int
    private let state = Mutex<State>(State())

    init(
        prefillCount: Int,
        perAttemptQueues: [[[Float]]],
        defaultLogits: [Float],
        logitsName: String = WhisperDecoder.FeatureNames.default.logitsOutput,
        tokenInputName: String = WhisperDecoder.FeatureNames.default.tokenInput,
        cacheLengthName: String = WhisperDecoder.FeatureNames.default.cacheLength
    ) {
        self.logitsName = logitsName
        self.tokenInputName = tokenInputName
        self.cacheLengthName = cacheLengthName
        self.prefillCount = prefillCount
        self.perAttemptQueues = perAttemptQueues
        self.defaultLogits = defaultLogits
        self.state.withLock {
            $0.canned = perAttemptQueues.first ?? []
        }
    }

    var resetCount: Int { state.withLock { $0.resetCount } }
    var attemptCount: Int { state.withLock { $0.attemptCount } }

    func resetState() async {
        state.withLock { state in
            state.resetCount += 1
            let nextIndex = state.attemptCount
            state.attemptCount += 1
            state.calls = 0
            if nextIndex < perAttemptQueues.count {
                state.canned = perAttemptQueues[nextIndex]
            } else if let last = perAttemptQueues.last {
                state.canned = last
            } else {
                state.canned = []
            }
        }
    }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        let logits: [Float] = state.withLock { state in
            state.calls += 1
            if state.calls <= prefillCount {
                return defaultLogits
            }
            if !state.canned.isEmpty {
                return state.canned.removeFirst()
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

// MARK: - Helpers

private let antiHallucinationSpecials: [String: Int] = [
    "<|endoftext|>": 50_257,
    "<|startoftranscript|>": 50_258,
    "<|nospeech|>": 50_362,
    "<|notimestamps|>": 50_363,
    "<|transcribe|>": 50_359,
    "<|translate|>": 50_358,
]

private func makeTokenizer() -> WhisperTokenizer {
    WhisperTokenizer(specialTokens: antiHallucinationSpecials)
}

private func makeEncoderArray(shape: [Int] = [1, 1500, 16], fill: Float = 0.0) throws -> MLMultiArray {
    let array = try MLMultiArray(
        shape: shape.map { NSNumber(value: $0) },
        dataType: .float32
    )
    let count = shape.reduce(1, *)
    let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: count)
    for i in 0..<count {
        pointer[i] = fill
    }
    return array
}

private func confidentLogits(vocabSize: Int, hot: Int, value: Float = 30) -> [Float] {
    var out = [Float](repeating: -10, count: vocabSize)
    out[hot] = value
    return out
}

/// Gentle distribution where the "winner" beats the field by only ~2 nats.
/// Designed so `logSoftmax(winner)` lands near `-1.5` and trips strict
/// log-probability thresholds in fallback tests.
private func murkyLogits(vocabSize: Int, hot: Int) -> [Float] {
    var out = [Float](repeating: 0, count: vocabSize)
    out[hot] = 2
    return out
}

private func noSpeechLogits(vocabSize: Int, noSpeechToken: Int) -> [Float] {
    var out = [Float](repeating: -20, count: vocabSize)
    out[noSpeechToken] = 30
    return out
}

@Suite("WhisperDecoder anti-hallucination")
struct WhisperDecoderAntiHallucinationTests {

    private let vocabSize = 50_400

    @Test("compressionRatio of empty text is zero")
    func compressionRatioEmpty() {
        #expect(WhisperDecoder.compressionRatio(text: "") == 0)
    }

    @Test("compressionRatio of healthy text stays under threshold")
    func compressionRatioHealthy() {
        let ratio = WhisperDecoder.compressionRatio(text: "the quick brown fox jumps over the lazy dog")
        #expect(ratio < 2.4)
    }

    @Test("compressionRatio of repetitive text exceeds threshold")
    func compressionRatioRepetitive() {
        let repetitive = String(repeating: "the ", count: 50)
        let ratio = WhisperDecoder.compressionRatio(text: repetitive)
        #expect(ratio > 2.4)
    }

    @Test("compressionRatio of single-character input is the length")
    func compressionRatioSingleChar() {
        let ratio = WhisperDecoder.compressionRatio(text: "aaaaaa")
        #expect(ratio == 6)
    }

    @Test("noSpeechProbability returns zero for out-of-range token id")
    func noSpeechProbabilityOutOfRange() {
        let logits: [Float] = [1, 2, 3]
        #expect(WhisperDecoder.noSpeechProbability(logits: logits, noSpeechToken: 999) == 0)
        #expect(WhisperDecoder.noSpeechProbability(logits: logits, noSpeechToken: -1) == 0)
    }

    @Test("noSpeechProbability returns the softmax value at the no-speech slot")
    func noSpeechProbabilityHappy() {
        var logits = [Float](repeating: -20, count: 5)
        logits[2] = 5
        let prob = WhisperDecoder.noSpeechProbability(logits: logits, noSpeechToken: 2)
        #expect(prob > 0.99)
    }

    @Test("DecodeAttempt passes thresholds when both checks succeed")
    func attemptPassesWhenHealthy() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.temperatureFallback = [0.0]

        let healthyAttempt: [[Float]] = [
            confidentLogits(vocabSize: vocabSize, hot: 100),
            confidentLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
        ]
        let runner = FallbackMockRunner(
            prefillCount: 3,
            perAttemptQueues: [healthyAttempt],
            defaultLogits: confidentLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        let result = try await decoder.decode(encoderOutput: encoder, options: options)
        #expect(result.map(\.id) == [100])
        #expect(runner.attemptCount == 1)
    }

    @Test("Fallback retries when the first attempt fails the log-prob threshold")
    func fallbackRetriesOnLogProbFailure() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.temperatureFallback = [0.0, 0.2]
        // Murky logits give an avgLogProb well below 0; threshold above that
        // forces a fallback retry on the second temperature.
        options.logProbThreshold = -0.5

        let degenerate: [[Float]] = [
            murkyLogits(vocabSize: vocabSize, hot: 100),
            murkyLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
        ]
        let runner = FallbackMockRunner(
            prefillCount: 3,
            perAttemptQueues: [degenerate, degenerate],
            defaultLogits: murkyLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        _ = try await decoder.decode(encoderOutput: encoder, options: options)
        #expect(runner.attemptCount == 2)
    }

    @Test("Skip rule returns empty tokens when no-speech and degenerate")
    func skipRuleReturnsEmpty() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.suppressTokens = [tokenizer.noSpeechToken]  // Whisper masks the no-speech token from emission.
        options.temperatureFallback = [0.0]
        options.noSpeechThreshold = 0.5
        // Murky logits land avgLogProb well below 0 so the skip rule fires
        // alongside the high noSpeech probability.
        options.logProbThreshold = -0.5

        let queue: [[Float]] = [
            noSpeechLogits(vocabSize: vocabSize, noSpeechToken: tokenizer.noSpeechToken),
            murkyLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
        ]
        let runner = FallbackMockRunner(
            prefillCount: 3,
            perAttemptQueues: [queue],
            defaultLogits: murkyLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        let result = try await decoder.decode(encoderOutput: encoder, options: options)
        #expect(result.isEmpty)
    }

    @Test("All fallback temps fail thresholds: returns last attempt's tokens")
    func returnsLastAttemptIfAllFail() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.temperatureFallback = [0.0, 0.2]
        options.logProbThreshold = 100  // Impossible threshold.

        let degenerate: [[Float]] = [
            confidentLogits(vocabSize: vocabSize, hot: 42),
            confidentLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
        ]
        let runner = FallbackMockRunner(
            prefillCount: 3,
            perAttemptQueues: [degenerate, degenerate],
            defaultLogits: confidentLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        let result = try await decoder.decode(encoderOutput: encoder, options: options)
        // Without the no-speech trigger, fallback exhaustion still returns
        // the final attempt's tokens.
        #expect(result.map(\.id) == [42])
    }

    @Test("Validate rejects empty temperatureFallback")
    func validateRejectsEmptyFallback() throws {
        var options = DecodingOptions.default
        options.temperatureFallback = []
        do {
            try WhisperDecoder.validate(options: options)
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            if case .invalidDecodingOption = error {} else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    @Test("Validate rejects negative temperatureFallback entry")
    func validateRejectsNegativeFallback() throws {
        var options = DecodingOptions.default
        options.temperatureFallback = [0.0, -0.5]
        do {
            try WhisperDecoder.validate(options: options)
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            if case .invalidDecodingOption = error {} else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    @Test("Default decoding options expose the documented threshold values")
    func defaultThresholdsMatchSpec() {
        let options = DecodingOptions.default
        #expect(options.logProbThreshold == -1.0)
        #expect(options.noSpeechThreshold == 0.6)
        #expect(options.compressionRatioThreshold == 2.4)
        #expect(options.topP == 1.0)
        #expect(options.repetitionPenalty == 1.0)
        #expect(options.temperatureFallback == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        #expect(options.logitBias.isEmpty)
    }
}
