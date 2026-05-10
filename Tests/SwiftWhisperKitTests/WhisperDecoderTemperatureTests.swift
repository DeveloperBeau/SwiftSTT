@preconcurrency import CoreML
import Foundation
import Synchronization
import Testing
@testable import SwiftWhisperKit
import SwiftWhisperCore

// MARK: - Local mock runner

private final class TempMockRunner: StatefulCoreMLModelRunner, @unchecked Sendable {

    struct Call: Sendable {
        let tokenId: Int32
        let cacheLength: Int32
    }

    private struct State {
        var resetCount: Int = 0
        var calls: [Call] = []
        var canned: [[Float]] = []
    }

    private let logitsName: String
    private let tokenInputName: String
    private let cacheLengthName: String
    private let prefillCount: Int
    private let originalQueue: [[Float]]
    private let defaultLogits: [Float]
    private let state = Mutex<State>(State())

    init(
        prefillCount: Int,
        logitsPerCall: [[Float]],
        defaultLogits: [Float],
        logitsName: String = WhisperDecoder.FeatureNames.default.logitsOutput,
        tokenInputName: String = WhisperDecoder.FeatureNames.default.tokenInput,
        cacheLengthName: String = WhisperDecoder.FeatureNames.default.cacheLength
    ) {
        self.logitsName = logitsName
        self.tokenInputName = tokenInputName
        self.cacheLengthName = cacheLengthName
        self.prefillCount = prefillCount
        self.originalQueue = logitsPerCall
        self.defaultLogits = defaultLogits
        self.state.withLock {
            $0.canned = logitsPerCall
        }
    }

    var resetCount: Int { state.withLock { $0.resetCount } }

    func resetState() async {
        state.withLock {
            $0.resetCount += 1
            $0.canned = originalQueue
        }
    }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        let token = features.featureValue(for: tokenInputName)?.multiArrayValue
        let cache = features.featureValue(for: cacheLengthName)?.multiArrayValue
        let tokenId = token.map { $0[[0, 0] as [NSNumber]].int32Value } ?? -1
        let cacheLength = cache.map { $0[[0] as [NSNumber]].int32Value } ?? -1

        let logits: [Float] = state.withLock { state in
            state.calls.append(Call(tokenId: tokenId, cacheLength: cacheLength))
            if state.calls.count <= prefillCount {
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

private let defaultSpecials: [String: Int] = [
    "<|endoftext|>": 50_257,
    "<|startoftranscript|>": 50_258,
    "<|en|>": 50_259,
    "<|translate|>": 50_358,
    "<|transcribe|>": 50_359,
    "<|notimestamps|>": 50_363,
]

private func makeTokenizer() -> WhisperTokenizer {
    WhisperTokenizer(specialTokens: defaultSpecials)
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

private func twoPeakLogits(vocabSize: Int, hotA: Int, hotB: Int, valueA: Float = 1.0, valueB: Float = 1.0) -> [Float] {
    var out = [Float](repeating: -10, count: vocabSize)
    out[hotA] = valueA
    out[hotB] = valueB
    return out
}

private func oneHotLogits(vocabSize: Int, hot: Int, value: Float = 10) -> [Float] {
    var out = [Float](repeating: 0, count: vocabSize)
    out[hot] = value
    return out
}

@Suite("WhisperDecoder temperature sampling")
struct WhisperDecoderTemperatureTests {

    private let vocabSize = 50_400

    @Test("Temperature 0 matches greedy output")
    func temperatureZeroMatchesGreedy() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.temperature = 0.0
        options.temperatureFallback = [0.0]

        let queue: [[Float]] = [
            oneHotLogits(vocabSize: vocabSize, hot: 11),
            oneHotLogits(vocabSize: vocabSize, hot: 22),
            oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
        ]
        let runner = TempMockRunner(
            prefillCount: 3,
            logitsPerCall: queue,
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer, rng: SeededRandom(seed: 42))
        let encoder = try makeEncoderArray()

        let result = try await decoder.decode(encoderOutput: encoder, options: options)
        #expect(result.map(\.id) == [11, 22])
    }

    @Test("Sampling with seeded RNG is deterministic")
    func sampledOutputIsDeterministic() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.temperature = 0.5
        options.temperatureFallback = [0.5]

        let queue: [[Float]] = [
            twoPeakLogits(vocabSize: vocabSize, hotA: 100, hotB: 200, valueA: 0.5, valueB: 1.0),
            twoPeakLogits(vocabSize: vocabSize, hotA: 300, hotB: 400, valueA: 1.0, valueB: 0.5),
            oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
        ]

        func runOnce() async throws -> [Int] {
            let runner = TempMockRunner(
                prefillCount: 3,
                logitsPerCall: queue,
                defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
            )
            let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer, rng: SeededRandom(seed: 1234))
            let encoder = try makeEncoderArray()
            let result = try await decoder.decode(encoderOutput: encoder, options: options)
            return result.map(\.id)
        }

        let a = try await runOnce()
        let b = try await runOnce()
        #expect(a == b)
        #expect(a.count >= 1)
    }

    @Test("softmax sums to ~1")
    func softmaxSumsToOne() {
        let probs = WhisperDecoder.softmax(logits: [1, 2, 3, 4, 5], temperature: 1.0)
        let total = probs.reduce(0, +)
        #expect(abs(total - 1.0) < 0.0001)
    }

    @Test("Lower temperature is sharper")
    func lowerTemperatureIsSharper() {
        let logits: [Float] = [1, 2, 3]
        let sharp = WhisperDecoder.softmax(logits: logits, temperature: 0.1)
        let soft = WhisperDecoder.softmax(logits: logits, temperature: 1.0)
        #expect((sharp.max() ?? 0) > (soft.max() ?? 0))
    }

    @Test("softmax handles all-suppressed input as uniform")
    func softmaxAllInfiniteFallback() {
        let probs = WhisperDecoder.softmax(logits: [-Float.infinity, -Float.infinity, -Float.infinity], temperature: 1.0)
        #expect(probs.count == 3)
        for p in probs {
            #expect(abs(p - 1.0 / 3.0) < 0.0001)
        }
    }

    @Test("softmax of empty input is empty")
    func softmaxEmpty() {
        #expect(WhisperDecoder.softmax(logits: [], temperature: 1.0).isEmpty)
    }

    @Test("softmax with temperature 0 falls back without dividing by zero")
    func softmaxZeroTemperature() {
        let probs = WhisperDecoder.softmax(logits: [1, 2, 3], temperature: 0)
        #expect(probs.count == 3)
        let total = probs.reduce(0, +)
        #expect(abs(total - 1.0) < 0.001)
    }

    @Test("sample respects probability distribution")
    func sampleRespectsDistribution() {
        let probs: [Float] = [0.7, 0.2, 0.1]
        var rng = SeededRandom(seed: 42)
        var counts = [0, 0, 0]
        let trials = 30_000
        for _ in 0..<trials {
            let pick = WhisperDecoder.sample(probs: probs, rng: &rng)
            counts[pick] += 1
        }
        let frequencies = counts.map { Float($0) / Float(trials) }
        #expect(abs(frequencies[0] - 0.7) < 0.05)
        #expect(abs(frequencies[1] - 0.2) < 0.05)
        #expect(abs(frequencies[2] - 0.1) < 0.05)
    }

    @Test("sample on empty distribution returns 0")
    func sampleEmpty() {
        var rng = SeededRandom(seed: 1)
        #expect(WhisperDecoder.sample(probs: [], rng: &rng) == 0)
    }

    @Test("Negative temperature throws invalidDecodingOption")
    func negativeTemperatureThrows() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.temperature = -0.5
        let runner = TempMockRunner(
            prefillCount: 0,
            logitsPerCall: [],
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        do {
            _ = try await decoder.decode(encoderOutput: encoder, options: options)
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            if case .invalidDecodingOption = error {} else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    @Test("SystemRandom returns distinct values")
    func systemRandomYieldsDifferentValues() {
        var rng = SystemRandom()
        let a = rng.next()
        let b = rng.next()
        #expect(a != b)
    }

    @Test("SeededRandom is deterministic")
    func seededRandomIsDeterministic() {
        var a = SeededRandom(seed: 99)
        var b = SeededRandom(seed: 99)
        for _ in 0..<10 {
            #expect(a.next() == b.next())
        }
    }
}
