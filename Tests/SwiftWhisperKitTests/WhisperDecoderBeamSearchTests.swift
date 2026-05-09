@preconcurrency import CoreML
import Foundation
import Synchronization
import Testing
@testable import SwiftWhisperKit
import SwiftWhisperCore

// MARK: - Beam-aware mock runner

/// Mock that scores the next token by inspecting the previously fed token id.
/// Beam search resets and re-prefills repeatedly, so the runner has to be
/// stateless across resets but maintain a record of the most recent input.
private final class BeamMockRunner: StatefulCoreMLModelRunner, @unchecked Sendable {

    /// `tokenScores[lastTokenId]` returns logits to produce on the next call.
    /// Calls whose `lastTokenId` is missing fall back to `defaultLogits`.
    typealias ScoreTable = [Int: [Float]]

    private struct State {
        var resetCount: Int = 0
        var lastSeenToken: Int? = nil
        var totalCalls: Int = 0
    }

    private let scoreTable: ScoreTable
    private let defaultLogits: [Float]
    private let logitsName: String
    private let tokenInputName: String
    private let cacheLengthName: String
    private let state = Mutex<State>(State())

    init(
        scoreTable: ScoreTable,
        defaultLogits: [Float],
        logitsName: String = WhisperDecoder.FeatureNames.default.logitsOutput,
        tokenInputName: String = WhisperDecoder.FeatureNames.default.tokenInput,
        cacheLengthName: String = WhisperDecoder.FeatureNames.default.cacheLength
    ) {
        self.scoreTable = scoreTable
        self.defaultLogits = defaultLogits
        self.logitsName = logitsName
        self.tokenInputName = tokenInputName
        self.cacheLengthName = cacheLengthName
    }

    var resetCount: Int { state.withLock { $0.resetCount } }
    var totalCalls: Int { state.withLock { $0.totalCalls } }

    func resetState() async {
        state.withLock {
            $0.resetCount += 1
            $0.lastSeenToken = nil
        }
    }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        let token = features.featureValue(for: tokenInputName)?.multiArrayValue
        let tokenId = token.map { Int($0[[0, 0] as [NSNumber]].int32Value) } ?? -1

        let logits: [Float] = state.withLock { state in
            state.totalCalls += 1
            let previous = state.lastSeenToken ?? tokenId
            state.lastSeenToken = tokenId
            return scoreTable[previous] ?? defaultLogits
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

private func oneHotLogits(vocabSize: Int, hot: Int, value: Float = 10) -> [Float] {
    var out = [Float](repeating: 0, count: vocabSize)
    out[hot] = value
    return out
}

@Suite("WhisperDecoder beam search")
struct WhisperDecoderBeamSearchTests {

    private let vocabSize = 50_400

    @Test("beamSize=1 falls through to greedy path")
    func beamSizeOneIsGreedy() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.beamSize = 1

        let table: BeamMockRunner.ScoreTable = [:]
        let runner = BeamMockRunner(
            scoreTable: table,
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        _ = try await decoder.decode(encoderOutput: encoder, options: options)
        #expect(runner.resetCount == 1)
    }

    @Test("beamSize=2 explores hypotheses and returns best")
    func beamSizeTwoExploresHypotheses() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.beamSize = 2

        // After the prompt, token 100 is best, then 200. Either path leads to
        // EOT on the next step. Beam should keep both and pick the higher
        // log-prob path (token 100, then EOT).
        let promptTail = tokenizer.noTimestampsToken
        var firstStep = oneHotLogits(vocabSize: vocabSize, hot: 0, value: -10)
        firstStep[100] = 5
        firstStep[200] = 4
        var continueA = oneHotLogits(vocabSize: vocabSize, hot: 0, value: -10)
        continueA[tokenizer.endOfTextToken] = 8
        var continueB = oneHotLogits(vocabSize: vocabSize, hot: 0, value: -10)
        continueB[tokenizer.endOfTextToken] = 6

        let table: BeamMockRunner.ScoreTable = [
            promptTail: firstStep,
            100: continueA,
            200: continueB,
        ]
        let runner = BeamMockRunner(
            scoreTable: table,
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        let result = try await decoder.decode(encoderOutput: encoder, options: options)
        #expect(result.first?.id == 100)
    }

    @Test("selectTopK returns k highest entries in score order")
    func selectTopKBasic() {
        let result = WhisperDecoder.selectTopK(logits: [0.1, 0.5, 0.3, 0.9, 0.2], k: 3)
        let ids = result.map(\.token)
        let scores = result.map(\.score)
        #expect(ids == [3, 1, 2])
        #expect(scores == [0.9, 0.5, 0.3])
    }

    @Test("selectTopK with k > vocab returns all entries")
    func selectTopKWithKGreaterThanVocab() {
        let result = WhisperDecoder.selectTopK(logits: [1, 2, 3], k: 99)
        #expect(result.count == 3)
        #expect(result.map(\.token) == [2, 1, 0])
    }

    @Test("selectTopK with empty logits returns empty")
    func selectTopKEmpty() {
        #expect(WhisperDecoder.selectTopK(logits: [], k: 5).isEmpty)
    }

    @Test("selectTopK with k <= 0 returns empty")
    func selectTopKZeroK() {
        #expect(WhisperDecoder.selectTopK(logits: [1, 2, 3], k: 0).isEmpty)
    }

    @Test("selectTopK breaks ties by lowest index")
    func selectTopKTieBreak() {
        let result = WhisperDecoder.selectTopK(logits: [3, 3, 3, 3], k: 2)
        #expect(result.map(\.token) == [0, 1])
    }

    @Test("Beam search end-to-end produces expected sequence")
    func beamEndToEndProgrammed() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.beamSize = 2

        // Make beam token 100 win; then on its second step prefer 50.
        var stepFromPrompt = oneHotLogits(vocabSize: vocabSize, hot: 0, value: -20)
        stepFromPrompt[100] = 10
        stepFromPrompt[200] = 1

        var stepFrom100 = oneHotLogits(vocabSize: vocabSize, hot: 0, value: -20)
        stepFrom100[50] = 10
        stepFrom100[60] = 1

        var stepFrom50 = oneHotLogits(vocabSize: vocabSize, hot: 0, value: -20)
        stepFrom50[tokenizer.endOfTextToken] = 10

        var stepFrom200 = oneHotLogits(vocabSize: vocabSize, hot: 0, value: -20)
        stepFrom200[tokenizer.endOfTextToken] = 5

        let table: BeamMockRunner.ScoreTable = [
            tokenizer.noTimestampsToken: stepFromPrompt,
            100: stepFrom100,
            50: stepFrom50,
            200: stepFrom200,
        ]
        let runner = BeamMockRunner(
            scoreTable: table,
            defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        let result = try await decoder.decode(encoderOutput: encoder, options: options)
        #expect(result.map(\.id).prefix(2) == [100, 50])
    }

    @Test("beamSize <= 0 throws")
    func beamSizeZeroThrows() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.beamSize = 0
        let runner = BeamMockRunner(
            scoreTable: [:],
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

    @Test("beamSize > 1 with temperature > 0 throws")
    func beamPlusTemperatureThrows() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.beamSize = 3
        options.temperature = 0.5
        let runner = BeamMockRunner(
            scoreTable: [:],
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

    @Test("logSoftmax is numerically stable")
    func logSoftmaxStability() {
        let result = WhisperDecoder.logSoftmax(logits: [1000, 1001, 1002])
        for value in result {
            #expect(value.isFinite)
        }
        let sumExp = result.reduce(Float(0)) { acc, value in
            acc + Foundation.exp(value)
        }
        #expect(abs(sumExp - 1.0) < 0.001)
    }

    @Test("logSoftmax of all-negative-infinity returns input as-is")
    func logSoftmaxAllInf() {
        let input: [Float] = [-Float.infinity, -Float.infinity]
        let result = WhisperDecoder.logSoftmax(logits: input)
        #expect(result == input)
    }

    @Test("logSoftmax of empty input is empty")
    func logSoftmaxEmpty() {
        #expect(WhisperDecoder.logSoftmax(logits: []).isEmpty)
    }
}
