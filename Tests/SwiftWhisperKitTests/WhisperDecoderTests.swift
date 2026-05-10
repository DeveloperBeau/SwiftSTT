@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Synchronization
import Testing

@testable import SwiftWhisperKit

// MARK: - Mock runner

/// Records every call to `predict` and reset, and yields canned logits per
/// step from a queue. When the queue runs dry, it serves the configured
/// `defaultLogits`. Concurrency follows the encoder MockRunner pattern: the
/// type is `@unchecked Sendable` and `Mutex` guards mutable state because
/// `MLFeatureProvider` is not `Sendable`.
private final class MockStatefulRunner: StatefulCoreMLModelRunner, @unchecked Sendable {

    struct Call: Sendable {
        let tokenId: Int32
        let cacheLength: Int32
    }

    enum Behaviour {
        case canned(prefillCount: Int, logitsPerCall: [[Float]], defaultLogits: [Float])
        case throwError(SwiftWhisperError)
    }

    private struct State {
        var resetCount: Int = 0
        var calls: [Call] = []
        var canned: [[Float]] = []
        var remainingPrefill: Int = 0
    }

    private let behaviour: Behaviour
    private let logitsName: String
    private let tokenInputName: String
    private let cacheLengthName: String
    private let defaultLogits: [Float]
    private let state = Mutex<State>(State())

    init(
        behaviour: Behaviour,
        logitsName: String = WhisperDecoder.FeatureNames.default.logitsOutput,
        tokenInputName: String = WhisperDecoder.FeatureNames.default.tokenInput,
        cacheLengthName: String = WhisperDecoder.FeatureNames.default.cacheLength
    ) {
        self.behaviour = behaviour
        self.logitsName = logitsName
        self.tokenInputName = tokenInputName
        self.cacheLengthName = cacheLengthName

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

    var resetCount: Int { state.withLock { $0.resetCount } }
    var calls: [Call] { state.withLock { $0.calls } }

    func resetState() async {
        state.withLock { $0.resetCount += 1 }
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

private let defaultSpecials: [String: Int] = [
    "<|endoftext|>": 50_257,
    "<|startoftranscript|>": 50_258,
    "<|en|>": 50_259,
    "<|de|>": 50_261,
    "<|translate|>": 50_358,
    "<|transcribe|>": 50_359,
    "<|notimestamps|>": 50_363,
]

private func makeTokenizer() -> WhisperTokenizer {
    WhisperTokenizer(specialTokens: defaultSpecials)
}

private func makeEncoderArray(shape: [Int] = [1, 1500, 16], fill: Float = 0.0) throws
    -> MLMultiArray
{
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

private func makeLogitsProvider(values: [Float], name: String = "logits") throws
    -> any MLFeatureProvider
{
    let array = try MLMultiArray(
        shape: [1, 1, NSNumber(value: values.count)],
        dataType: .float32
    )
    let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
    for i in 0..<values.count {
        pointer[i] = values[i]
    }
    return try MLDictionaryFeatureProvider(dictionary: [name: array])
}

// MARK: - Tests

@Suite("WhisperDecoder")
struct WhisperDecoderTests {

    private let vocabSize = 50_400  // Covers all default special token IDs above.

    // MARK: end-to-end decode behaviour

    @Test("resetState is called once per decode call")
    func resetStateCalledOnce() async throws {
        let tokenizer = makeTokenizer()
        let runner = MockStatefulRunner(
            behaviour: .canned(
                prefillCount: 3,
                logitsPerCall: [oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)],
                defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
            )
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        _ = try await decoder.decode(encoderOutput: encoder, options: .default)
        #expect(runner.resetCount == 1)
    }

    @Test("Prefill feeds every prompt token through the runner")
    func prefillFeedsAllPromptTokens() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = "en"
        options.task = .transcribe
        // Force EOT on the first generation step.
        let runner = MockStatefulRunner(
            behaviour: .canned(
                prefillCount: 4,
                logitsPerCall: [],
                defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
            )
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        _ = try await decoder.decode(encoderOutput: encoder, options: options)

        // Prompt = [SOT, EN, transcribe, notimestamps] -> 4 prefill calls,
        // plus 1 generation call that produces EOT.
        #expect(runner.calls.count == 5)
        let firstFour = runner.calls.prefix(4).map(\.tokenId)
        #expect(
            firstFour == [
                Int32(tokenizer.startOfTranscriptToken),
                Int32(50_259),  // <|en|>
                Int32(tokenizer.transcribeToken),
                Int32(tokenizer.noTimestampsToken),
            ])
    }

    @Test("Generation stops when end-of-text logit wins")
    func generationStopsOnEndOfText() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.temperatureFallback = [0.0]

        let queue: [[Float]] = [
            oneHotLogits(vocabSize: vocabSize, hot: 100),
            oneHotLogits(vocabSize: vocabSize, hot: 200),
            oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
        ]
        let runner = MockStatefulRunner(
            behaviour: .canned(
                prefillCount: 3,
                logitsPerCall: queue,
                defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
            )
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        let result = try await decoder.decode(encoderOutput: encoder, options: options)
        #expect(result.map(\.id) == [100, 200])
    }

    @Test("Generation stops at maxTokens when EOT never appears")
    func generationStopsAtMaxTokens() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        // Always pick a non-EOT id.
        let runner = MockStatefulRunner(
            behaviour: .canned(
                prefillCount: 3,
                logitsPerCall: [],
                defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: 42)
            )
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        let result = try await decoder.decode(encoderOutput: encoder, options: options)
        #expect(result.count == WhisperDecoder.maxTokens)
        #expect(result.first?.id == 42)
        #expect(result.last?.id == 42)
    }

    @Test("Cache length increments by one for every call")
    func cacheLengthIncrementsCorrectly() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.temperatureFallback = [0.0]

        let queue: [[Float]] = [
            oneHotLogits(vocabSize: vocabSize, hot: 100),
            oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
        ]
        let runner = MockStatefulRunner(
            behaviour: .canned(
                prefillCount: 3,
                logitsPerCall: queue,
                defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
            )
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        _ = try await decoder.decode(encoderOutput: encoder, options: options)

        let cacheLengths = runner.calls.map(\.cacheLength)
        // Prompt without language = [SOT, transcribe, notimestamps] -> 3 prefill
        // calls (cache lengths 0, 1, 2), then generation calls 3 and 4.
        #expect(cacheLengths == [0, 1, 2, 3, 4])
    }

    @Test("Decode returns only generated tokens, never the prompt")
    func decodeReturnsExcludingPrompt() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = "en"
        options.suppressBlank = false
        options.temperatureFallback = [0.0]

        let queue: [[Float]] = [
            oneHotLogits(vocabSize: vocabSize, hot: 555),
            oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
        ]
        let runner = MockStatefulRunner(
            behaviour: .canned(
                prefillCount: 4,
                logitsPerCall: queue,
                defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
            )
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        let result = try await decoder.decode(encoderOutput: encoder, options: options)
        let promptIds: Set<Int> = [
            tokenizer.startOfTranscriptToken,
            50_259,
            tokenizer.transcribeToken,
            tokenizer.noTimestampsToken,
        ]
        for token in result {
            #expect(!promptIds.contains(token.id))
        }
        #expect(result.map(\.id) == [555])
    }

    @Test("Decode propagates runner errors")
    func decodeThrowsWhenRunnerThrows() async throws {
        let tokenizer = makeTokenizer()
        let runner = MockStatefulRunner(behaviour: .throwError(.decoderFailure("boom")))
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        do {
            _ = try await decoder.decode(encoderOutput: encoder, options: .default)
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            #expect(error == .decoderFailure("boom"))
        }
    }

    @Test("End-to-end programmed sequence yields expected ids")
    func endToEndDecodeProducesExpectedSequence() async throws {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        options.temperatureFallback = [0.0]

        let queue: [[Float]] = [
            oneHotLogits(vocabSize: vocabSize, hot: 7),
            oneHotLogits(vocabSize: vocabSize, hot: 8),
            oneHotLogits(vocabSize: vocabSize, hot: 9),
            oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken),
        ]
        let runner = MockStatefulRunner(
            behaviour: .canned(
                prefillCount: 3,
                logitsPerCall: queue,
                defaultLogits: oneHotLogits(vocabSize: vocabSize, hot: tokenizer.endOfTextToken)
            )
        )
        let decoder = WhisperDecoder(runner: runner, tokenizer: tokenizer)
        let encoder = try makeEncoderArray()

        let result = try await decoder.decode(encoderOutput: encoder, options: options)
        #expect(result.map(\.id) == [7, 8, 9])
    }

    // MARK: initialPromptTokens

    @Test("initialPromptTokens includes language tag when known")
    func initialPromptTokensIncludesLanguage() {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = "en"
        options.task = .transcribe
        let prompt = WhisperDecoder.initialPromptTokens(options: options, tokenizer: tokenizer)
        #expect(
            prompt == [
                tokenizer.startOfTranscriptToken,
                50_259,
                tokenizer.transcribeToken,
                tokenizer.noTimestampsToken,
            ])
    }

    @Test("initialPromptTokens swaps task token for translate")
    func initialPromptTokensTranscribeVsTranslate() {
        let tokenizer = makeTokenizer()
        var transcribe = DecodingOptions.default
        transcribe.language = "de"
        transcribe.task = .transcribe
        let transcribePrompt = WhisperDecoder.initialPromptTokens(
            options: transcribe, tokenizer: tokenizer)

        var translate = DecodingOptions.default
        translate.language = "de"
        translate.task = .translate
        let translatePrompt = WhisperDecoder.initialPromptTokens(
            options: translate, tokenizer: tokenizer)

        #expect(transcribePrompt.contains(tokenizer.transcribeToken))
        #expect(!transcribePrompt.contains(tokenizer.translateToken))
        #expect(translatePrompt.contains(tokenizer.translateToken))
        #expect(!translatePrompt.contains(tokenizer.transcribeToken))
    }

    @Test("initialPromptTokens skips language slot when not provided")
    func initialPromptTokensHandlesNoLanguage() {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = nil
        let prompt = WhisperDecoder.initialPromptTokens(options: options, tokenizer: tokenizer)
        #expect(
            prompt == [
                tokenizer.startOfTranscriptToken,
                tokenizer.transcribeToken,
                tokenizer.noTimestampsToken,
            ])
    }

    @Test("initialPromptTokens silently skips an unknown language tag")
    func initialPromptTokensSkipsUnknownLanguage() {
        let tokenizer = makeTokenizer()
        var options = DecodingOptions.default
        options.language = "xx"  // Not in the default specials map.
        let prompt = WhisperDecoder.initialPromptTokens(options: options, tokenizer: tokenizer)
        #expect(!prompt.contains(where: { $0 == 50_259 }))
        #expect(prompt.first == tokenizer.startOfTranscriptToken)
    }

    // MARK: extractLogits

    @Test("extractLogits returns the flat float buffer")
    func extractLogitsHappy() throws {
        let provider = try makeLogitsProvider(values: [0.1, 0.2, 0.3, 0.4])
        let logits = try WhisperDecoder.extractLogits(from: provider, name: "logits")
        #expect(logits == [0.1, 0.2, 0.3, 0.4])
    }

    @Test("extractLogits throws when feature missing")
    func extractLogitsThrowsOnMissingFeature() throws {
        let provider = try makeLogitsProvider(values: [0, 0], name: "wrong_name")
        do {
            _ = try WhisperDecoder.extractLogits(from: provider, name: "logits")
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            if case .decoderFailure = error {
            } else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    @Test("extractLogits throws when feature is not multiarray")
    func extractLogitsThrowsOnWrongType() throws {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "logits": MLFeatureValue(string: "not an array")
        ])
        do {
            _ = try WhisperDecoder.extractLogits(from: provider, name: "logits")
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            if case .decoderFailure = error {
            } else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    // MARK: buildFeatureProvider

    @Test("buildFeatureProvider exposes encoder embeds, token, cache features")
    func buildFeatureProviderHappyPath() throws {
        let encoder = try makeEncoderArray(shape: [1, 1500, 16], fill: 0.5)
        let provider = try WhisperDecoder.buildFeatureProvider(
            encoderOutput: encoder,
            tokenId: 1234,
            cacheLength: 7,
            names: .default
        )
        #expect(provider.featureNames.contains("encoder_output_embeds"))
        #expect(provider.featureNames.contains("decoder_input_ids"))
        #expect(provider.featureNames.contains("cache_length"))

        let tokenArray = try #require(
            provider.featureValue(for: "decoder_input_ids")?.multiArrayValue)
        #expect(tokenArray.shape == [1, 1])
        #expect(tokenArray[[0, 0] as [NSNumber]].int32Value == 1234)

        let cacheArray = try #require(provider.featureValue(for: "cache_length")?.multiArrayValue)
        #expect(cacheArray.shape == [1])
        #expect(cacheArray[[0] as [NSNumber]].int32Value == 7)

        let embeds = try #require(
            provider.featureValue(for: "encoder_output_embeds")?.multiArrayValue)
        #expect(embeds.shape == [1, 1500, 16])
    }

    // MARK: applySuppression

    @Test("applySuppression sets listed indices to -inf")
    func suppressionZerosOutTokens() {
        var logits: [Float] = [1, 2, 3, 4, 5]
        WhisperDecoder.applySuppression(
            logits: &logits,
            suppressTokens: [0, 2],
            blankToken: nil
        )
        #expect(logits[0] == -.infinity)
        #expect(logits[1] == 2)
        #expect(logits[2] == -.infinity)
        #expect(logits[3] == 4)
        #expect(logits[4] == 5)
    }

    @Test("applySuppression honors the blank token slot")
    func suppressBlankRespected() {
        var logits: [Float] = [1, 2, 3]
        WhisperDecoder.applySuppression(
            logits: &logits,
            suppressTokens: [],
            blankToken: 1
        )
        #expect(logits[0] == 1)
        #expect(logits[1] == -.infinity)
        #expect(logits[2] == 3)
    }

    @Test("applySuppression with empty list and nil blank is a no-op")
    func applySuppressionEmptyListNoOp() {
        var logits: [Float] = [1.5, 2.5, 3.5]
        let original = logits
        WhisperDecoder.applySuppression(
            logits: &logits,
            suppressTokens: [],
            blankToken: nil
        )
        #expect(logits == original)
    }

    @Test("applySuppression silently skips out-of-range indices")
    func suppressionIgnoresOutOfRange() {
        var logits: [Float] = [1, 2, 3]
        let original = logits
        WhisperDecoder.applySuppression(
            logits: &logits,
            suppressTokens: [-1, 999],
            blankToken: 999
        )
        #expect(logits == original)
    }

    // MARK: greedyArgmax

    @Test("greedyArgmax picks the largest value")
    func greedyArgmaxPicksLargest() {
        #expect(WhisperDecoder.greedyArgmax(logits: [0.1, 0.5, 0.4, 0.9, 0.2]) == 3)
    }

    @Test("greedyArgmax breaks ties by lowest index")
    func greedyArgmaxBreaksTiesByIndex() {
        #expect(WhisperDecoder.greedyArgmax(logits: [3, 3, 3, 3]) == 0)
    }

    // MARK: blankTokenId

    @Test("blankTokenId is nil when tokenizer cannot encode a leading space")
    func blankTokenIdEmpty() {
        let tokenizer = makeTokenizer()
        // Default specials map has no merges, no vocab; encoding " " yields [].
        let blank = WhisperDecoder.blankTokenId(tokenizer: tokenizer)
        #expect(blank == nil)
    }
}
