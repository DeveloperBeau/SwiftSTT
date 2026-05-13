@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

/// Drives `WhisperDecoder.decode(encoderOutput:options:)` end-to-end.
///
/// Wires `StatelessDecoderRunner` to a scripted `DecoderForwardModel` so the
/// full decode loop runs without a real Core ML model. Per-step runner
/// behaviour lives in `StatelessDecoderRunnerTests`; this suite covers only
/// the integration seam.
@Suite("WhisperDecoder stateless integration")
struct WhisperDecoderStatelessIntegrationTests {

    /// Scripted forward model that drives the decode loop.
    ///
    /// Returns logits that 100% favour the next token in `tokenSequence`,
    /// then EOT, then repeats. Each prediction also returns single-slot
    /// key/value cache updates so the splice path runs.
    ///
    /// The first ``prefillCount`` calls correspond to the prompt prefix
    /// being fed through the decoder and are ignored - the scripted token
    /// sequence only drives the generation-phase calls.
    private final class ScriptedStatelessModel: DecoderForwardModel, @unchecked Sendable {
        private let vocabSize: Int
        private let layerWidth: Int
        private let endOfTextToken: Int
        private let tokenSequence: [Int]
        private let prefillCount: Int
        private var callIndex: Int = 0
        private let lock = NSLock()

        init(
            vocabSize: Int,
            layerWidth: Int,
            endOfTextToken: Int,
            prefillCount: Int,
            tokenSequence: [Int]
        ) {
            self.vocabSize = vocabSize
            self.layerWidth = layerWidth
            self.endOfTextToken = endOfTextToken
            self.prefillCount = prefillCount
            self.tokenSequence = tokenSequence
        }

        func prediction(
            from features: any MLFeatureProvider
        ) async throws -> any MLFeatureProvider {
            let nextToken: Int = lock.withLock {
                let generationIndex = callIndex - prefillCount
                let token: Int
                if generationIndex < 0 {
                    // Prefill: token value is unused by the decoder loop
                    // (only the very last generation step's logits drive
                    // sampling). Emit EOT as a harmless placeholder.
                    token = endOfTextToken
                } else if generationIndex < tokenSequence.count {
                    token = tokenSequence[generationIndex]
                } else {
                    token = endOfTextToken
                }
                callIndex += 1
                return token
            }

            // Logits are Float32 because `WhisperDecoder.extractLogits` binds
            // the buffer to `Float`. A one-hot at the scripted token guarantees
            // greedy argmax picks it and yields logProb = 0 (passes thresholds).
            let logits = try MLMultiArray(
                shape: [1, 1, NSNumber(value: vocabSize)],
                dataType: .float32
            )
            let logitsPtr = logits.dataPointer.bindMemory(to: Float.self, capacity: vocabSize)
            for i in 0..<vocabSize { logitsPtr[i] = 0 }
            logitsPtr[nextToken] = 100

            // Cache updates are Float16 per the stateless decoder contract
            // (`DecoderKVCache.splice` reads them as UInt16-coded half floats).
            let keyUpdate = try MLMultiArray(
                shape: [1, NSNumber(value: layerWidth), 1, 1],
                dataType: .float16
            )
            let valueUpdate = try MLMultiArray(
                shape: [1, NSNumber(value: layerWidth), 1, 1],
                dataType: .float16
            )

            return try MLDictionaryFeatureProvider(dictionary: [
                "logits": logits,
                "key_cache_updates": keyUpdate,
                "value_cache_updates": valueUpdate,
            ])
        }
    }

    /// Special-tokens map mirrors the one used by `WhisperDecoderTests`.
    private static let defaultSpecials: [String: Int] = [
        "<|endoftext|>": 50_257,
        "<|startoftranscript|>": 50_258,
        "<|en|>": 50_259,
        "<|de|>": 50_261,
        "<|translate|>": 50_358,
        "<|transcribe|>": 50_359,
        "<|notimestamps|>": 50_363,
    ]

    /// Vocab size covers every special token id above and leaves room for
    /// the scripted text tokens (10, 20, 30).
    private let vocabSize: Int = 50_400

    @Test("decode end-to-end through StatelessDecoderRunner emits scripted tokens")
    func endToEndDecodeLoop() async throws {
        let tokenizer = WhisperTokenizer(specialTokens: Self.defaultSpecials)
        let endOfText = tokenizer.endOfTextToken
        // Pick three text tokens guaranteed to be in vocab and not special.
        let scriptedTokens = [10, 20, 30]

        // Prompt with `language: nil` is [SOT, transcribe, notimestamps] -> 3
        // prefill calls before generation starts.
        let scripted = ScriptedStatelessModel(
            vocabSize: vocabSize,
            layerWidth: 4,
            endOfTextToken: endOfText,
            prefillCount: 3,
            tokenSequence: scriptedTokens
        )

        let inputNames: Set<String> = [
            "input_ids", "cache_length", "key_cache", "value_cache",
            "kv_cache_update_mask", "encoder_output_embeds", "decoder_key_padding_mask",
        ]
        let runner = try StatelessDecoderRunner(
            model: scripted,
            inputNames: inputNames,
            layerWidth: 4,
            capacity: 16
        )

        let statelessFeatureNames = WhisperDecoder.FeatureNames(
            encoderEmbeds: "encoder_output_embeds",
            tokenInput: "input_ids",
            cacheLength: "cache_length",
            logitsOutput: "logits"
        )
        let decoder = WhisperDecoder(
            runner: runner,
            tokenizer: tokenizer,
            featureNames: statelessFeatureNames
        )

        // Encoder output shape matches argmaxinc tiny: [1, 384, 1, 1500].
        let encoderOutput = try MLMultiArray(
            shape: [1, 384, 1, 1500],
            dataType: .float16
        )

        var options = DecodingOptions.default
        options.language = nil
        options.suppressBlank = false
        // Keep the fallback loop simple; we expect the first attempt to pass.
        options.temperatureFallback = [0.0]

        let tokens = try await decoder.decode(
            encoderOutput: encoderOutput,
            options: options
        )

        let ids = tokens.map { Int($0.id) }
        #expect(ids == scriptedTokens)
    }
}
