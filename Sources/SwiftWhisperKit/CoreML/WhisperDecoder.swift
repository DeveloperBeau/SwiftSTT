@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore

/// Core ML implementation of the Whisper text decoder.
///
/// The decoder is autoregressive: each forward pass produces logits over the
/// vocabulary for one new token, conditioned on the encoder embedding plus
/// all previously generated tokens. Per-token cost is held flat by routing
/// the key-value cache through Core ML's `MLState` (iOS 18 / macOS 15
/// minimum).
///
/// ## Pipeline
///
/// 1. ``StatefulCoreMLModelRunner/resetState()`` flushes the KV cache.
/// 2. The prompt prefix (`<|startoftranscript|>`, optional language tag,
///    `<|transcribe|>` or `<|translate|>`, `<|notimestamps|>`) is fed
///    through the runner one token at a time to prime the cache.
/// 3. Generation runs greedily up to ``maxTokens`` or until the runner
///    returns the end-of-text token.
///
/// ## Sampling
///
/// Greedy only in M5c. Beam search and temperature sampling land in a later
/// milestone.
public actor WhisperDecoder: TokenDecoding {

    /// Hard cap on generated tokens per decode call. Matches the Whisper
    /// per-segment limit of 224 content tokens.
    public static let maxTokens: Int = 224

    /// Names of the model's input and output features. Default values match
    /// the argmaxinc/whisperkit-coreml stateful decoder convention. Override
    /// when wiring against a different model export.
    public struct FeatureNames: Sendable, Equatable {
        public var encoderEmbeds: String
        public var tokenInput: String
        public var cacheLength: String
        public var logitsOutput: String

        public init(
            encoderEmbeds: String,
            tokenInput: String,
            cacheLength: String,
            logitsOutput: String
        ) {
            self.encoderEmbeds = encoderEmbeds
            self.tokenInput = tokenInput
            self.cacheLength = cacheLength
            self.logitsOutput = logitsOutput
        }

        public static let `default` = FeatureNames(
            encoderEmbeds: "encoder_output_embeds",
            tokenInput: "decoder_input_ids",
            cacheLength: "cache_length",
            logitsOutput: "logits"
        )
    }

    private let runner: any StatefulCoreMLModelRunner
    private let tokenizer: WhisperTokenizer
    private let featureNames: FeatureNames

    public init(
        runner: any StatefulCoreMLModelRunner,
        tokenizer: WhisperTokenizer,
        featureNames: FeatureNames = .default
    ) {
        self.runner = runner
        self.tokenizer = tokenizer
        self.featureNames = featureNames
    }

    public func decode(
        encoderOutput: MLMultiArray,
        options: DecodingOptions
    ) async throws(SwiftWhisperError) -> [WhisperToken] {
        await runner.resetState()

        let prompt = Self.initialPromptTokens(options: options, tokenizer: tokenizer)
        var cacheLength = 0

        // Prefill: feed prompt through the runner so the KV cache holds the
        // prefix attention. Logits from these calls are discarded.
        for tokenId in prompt {
            let provider = try Self.buildFeatureProvider(
                encoderOutput: encoderOutput,
                tokenId: tokenId,
                cacheLength: cacheLength,
                names: featureNames
            )
            _ = try await runner.predict(features: provider)
            cacheLength += 1
        }

        let endOfText = tokenizer.endOfTextToken
        // The token id for the leading space, used to suppress emitting silence
        // as the very first generated token when `suppressBlank` is on.
        let blankToken: Int? = options.suppressBlank ? Self.blankTokenId(tokenizer: tokenizer) : nil

        var generated: [WhisperToken] = []

        while generated.count < Self.maxTokens {
            let lastTokenId = generated.last?.id ?? prompt.last ?? endOfText
            let provider = try Self.buildFeatureProvider(
                encoderOutput: encoderOutput,
                tokenId: lastTokenId,
                cacheLength: cacheLength,
                names: featureNames
            )
            let output = try await runner.predict(features: provider)
            var logits = try Self.extractLogits(from: output, name: featureNames.logitsOutput)

            // Suppress blank only on the first generated token (matches the
            // reference Whisper rule).
            let blankForThisStep = generated.isEmpty ? blankToken : nil
            Self.applySuppression(
                logits: &logits,
                suppressTokens: options.suppressTokens,
                blankToken: blankForThisStep
            )

            let next = Self.greedyArgmax(logits: logits)
            cacheLength += 1

            if next == endOfText {
                break
            }

            let text = tokenizer.decode(tokens: [next])
            generated.append(WhisperToken(id: next, text: text))
        }

        return generated
    }

    // MARK: - Pure helpers (visible to tests)

    /// Builds the prompt prefix the model expects before generation:
    /// `<|startoftranscript|>` optionally followed by a language tag,
    /// then the task token, then `<|notimestamps|>`.
    static func initialPromptTokens(
        options: DecodingOptions,
        tokenizer: WhisperTokenizer
    ) -> [Int] {
        var tokens: [Int] = [tokenizer.startOfTranscriptToken]

        if let language = options.language {
            let tag = "<|\(language)|>"
            let encoded = tokenizer.encode(text: tag)
            // Only include if the tokenizer recognises it as a single special
            // token; otherwise an unknown language is silently skipped.
            if encoded.count == 1, tokenizer.isSpecial(token: encoded[0]) {
                tokens.append(encoded[0])
            }
        }

        switch options.task {
        case .transcribe:
            tokens.append(tokenizer.transcribeToken)
        case .translate:
            tokens.append(tokenizer.translateToken)
        }

        tokens.append(tokenizer.noTimestampsToken)
        return tokens
    }

    /// Packs the per-step decoder inputs into a feature provider. Encoder
    /// embeddings are passed every step; the model is free to ignore them
    /// after the first call once the cross-attention cache is populated.
    static func buildFeatureProvider(
        encoderOutput: MLMultiArray,
        tokenId: Int,
        cacheLength: Int,
        names: FeatureNames
    ) throws(SwiftWhisperError) -> any MLFeatureProvider {
        let tokenArray: MLMultiArray
        let cacheArray: MLMultiArray
        do {
            tokenArray = try MLMultiArray(shape: [1, 1], dataType: .int32)
            cacheArray = try MLMultiArray(shape: [1], dataType: .int32)
        } catch {
            throw .decoderFailure("MLMultiArray init: \(error.localizedDescription)")
        }
        tokenArray[[0, 0] as [NSNumber]] = NSNumber(value: Int32(tokenId))
        cacheArray[[0] as [NSNumber]] = NSNumber(value: Int32(cacheLength))

        do {
            return try MLDictionaryFeatureProvider(dictionary: [
                names.encoderEmbeds: encoderOutput,
                names.tokenInput: tokenArray,
                names.cacheLength: cacheArray,
            ])
        } catch {
            throw .decoderFailure("feature provider: \(error.localizedDescription)")
        }
    }

    /// Pulls logits from the model output and copies them into a flat
    /// `[Float]` for sampling.
    static func extractLogits(
        from output: any MLFeatureProvider,
        name: String
    ) throws(SwiftWhisperError) -> [Float] {
        guard let value = output.featureValue(for: name) else {
            throw .decoderFailure("missing feature '\(name)' in decoder output")
        }
        guard let array = value.multiArrayValue else {
            throw .decoderFailure("feature '\(name)' is not an MLMultiArray")
        }
        let count = array.count
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        var logits = [Float](repeating: 0, count: count)
        for i in 0..<count {
            logits[i] = pointer[i]
        }
        return logits
    }

    /// Sets `-Float.infinity` at every suppressed index so they cannot be
    /// picked by argmax. Out-of-range indices are silently ignored.
    static func applySuppression(
        logits: inout [Float],
        suppressTokens: [Int],
        blankToken: Int?
    ) {
        for index in suppressTokens where index >= 0 && index < logits.count {
            logits[index] = -.infinity
        }
        if let blank = blankToken, blank >= 0, blank < logits.count {
            logits[blank] = -.infinity
        }
    }

    /// Returns the index of the largest logit. Ties are broken by lowest
    /// index (the natural result of a single forward pass).
    static func greedyArgmax(logits: [Float]) -> Int {
        var bestIndex = 0
        var bestValue: Float = -.infinity
        for (i, value) in logits.enumerated() where value > bestValue {
            bestIndex = i
            bestValue = value
        }
        return bestIndex
    }

    /// Whisper's blank token is the BPE encoding of a single leading space
    /// (`" "`). Resolving it dynamically avoids hard-coding the id.
    static func blankTokenId(tokenizer: WhisperTokenizer) -> Int? {
        let encoded = tokenizer.encode(text: " ")
        return encoded.first
    }
}
