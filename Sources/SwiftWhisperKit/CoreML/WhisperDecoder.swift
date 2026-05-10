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
/// Three sampling modes, picked from ``DecodingOptions``:
///
/// - `temperature == 0 && beamSize == 1` runs greedy argmax (the M5c path).
/// - `temperature > 0 && beamSize == 1` samples from `softmax(logits / T)`
///   using the injected ``RandomSource``.
/// - `beamSize > 1 && temperature == 0` runs beam search width N. The current
///   implementation shares one Core ML `MLState` between beams and re-prefills
///   the prompt for each beam at every step, so it is correct but quadratic in
///   step count. Per-beam state is a future milestone.
///
/// Beam search combined with sampling is rejected with
/// ``SwiftWhisperError/invalidDecodingOption(_:)`` because the reference
/// Python implementation does not mix the two paths either.
public actor WhisperDecoder: TokenDecoding {

    /// Hard cap on generated tokens per decode call.
    ///
    /// Matches the Whisper per-segment limit of 224 content tokens.
    public static let maxTokens: Int = 224

    /// Names of the model's input and output features.
    ///
    /// Default values match the argmaxinc/whisperkit-coreml stateful decoder convention. Override
    /// when wiring against a different model export.
    public struct FeatureNames: Sendable, Equatable {
        /// Name of the encoder-embeddings input feature.
        public var encoderEmbeds: String
        /// Name of the token-id input feature.
        public var tokenInput: String
        /// Name of the KV-cache length input feature.
        public var cacheLength: String
        /// Name of the logits output feature.
        public var logitsOutput: String

        /// Creates a new FeatureNames with the supplied values.
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

        /// Default feature names matching the argmaxinc/whisperkit-coreml stateful decoder export.
        public static let `default` = FeatureNames(
            encoderEmbeds: "encoder_output_embeds",
            tokenInput: "decoder_input_ids",
            cacheLength: "cache_length",
            logitsOutput: "logits"
        )
    }

    private let runner: any StatefulCoreMLModelRunner
    private let branchableRunner: (any BranchableStatefulRunner)?
    private let tokenizer: WhisperTokenizer
    private let featureNames: FeatureNames
    private var rng: any RandomSource

    /// Creates a new WhisperDecoder with the supplied values.
    public init(
        runner: any StatefulCoreMLModelRunner,
        tokenizer: WhisperTokenizer,
        featureNames: FeatureNames = .default,
        rng: any RandomSource = SystemRandom()
    ) {
        self.runner = runner
        self.branchableRunner = nil
        self.tokenizer = tokenizer
        self.featureNames = featureNames
        self.rng = rng
    }

    /// Initialiser variant that accepts a runner capable of branching its
    /// state.
    ///
    /// Beam search uses the branchable path so each beam owns its own KV cache and only re-prefills when its prefix changes.
    public init(
        branchableRunner: any BranchableStatefulRunner,
        tokenizer: WhisperTokenizer,
        featureNames: FeatureNames = .default,
        rng: any RandomSource = SystemRandom()
    ) {
        self.runner = branchableRunner
        self.branchableRunner = branchableRunner
        self.tokenizer = tokenizer
        self.featureNames = featureNames
        self.rng = rng
    }

    /// Decodes the input.
    public func decode(
        encoderOutput: MLMultiArray,
        options: DecodingOptions
    ) async throws(SwiftWhisperError) -> [WhisperToken] {
        try Self.validate(options: options)

        if options.beamSize > 1 {
            if let branchable = branchableRunner {
                return try await beamSearchDecodeBranchable(
                    encoderOutput: encoderOutput,
                    options: options,
                    rootRunner: branchable
                )
            }
            return try await beamSearchDecode(encoderOutput: encoderOutput, options: options)
        }

        return try await decodeWithFallback(encoderOutput: encoderOutput, options: options)
    }

    /// Retry attempts at successive temperatures until one passes the
    /// anti-hallucination thresholds, or all are exhausted (in which case the
    /// final attempt is returned).
    ///
    /// When the final attempt is also flagged as silence by the no-speech check, an empty token list is returned.
    private func decodeWithFallback(
        encoderOutput: MLMultiArray,
        options: DecodingOptions
    ) async throws(SwiftWhisperError) -> [WhisperToken] {
        var lastAttempt: DecodeAttempt?
        for temperature in options.temperatureFallback {
            var stepOptions = options
            stepOptions.temperature = temperature
            let attempt = try await singleHypothesisAttempt(
                encoderOutput: encoderOutput,
                options: stepOptions
            )
            lastAttempt = attempt
            if attempt.passesThresholds(options: options) {
                break
            }
        }
        guard let final = lastAttempt else { return [] }
        if final.isNoSpeech(options: options) { return [] }
        return final.tokens
    }

    // MARK: - Single hypothesis (greedy or temperature sampling)

    /// Outcome of one greedy or sampled decode pass.
    ///
    /// Carries the per-attempt statistics that the temperature-fallback loop in
    /// ``decodeWithFallback(encoderOutput:options:)`` consults to decide
    /// whether to retry at a hotter temperature, accept the result, or skip
    /// the segment entirely.
    private struct DecodeAttempt {
        var tokens: [WhisperToken]
        var avgLogProb: Float
        var noSpeechProb: Float
        var compressionRatio: Float

        func passesThresholds(options: DecodingOptions) -> Bool {
            if avgLogProb < options.logProbThreshold { return false }
            if compressionRatio > options.compressionRatioThreshold { return false }
            return true
        }

        func isNoSpeech(options: DecodingOptions) -> Bool {
            noSpeechProb > options.noSpeechThreshold
                && avgLogProb < options.logProbThreshold
        }
    }

    private func singleHypothesisAttempt(
        encoderOutput: MLMultiArray,
        options: DecodingOptions
    ) async throws(SwiftWhisperError) -> DecodeAttempt {
        await runner.resetState()

        let prompt = Self.initialPromptTokens(options: options, tokenizer: tokenizer)
        var cacheLength = 0

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
        let blankToken: Int? = options.suppressBlank ? Self.blankTokenId(tokenizer: tokenizer) : nil
        var generated: [WhisperToken] = []
        var logProbSum: Float = 0
        var noSpeechProb: Float = 0

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

            // Capture the no-speech probability from RAW logits (before any
            // masking) so the silence-skip heuristic still sees it when the
            // caller suppresses the token from emission.
            if generated.isEmpty {
                noSpeechProb = Self.noSpeechProbability(
                    logits: logits,
                    noSpeechToken: tokenizer.noSpeechToken
                )
            }

            // Apply repetition penalty to raw logits before any masking so the
            // penalty mirrors the Whisper reference implementation, which
            // operates on logits prior to suppression.
            if options.repetitionPenalty != 1.0 {
                let previousIds = generated.map(\.id)
                Self.applyRepetitionPenalty(
                    logits: &logits,
                    previousTokens: previousIds,
                    penalty: options.repetitionPenalty
                )
            }

            let blankForThisStep = generated.isEmpty ? blankToken : nil
            Self.applySuppression(
                logits: &logits,
                suppressTokens: options.suppressTokens,
                blankToken: blankForThisStep
            )

            // Logit bias is purely additive and runs after suppression so the
            // caller can lift a token even when it would otherwise be masked
            // by `-inf`. This matches the documented "any finite Float"
            // contract on `DecodingOptions.logitBias`.
            if !options.logitBias.isEmpty {
                Self.applyLogitBias(logits: &logits, bias: options.logitBias)
            }

            let next: Int
            let stepLogProb: Float
            if options.temperature > 0 {
                var probs = Self.softmax(logits: logits, temperature: options.temperature)
                if options.topP < 1.0 {
                    probs = Self.topPFilter(probs: probs, threshold: options.topP)
                }
                next = Self.sample(probs: probs, rng: &rng)
                stepLogProb =
                    next < probs.count && probs[next] > 0
                    ? logFloat(probs[next])
                    : -.infinity
            } else {
                next = Self.greedyArgmax(logits: logits)
                let logProbs = Self.logSoftmax(logits: logits)
                stepLogProb = next < logProbs.count ? logProbs[next] : -.infinity
            }
            cacheLength += 1

            if next == endOfText {
                break
            }

            logProbSum += stepLogProb
            let text = tokenizer.decode(tokens: [next])
            generated.append(WhisperToken(id: next, text: text))
        }

        let avgLogProb = generated.isEmpty ? 0 : logProbSum / Float(generated.count)
        let decodedText = tokenizer.decode(tokens: generated.map(\.id))
        let ratio = Self.compressionRatio(text: decodedText)

        return DecodeAttempt(
            tokens: generated,
            avgLogProb: avgLogProb,
            noSpeechProb: noSpeechProb,
            compressionRatio: ratio
        )
    }

    // MARK: - Beam search

    /// Width-N beam search.
    ///
    /// Each step expands every beam, scores all resulting (beam, token) pairs by sum of log-probabilities, and keeps
    /// the top N. Because all beams share a single Core ML KV cache, every
    /// per-beam expansion has to reset the state and replay the prefix +
    /// the beam's tokens to repopulate the cache. That is O(steps^2) work
    /// per beam; correct but slow. Per-beam state ships in a future milestone.
    private func beamSearchDecode(
        encoderOutput: MLMultiArray,
        options: DecodingOptions
    ) async throws(SwiftWhisperError) -> [WhisperToken] {
        let prompt = Self.initialPromptTokens(options: options, tokenizer: tokenizer)
        let endOfText = tokenizer.endOfTextToken
        let blankToken: Int? = options.suppressBlank ? Self.blankTokenId(tokenizer: tokenizer) : nil

        struct Beam: Sendable {
            var tokens: [WhisperToken]
            var logProbSum: Float
            var finished: Bool
        }

        var beams: [Beam] = [Beam(tokens: [], logProbSum: 0, finished: false)]

        for step in 0..<Self.maxTokens {
            if beams.allSatisfy(\.finished) { break }

            var candidates: [(beamIndex: Int, tokenId: Int, score: Float, logProb: Float)] = []

            for (beamIndex, beam) in beams.enumerated() {
                if beam.finished {
                    candidates.append((beamIndex, endOfText, beam.logProbSum, 0))
                    continue
                }

                var logits = try await replayBeam(
                    encoderOutput: encoderOutput,
                    prompt: prompt,
                    beamTokens: beam.tokens
                )

                let blankForThisStep = step == 0 ? blankToken : nil
                Self.applySuppression(
                    logits: &logits,
                    suppressTokens: options.suppressTokens,
                    blankToken: blankForThisStep
                )

                let logProbs = Self.logSoftmax(logits: logits)
                let topK = Self.selectTopK(logits: logProbs, k: options.beamSize)
                for entry in topK {
                    candidates.append(
                        (beamIndex, entry.token, beam.logProbSum + entry.score, entry.score))
                }
            }

            candidates.sort { $0.score > $1.score }
            let kept = candidates.prefix(options.beamSize)
            var nextBeams: [Beam] = []
            for candidate in kept {
                let parent = beams[candidate.beamIndex]
                if parent.finished {
                    nextBeams.append(parent)
                    continue
                }
                if candidate.tokenId == endOfText {
                    var finished = parent
                    finished.logProbSum = candidate.score
                    finished.finished = true
                    nextBeams.append(finished)
                    continue
                }
                let text = tokenizer.decode(tokens: [candidate.tokenId])
                var grown = parent
                grown.tokens.append(
                    WhisperToken(
                        id: candidate.tokenId, text: text, probability: expFloat(candidate.logProb))
                )
                grown.logProbSum = candidate.score
                nextBeams.append(grown)
            }
            beams = nextBeams
        }

        let best =
            beams.max { lhs, rhs in
                lhs.logProbSum < rhs.logProbSum
            } ?? beams[0]
        return best.tokens
    }

    // MARK: - Beam search with per-beam KV state

    /// Beam container used by the branchable path.
    ///
    /// Owns its runner clone and tracks how much of its prefix the runner has already consumed so the
    /// step loop can avoid redundant resets+replays.
    private struct StatefulBeam {
        var tokens: [WhisperToken]
        var logProbSum: Float
        var finished: Bool
        var runner: any BranchableStatefulRunner
        /// Token id sequence (prompt + beam tokens) the runner's KV cache
        /// already contains.
        ///
        /// When the next step's required prefix matches this, only one new
        /// predict is needed; otherwise reset+replay.
        var consumedPrefix: [Int]
    }

    /// Width-N beam search where each beam owns its own branchable runner.
    ///
    /// Because KV cache state is independent per beam, the typical step costs
    /// one predict per beam (just the new token) instead of replaying the
    /// full prefix every step.
    ///
    /// When two child beams come from the same parent, the parent runner is
    /// branched once for the second child so both can advance independently.
    private func beamSearchDecodeBranchable(
        encoderOutput: MLMultiArray,
        options: DecodingOptions,
        rootRunner: any BranchableStatefulRunner
    ) async throws(SwiftWhisperError) -> [WhisperToken] {
        let prompt = Self.initialPromptTokens(options: options, tokenizer: tokenizer)
        let endOfText = tokenizer.endOfTextToken
        let blankToken: Int? = options.suppressBlank ? Self.blankTokenId(tokenizer: tokenizer) : nil

        // Prime the root runner with the prompt prefix once. Subsequent
        // beams are branched from a freshly primed runner using replayPrefix.
        await rootRunner.resetState()
        var rootCacheLength = 0
        for tokenId in prompt {
            let provider = try Self.buildFeatureProvider(
                encoderOutput: encoderOutput,
                tokenId: tokenId,
                cacheLength: rootCacheLength,
                names: featureNames
            )
            _ = try await rootRunner.predict(features: provider)
            rootCacheLength += 1
        }

        var beams: [StatefulBeam] = [
            StatefulBeam(
                tokens: [],
                logProbSum: 0,
                finished: false,
                runner: rootRunner,
                consumedPrefix: prompt
            )
        ]

        for step in 0..<Self.maxTokens {
            if beams.allSatisfy(\.finished) { break }

            var candidates: [(beamIndex: Int, tokenId: Int, score: Float, logProb: Float)] = []

            for (beamIndex, beam) in beams.enumerated() {
                if beam.finished {
                    candidates.append((beamIndex, endOfText, beam.logProbSum, 0))
                    continue
                }

                let requiredPrefix = prompt + beam.tokens.map(\.id)
                let lastTokenId = requiredPrefix.last ?? endOfText
                let cacheLength = requiredPrefix.count - 1

                if beam.consumedPrefix != requiredPrefix {
                    try await replayPrefixIntoRunner(
                        runner: beam.runner,
                        encoderOutput: encoderOutput,
                        prefix: requiredPrefix
                    )
                    beams[beamIndex].consumedPrefix = requiredPrefix
                }

                let provider = try Self.buildFeatureProvider(
                    encoderOutput: encoderOutput,
                    tokenId: lastTokenId,
                    cacheLength: cacheLength,
                    names: featureNames
                )
                let output = try await beams[beamIndex].runner.predict(features: provider)
                var logits = try Self.extractLogits(from: output, name: featureNames.logitsOutput)

                let blankForThisStep = step == 0 ? blankToken : nil
                Self.applySuppression(
                    logits: &logits,
                    suppressTokens: options.suppressTokens,
                    blankToken: blankForThisStep
                )
                if !options.logitBias.isEmpty {
                    Self.applyLogitBias(logits: &logits, bias: options.logitBias)
                }

                let logProbs = Self.logSoftmax(logits: logits)
                let topK = Self.selectTopK(logits: logProbs, k: options.beamSize)
                for entry in topK {
                    candidates.append(
                        (beamIndex, entry.token, beam.logProbSum + entry.score, entry.score))
                }
            }

            candidates.sort { $0.score > $1.score }
            let kept = Array(candidates.prefix(options.beamSize))

            // Reuse each parent runner for the first child only; subsequent
            // children sharing that parent get a freshly branched runner.
            var parentUsed = Set<Int>()
            var nextBeams: [StatefulBeam] = []
            for candidate in kept {
                let parent = beams[candidate.beamIndex]
                if parent.finished {
                    nextBeams.append(parent)
                    continue
                }
                let runner: any BranchableStatefulRunner
                if parentUsed.insert(candidate.beamIndex).inserted {
                    runner = parent.runner
                } else {
                    runner = try await parent.runner.branch()
                }

                if candidate.tokenId == endOfText {
                    var finished = parent
                    finished.runner = runner
                    finished.logProbSum = candidate.score
                    finished.finished = true
                    nextBeams.append(finished)
                    continue
                }

                let text = tokenizer.decode(tokens: [candidate.tokenId])
                var grown = StatefulBeam(
                    tokens: parent.tokens,
                    logProbSum: candidate.score,
                    finished: false,
                    runner: runner,
                    consumedPrefix: parent.consumedPrefix
                )
                grown.tokens.append(
                    WhisperToken(
                        id: candidate.tokenId, text: text, probability: expFloat(candidate.logProb))
                )
                nextBeams.append(grown)
            }
            beams = nextBeams
        }

        let best =
            beams.max { lhs, rhs in
                lhs.logProbSum < rhs.logProbSum
            } ?? beams[0]
        return best.tokens
    }

    /// Resets `runner` and feeds `prefix` through it so the KV cache mirrors
    /// the beam's full token prefix up to (but not including) the slot the
    /// next predict will produce.
    private func replayPrefixIntoRunner(
        runner: any BranchableStatefulRunner,
        encoderOutput: MLMultiArray,
        prefix: [Int]
    ) async throws(SwiftWhisperError) {
        await runner.resetState()
        var cacheLength = 0
        // Stop one short of the last token so the next predict() call (issued
        // by the caller) advances the cache to the right slot.
        let limit = max(prefix.count - 1, 0)
        for index in 0..<limit {
            let provider = try Self.buildFeatureProvider(
                encoderOutput: encoderOutput,
                tokenId: prefix[index],
                cacheLength: cacheLength,
                names: featureNames
            )
            _ = try await runner.predict(features: provider)
            cacheLength += 1
        }
    }

    /// Resets the runner state and feeds (prompt + already-chosen beam tokens)
    /// through it so the KV cache reflects this beam's full prefix.
    ///
    /// Returns the logits for the next token slot.
    private func replayBeam(
        encoderOutput: MLMultiArray,
        prompt: [Int],
        beamTokens: [WhisperToken]
    ) async throws(SwiftWhisperError) -> [Float] {
        await runner.resetState()
        var cacheLength = 0
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
        for token in beamTokens {
            let provider = try Self.buildFeatureProvider(
                encoderOutput: encoderOutput,
                tokenId: token.id,
                cacheLength: cacheLength,
                names: featureNames
            )
            _ = try await runner.predict(features: provider)
            cacheLength += 1
        }
        let lastTokenId = beamTokens.last?.id ?? prompt.last ?? tokenizer.endOfTextToken
        let provider = try Self.buildFeatureProvider(
            encoderOutput: encoderOutput,
            tokenId: lastTokenId,
            cacheLength: cacheLength,
            names: featureNames
        )
        let output = try await runner.predict(features: provider)
        return try Self.extractLogits(from: output, name: featureNames.logitsOutput)
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

        if options.withoutTimestamps {
            tokens.append(tokenizer.noTimestampsToken)
        }
        return tokens
    }

    /// Validates user-supplied decoding options.
    ///
    /// Throws on negative temperature, beam width below 1, or the unsupported combination of
    /// beam search with sampling.
    static func validate(options: DecodingOptions) throws(SwiftWhisperError) {
        if options.temperature < 0 {
            throw .invalidDecodingOption("temperature must be >= 0; got \(options.temperature)")
        }
        if options.beamSize < 1 {
            throw .invalidDecodingOption("beamSize must be >= 1; got \(options.beamSize)")
        }
        if options.beamSize > 1 && options.temperature > 0 {
            throw .invalidDecodingOption(
                "beam search and temperature sampling are mutually exclusive")
        }
        if options.topP <= 0 || options.topP > 1.0 {
            throw .invalidDecodingOption("topP must be in (0, 1]; got \(options.topP)")
        }
        if options.repetitionPenalty <= 0 {
            throw .invalidDecodingOption(
                "repetitionPenalty must be > 0; got \(options.repetitionPenalty)")
        }
        if options.beamSize == 1 && options.temperatureFallback.isEmpty {
            throw .invalidDecodingOption("temperatureFallback must contain at least one entry")
        }
        if options.temperatureFallback.contains(where: { $0 < 0 }) {
            throw .invalidDecodingOption("temperatureFallback entries must be >= 0")
        }
    }

    /// Packs the per-step decoder inputs into a feature provider.
    ///
    /// Encoder embeddings are passed every step; the model is free to ignore them
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
    /// picked by argmax.
    ///
    /// Out-of-range indices are silently ignored.
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

    /// Returns the index of the largest logit.
    ///
    /// Ties are broken by lowest index (the natural result of a single forward pass).
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
    /// (`" "`).
    ///
    /// Resolving it dynamically avoids hard-coding the id.
    static func blankTokenId(tokenizer: WhisperTokenizer) -> Int? {
        let encoded = tokenizer.encode(text: " ")
        return encoded.first
    }

    /// Numerically stable softmax over `logits`, scaled by `temperature`.
    ///
    /// Values at `-Float.infinity` (suppressed entries) contribute zero
    /// probability. The caller is expected to pass `temperature > 0`; a value
    /// of `0` would divide by zero, so this method asserts the contract by
    /// returning a uniform distribution in that case.
    static func softmax(logits: [Float], temperature: Float) -> [Float] {
        guard !logits.isEmpty else { return [] }
        let safeTemp = max(temperature, .leastNormalMagnitude)
        var maxLogit: Float = -.infinity
        for value in logits where value > maxLogit {
            maxLogit = value
        }
        if maxLogit == -.infinity {
            return [Float](repeating: 1.0 / Float(logits.count), count: logits.count)
        }
        var exps = [Float](repeating: 0, count: logits.count)
        var sum: Float = 0
        for i in 0..<logits.count {
            let scaled = (logits[i] - maxLogit) / safeTemp
            let value = expFloat(scaled)
            exps[i] = value
            sum += value
        }
        guard sum > 0 else {
            return [Float](repeating: 1.0 / Float(logits.count), count: logits.count)
        }
        for i in 0..<exps.count {
            exps[i] /= sum
        }
        return exps
    }

    /// Log-softmax used by beam search scoring.
    ///
    /// Numerically stable via the max-subtraction trick. Suppressed entries (at `-inf`) stay at `-inf`.
    static func logSoftmax(logits: [Float]) -> [Float] {
        guard !logits.isEmpty else { return [] }
        var maxLogit: Float = -.infinity
        for value in logits where value > maxLogit {
            maxLogit = value
        }
        if maxLogit == -.infinity {
            return logits
        }
        var sumExp: Float = 0
        for value in logits {
            sumExp += expFloat(value - maxLogit)
        }
        let logSum = maxLogit + logFloat(sumExp)
        var out = [Float](repeating: 0, count: logits.count)
        for i in 0..<logits.count {
            out[i] = logits[i] - logSum
        }
        return out
    }

    /// Draws one index from the discrete distribution `probs` using inverse
    /// transform sampling against the injected RNG. `probs` is expected to
    /// sum to ~1; small drift is tolerated.
    static func sample<R: RandomSource>(probs: [Float], rng: inout R) -> Int {
        guard !probs.isEmpty else { return 0 }
        let raw = rng.next()
        let unit = Float(raw) / Float(UInt64.max)
        var cumulative: Float = 0
        for (index, p) in probs.enumerated() {
            cumulative += p
            if unit <= cumulative {
                return index
            }
        }
        return probs.count - 1
    }

    /// Returns the `k` highest-scoring entries in descending score order.
    ///
    /// Ties prefer the lower index. If `k > logits.count`, all entries are
    /// returned.
    static func selectTopK(logits: [Float], k: Int) -> [(token: Int, score: Float)] {
        guard k > 0, !logits.isEmpty else { return [] }
        let limit = min(k, logits.count)
        let indexed = logits.enumerated().map { (token: $0.offset, score: $0.element) }
        let sorted = indexed.sorted { lhs, rhs in
            if lhs.score == rhs.score {
                return lhs.token < rhs.token
            }
            return lhs.score > rhs.score
        }
        return Array(sorted.prefix(limit))
    }

    /// Top-p (nucleus) filter.
    ///
    /// Sorts `probs` descending, keeps the smallest prefix whose cumulative mass reaches `threshold`, zeros the rest, and
    /// renormalises so the result sums to 1. A `threshold` of `1.0` returns
    /// the input unchanged. Values at or below `0` would degenerate to a
    /// single-token distribution; the caller is expected to validate the
    /// option first.
    static func topPFilter(probs: [Float], threshold: Float) -> [Float] {
        guard !probs.isEmpty else { return [] }
        if threshold >= 1.0 { return probs }

        let indexed = probs.enumerated().map { (index: $0.offset, value: $0.element) }
        let sorted = indexed.sorted { lhs, rhs in
            if lhs.value == rhs.value {
                return lhs.index < rhs.index
            }
            return lhs.value > rhs.value
        }

        var keep = Set<Int>()
        var cumulative: Float = 0
        for entry in sorted {
            keep.insert(entry.index)
            cumulative += entry.value
            if cumulative >= threshold { break }
        }

        var filtered = [Float](repeating: 0, count: probs.count)
        var sum: Float = 0
        for index in keep {
            filtered[index] = probs[index]
            sum += probs[index]
        }
        guard sum > 0 else {
            return [Float](repeating: 1.0 / Float(probs.count), count: probs.count)
        }
        for i in 0..<filtered.count {
            filtered[i] /= sum
        }
        return filtered
    }

    /// Applies Whisper's repetition penalty in place.
    ///
    /// Tokens that already appeared in `previousTokens` have their positive logits divided by
    /// `penalty` and negative logits multiplied; both shrink the relative
    /// likelihood of repeating the token. Out-of-range token ids are
    /// silently ignored.
    static func applyRepetitionPenalty(
        logits: inout [Float],
        previousTokens: [Int],
        penalty: Float
    ) {
        guard penalty != 1.0 else { return }
        for tokenId in previousTokens where tokenId >= 0 && tokenId < logits.count {
            let value = logits[tokenId]
            logits[tokenId] = value > 0 ? value / penalty : value * penalty
        }
    }

    /// Adds the per-token bias values to `logits` in place.
    ///
    /// Out-of-range token ids are silently ignored.
    static func applyLogitBias(logits: inout [Float], bias: [Int: Float]) {
        for (tokenId, delta) in bias where tokenId >= 0 && tokenId < logits.count {
            logits[tokenId] += delta
        }
    }

    /// Length-based proxy for gzip compression ratio.
    ///
    /// Whisper's hallucination failure mode is to loop on a few tokens, which collapses the unique
    /// character count and inflates the ratio. A healthy English sentence
    /// stays under ~2.0; pathological loops climb past 3.
    ///
    /// Empty text returns `0` so the caller's threshold check accepts the
    /// value (no decoded content cannot be a hallucination by definition).
    static func compressionRatio(text: String) -> Float {
        guard !text.isEmpty else { return 0 }
        let unique = Set(text).count
        if unique == 0 { return 0 }
        return Float(text.count) / Float(unique)
    }

    /// Reads the softmax probability of `noSpeechToken` from the raw logits.
    ///
    /// Used by the decoder at the first generation step to flag silent
    /// segments so the temperature fallback loop can skip them.
    static func noSpeechProbability(logits: [Float], noSpeechToken: Int) -> Float {
        guard noSpeechToken >= 0, noSpeechToken < logits.count else { return 0 }
        let probs = softmax(logits: logits, temperature: 1.0)
        return probs[noSpeechToken]
    }

    /// Splits a token stream into segments using Whisper's timestamp markers.
    ///
    /// Looks for `<|t_start|> ... <|t_end|>` timestamp pairs. Only invoked
    /// when `DecodingOptions.withoutTimestamps` is `false` so the decoder
    /// actually emits the timestamp markers.
    ///
    /// Rules:
    ///
    /// - Tokens before the first timestamp are ignored (no boundary yet).
    /// - A start timestamp without a matching end terminates parsing; the
    ///   trailing tokens are dropped rather than emitted as an open-ended
    ///   segment.
    /// - `windowOffsetSeconds` is added to the per-segment `start` and `end`
    ///   so callers can stitch sliding-window decoding without bookkeeping.
    /// - Non-timestamp special tokens between text are passed through
    ///   `tokenizer.decode` which already strips them from the visible text.
    public static func parseSegments(
        tokens: [WhisperToken],
        tokenizer: WhisperTokenizer,
        windowOffsetSeconds: TimeInterval
    ) -> [TranscriptionSegment] {
        var segments: [TranscriptionSegment] = []
        var startSeconds: TimeInterval?
        var bodyTokens: [Int] = []

        for token in tokens {
            if tokenizer.isTimestamp(token: token.id) {
                let seconds = timestampSeconds(forTokenId: token.id)
                if let start = startSeconds {
                    let text = tokenizer.decode(tokens: bodyTokens)
                    segments.append(
                        TranscriptionSegment(
                            text: text,
                            start: start + windowOffsetSeconds,
                            end: seconds + windowOffsetSeconds
                        )
                    )
                    startSeconds = nil
                    bodyTokens.removeAll(keepingCapacity: true)
                } else {
                    startSeconds = seconds
                    bodyTokens.removeAll(keepingCapacity: true)
                }
                continue
            }
            if startSeconds != nil {
                bodyTokens.append(token.id)
            }
        }
        return segments
    }

    /// Whisper timestamp token IDs occupy a contiguous range starting at
    /// `<|0.00|>` (the first ID returned by `firstTimestampId`) at 0.02-second
    /// resolution.
    ///
    /// The token ID layout is fixed by the tokenizer; we use the reference Whisper offset of 50_364 because the tokenizer table is
    /// loaded from the vendor `tokenizer.json` and uses the same constants.
    static func timestampSeconds(forTokenId id: Int) -> TimeInterval {
        let firstTimestampId = 50_364
        let increment: TimeInterval = 0.02
        let steps = max(0, id - firstTimestampId)
        return TimeInterval(steps) * increment
    }
}

/// Wrappers around the libm primitives so the actor body and helpers stay
/// free of `import Darwin` noise.
@inline(__always)
private func expFloat(_ x: Float) -> Float {
    Foundation.exp(x)
}

@inline(__always)
private func logFloat(_ x: Float) -> Float {
    Foundation.log(x)
}
