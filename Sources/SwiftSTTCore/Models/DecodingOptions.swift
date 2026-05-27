import Foundation

/// What the decoder should produce.
///
/// Whisper supports two tasks. Use ``transcribe`` to keep the original language,
/// or ``translate`` to translate non-English input into English text. The model
/// does the language detection step internally before either task.
public enum TaskKind: String, Sendable, Equatable {

    /// Output text in the same language as the audio.
    case transcribe

    /// Translate the audio into English regardless of the source language.
    case translate
}

/// Tunable parameters that control how the decoder samples tokens.
///
/// Most callers can use ``default`` and only change one or two fields. The most
/// useful overrides are usually:
///
/// - ``language`` to skip auto-detection when the source language is known.
/// - ``temperature`` raised above zero on noisy audio where greedy decoding gets stuck.
/// - ``beamSize`` lowered to 1 for lower latency at the cost of accuracy.
///
/// The defaults match the reference Python `whisper` implementation, so output
/// should look the same when the rest of the pipeline is bit-exact.
public struct DecodingOptions: Sendable, Equatable {

    /// ISO-639-1 language code (`"en"`, `"de"`, `"ja"`, ...).
    ///
    /// When `nil`, Whisper runs a quick language detection pass on the first audio segment before
    /// decoding the rest.
    public var language: String?

    /// Whether to transcribe in the source language or translate to English.
    public var task: TaskKind

    /// Sampling temperature for the softmax.
    ///
    /// `0` means greedy decoding. Values up to about `0.6` are useful when greedy gets caught in a repetition loop.
    public var temperature: Float

    /// Beam search width. `1` is greedy or temperature sampling depending on.
    ///
    /// ``temperature``. Higher values trade compute for accuracy.
    ///
    /// SwiftSTT's beam path shares a single Core ML KV cache between beams
    /// and re-prefills per step, so values above `1` are functionally correct
    /// but slow. The reference Python implementation defaults to `5`; this port
    /// defaults to `1` until per-beam state lands.
    public var beamSize: Int

    /// Whether to feed the decoder a `<|prevtimestamp|>` prefix from the previous
    /// segment.
    ///
    /// Helps consistency across long audio at the cost of one extra token.
    public var usePrefixTimestamps: Bool

    /// When `true` (the default), the decoder injects `<|notimestamps|>` into the
    /// prompt prefix and Whisper produces text-only output.
    ///
    /// Set to `false` to make the decoder emit timestamp tokens for `parseSegments` to split into
    /// ``TranscriptionSegment`` boundaries.
    public var withoutTimestamps: Bool

    /// Whether to suppress the blank token at position 0.
    ///
    /// Disabling this lets the decoder emit silence as empty output, which is occasionally useful for
    /// alignment but usually harmful.
    public var suppressBlank: Bool

    /// Token IDs to forbid the decoder from picking.
    ///
    /// Useful for blocking specific vocabulary that shows up as hallucination on silent audio. Empty by default.
    public var suppressTokens: [Int]

    /// Temperatures to retry with when the previous attempt fails the anti-hallucination
    /// thresholds.
    ///
    /// The decoder runs the first temperature, checks the result against ``logProbThreshold`` and ``compressionRatioThreshold``, and falls back to the
    /// next temperature if either check fails. The reference Python implementation
    /// uses `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
    public var temperatureFallback: [Float]

    /// Average per-token log-probability below which an attempt is considered
    /// degenerate and triggers temperature fallback.
    ///
    /// Reference value is `-1.0`.
    public var logProbThreshold: Float

    /// Probability of the no-speech token at the first generation step.
    ///
    /// When the observed value exceeds this threshold AND the average log-probability falls
    /// below ``logProbThreshold``, the segment is treated as silence and the decoder
    /// returns no tokens for it. Reference value is `0.6`.
    public var noSpeechThreshold: Float

    /// Decoded-text length divided by unique-character count.
    ///
    /// Whisper's failure mode is to loop on a few tokens (`"the the the ..."`) which collapses unique-char
    /// count and pushes this ratio up. Above the threshold triggers fallback.
    /// Reference value is `2.4`.
    public var compressionRatioThreshold: Float

    /// Top-p (nucleus) sampling cutoff.
    ///
    /// After softmax the probabilities are sorted descending; the smallest set whose mass reaches `topP` is kept and the rest
    /// is zeroed. `1.0` (the default) keeps the full distribution intact. Only used
    /// when ``temperature`` is positive.
    public var topP: Float

    /// Multiplicative penalty applied to logits of tokens that already appeared in
    /// the running output.
    ///
    /// Positive logits get divided by `repetitionPenalty`, negative logits multiplied; both push the token's probability down. `1.0`
    /// (the default) disables the penalty.
    public var repetitionPenalty: Float

    /// Additive bias applied per-token to the raw logits before softmax.
    ///
    /// Positive values raise a token's likelihood, negative values lower it. Out-of-range
    /// keys are silently ignored.
    public var logitBias: [Int: Float]

    /// Creates a new DecodingOptions with the supplied values.
    public init(
        language: String? = nil,
        task: TaskKind = .transcribe,
        temperature: Float = 0.0,
        beamSize: Int = 1,
        usePrefixTimestamps: Bool = true,
        suppressBlank: Bool = true,
        suppressTokens: [Int] = [],
        withoutTimestamps: Bool = true,
        temperatureFallback: [Float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        logProbThreshold: Float = -1.0,
        noSpeechThreshold: Float = 0.6,
        compressionRatioThreshold: Float = 2.4,
        topP: Float = 1.0,
        repetitionPenalty: Float = 1.0,
        logitBias: [Int: Float] = [:]
    ) {
        self.language = language
        self.task = task
        self.temperature = temperature
        self.beamSize = beamSize
        self.usePrefixTimestamps = usePrefixTimestamps
        self.suppressBlank = suppressBlank
        self.suppressTokens = suppressTokens
        self.withoutTimestamps = withoutTimestamps
        self.temperatureFallback = temperatureFallback
        self.logProbThreshold = logProbThreshold
        self.noSpeechThreshold = noSpeechThreshold
        self.compressionRatioThreshold = compressionRatioThreshold
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.logitBias = logitBias
    }

    /// Sensible defaults for English transcription with beam search.
    public static let `default` = DecodingOptions()
}
