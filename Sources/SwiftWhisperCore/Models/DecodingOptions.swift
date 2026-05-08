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

    /// ISO-639-1 language code (`"en"`, `"de"`, `"ja"`, ...). When `nil`, Whisper
    /// runs a quick language detection pass on the first audio segment before
    /// decoding the rest.
    public var language: String?

    /// Whether to transcribe in the source language or translate to English.
    public var task: TaskKind

    /// Sampling temperature for the softmax. `0` means greedy decoding. Values up
    /// to about `0.6` are useful when greedy gets caught in a repetition loop.
    public var temperature: Float

    /// Beam search width. `1` is greedy. Higher values trade compute for accuracy;
    /// `5` is the reference default.
    public var beamSize: Int

    /// Whether to feed the decoder a `<|prevtimestamp|>` prefix from the previous
    /// segment. Helps consistency across long audio at the cost of one extra token.
    public var usePrefixTimestamps: Bool

    /// Whether to suppress the blank token at position 0. Disabling this lets the
    /// decoder emit silence as empty output, which is occasionally useful for
    /// alignment but usually harmful.
    public var suppressBlank: Bool

    /// Token IDs to forbid the decoder from picking. Useful for blocking specific
    /// vocabulary that shows up as hallucination on silent audio. Empty by default.
    public var suppressTokens: [Int]

    public init(
        language: String? = nil,
        task: TaskKind = .transcribe,
        temperature: Float = 0.0,
        beamSize: Int = 5,
        usePrefixTimestamps: Bool = true,
        suppressBlank: Bool = true,
        suppressTokens: [Int] = []
    ) {
        self.language = language
        self.task = task
        self.temperature = temperature
        self.beamSize = beamSize
        self.usePrefixTimestamps = usePrefixTimestamps
        self.suppressBlank = suppressBlank
        self.suppressTokens = suppressTokens
    }

    /// Sensible defaults for English transcription with beam search.
    public static let `default` = DecodingOptions()
}
