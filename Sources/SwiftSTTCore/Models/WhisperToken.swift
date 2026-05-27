import Foundation

/// A single token produced by the Whisper decoder.
///
/// Whisper's vocabulary holds about 50 000 entries plus a handful of special tokens
/// (start-of-transcript, language tags, timestamp markers, end-of-text). Each token
/// the decoder emits comes wrapped in this struct so callers can tell apart text
/// content, confidence scores and timing.
///
/// Probability is taken from the softmax over the decoder's final logit. A value
/// below about 0.4 is a useful signal that the model is uncertain, which often
/// correlates with hallucinations on silent audio.
public struct WhisperToken: Sendable, Equatable {

    /// The token's index in the Whisper BPE vocabulary.
    public let id: Int

    /// The decoded text fragment.
    ///
    /// May be a partial word, a punctuation mark, or an empty string for special tokens.
    public let text: String

    /// Softmax probability of this token at the decoding step that produced it.
    ///
    /// Defaults to `1.0` for tokens that were not sampled (for example, prompt prefixes).
    public let probability: Float

    /// Timestamp of the token in seconds from the start of the segment, if Whisper
    /// emitted a timestamp prediction near it. `nil` when the decoder ran without
    /// timestamps or no timestamp token was nearby.
    public let timestamp: TimeInterval?

    /// Creates a new WhisperToken with the supplied values.
    public init(
        id: Int,
        text: String,
        probability: Float = 1.0,
        timestamp: TimeInterval? = nil
    ) {
        self.id = id
        self.text = text
        self.probability = probability
        self.timestamp = timestamp
    }
}
