import Foundation

/// Whisper's BPE tokenizer.
///
/// Whisper uses byte-pair encoding with a vocabulary of about 50 257 entries
/// for the multilingual models, plus around a hundred special tokens that mark
/// language, task, timestamps, no-speech, and end-of-text. The vocabulary lives
/// in the `tokenizer.json` file that ships with each model from HuggingFace.
///
/// ## Special tokens
///
/// Whisper drives a lot of behaviour through special tokens rather than
/// configuration flags. The decoder is primed with a sequence like
/// `<|startoftranscript|><|en|><|transcribe|><|notimestamps|>` and the
/// generated text starts after that. Timestamps appear inline as
/// `<|0.00|>`, `<|0.50|>`, ..., `<|30.00|>` tokens.
///
/// > Important: Stub. Real BPE merge logic lands in milestone M5.
public actor WhisperTokenizer {
    public init() {}

    /// Encodes a UTF-8 string into BPE token IDs.
    public func encode(text: String) -> [Int] { [] }

    /// Decodes a sequence of token IDs back into text. Special tokens are
    /// stripped; timestamp tokens render as their numeric value.
    public func decode(tokens: [Int]) -> String { "" }

    /// `true` if the token is one of Whisper's `<|x.xx|>` timestamp markers.
    public func isTimestamp(token: Int) -> Bool { false }

    /// `true` if the token is a special token (anything outside the BPE vocab).
    public func isSpecial(token: Int) -> Bool { false }

    /// ID of the `<|endoftext|>` token. Decoding stops when this is sampled.
    public var endOfTextToken: Int { 50_257 }

    /// ID of the `<|startoftranscript|>` token used to prime decoding.
    public var startOfTranscriptToken: Int { 50_258 }
}
