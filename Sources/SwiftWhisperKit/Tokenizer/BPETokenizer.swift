import Foundation

/// Byte-pair encoding primitives reusable across tokenizers.
///
/// Whisper's tokenizer is BPE on top of a UTF-8 byte-level alphabet, similar
/// to GPT-2. The merge logic itself is the same regardless of the model, so
/// the plan is to factor it out here once ``WhisperTokenizer`` is implemented
/// and a second tokenizer (probably a sentence-piece variant) lands.
///
/// > Note: Reserved for future use; carries no behaviour yet.
public struct BPETokenizer: Sendable {
    public init() {}
}
