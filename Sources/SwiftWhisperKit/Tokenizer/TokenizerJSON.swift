import Foundation

/// Partial Codable view of HuggingFace `tokenizer.json` capturing only the
/// fields the Whisper BPE tokenizer needs.
///
/// The HuggingFace schema is much larger (pre-tokenizer config, normalizer, decoder); we ignore the rest
/// because Whisper's pre-tokenization, normalization, and decoding behaviour
/// are already known and hardcoded into ``BPETokenizer``.
public struct TokenizerJSON: Decodable, Sendable {
    /// BPE model section of the tokenizer JSON.
    public let model: Model
    /// Special tokens appended to the vocabulary.
    public let addedTokens: [AddedToken]

    enum CodingKeys: String, CodingKey {
        case model
        case addedTokens = "added_tokens"
    }

    /// The `model` block of `tokenizer.json`: vocabulary and merge rules.
    public struct Model: Decodable, Sendable {
        /// Token-string-to-id mapping.
        public let vocab: [String: Int]
        /// Ordered BPE merge rules.
        public let merges: [Merge]

        /// One BPE merge rule.
        ///
        /// HuggingFace ships `merges` in two historical formats. The old form
        /// is `["Ġ t", "Ġ a", ...]` (space-separated string) and the new form
        /// is `[["Ġ", "t"], ["Ġ", "a"], ...]` (array of pairs).
        public struct Merge: Decodable, Sendable {
            /// First token in the merge pair.
            public let first: String
            /// Second token in the merge pair.
            public let second: String

            /// Creates a new Merge with the supplied values.
            public init(from decoder: any Decoder) throws {
                if let pair = try? [String](from: decoder), pair.count == 2 {
                    self.first = pair[0]
                    self.second = pair[1]
                    return
                }
                let str = try String(from: decoder)
                let parts = str.split(separator: " ", maxSplits: 1).map(String.init)
                guard parts.count == 2 else {
                    throw DecodingError.dataCorrupted(
                        .init(
                            codingPath: decoder.codingPath,
                            debugDescription: "merge entry not parseable: \(str)"
                        ))
                }
                self.first = parts[0]
                self.second = parts[1]
            }
        }
    }

    /// One entry from `added_tokens`, used for special tokens (BOS, EOT, etc).
    public struct AddedToken: Decodable, Sendable {
        /// Stable identifier for the model variant.
        public let id: Int
        /// String content of the added token.
        public let content: String
    }
}
