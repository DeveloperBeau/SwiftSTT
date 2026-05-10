import Foundation

/// Partial Codable view of HuggingFace `tokenizer.json` capturing only the
/// fields the Whisper BPE tokenizer needs. The HuggingFace schema is much
/// larger (pre-tokenizer config, normalizer, decoder); we ignore the rest
/// because Whisper's pre-tokenization, normalization, and decoding behaviour
/// are already known and hardcoded into ``BPETokenizer``.
public struct TokenizerJSON: Decodable, Sendable {
    public let model: Model
    public let addedTokens: [AddedToken]

    enum CodingKeys: String, CodingKey {
        case model
        case addedTokens = "added_tokens"
    }

    /// The `model` block of `tokenizer.json`: vocabulary and merge rules.
    public struct Model: Decodable, Sendable {
        public let vocab: [String: Int]
        public let merges: [Merge]

        /// HuggingFace ships `merges` in two historical formats:
        /// - Old: `["Ġ t", "Ġ a", ...]` (space-separated string)
        /// - New: `[["Ġ", "t"], ["Ġ", "a"], ...]` (array of pairs)
        public struct Merge: Decodable, Sendable {
            public let first: String
            public let second: String

            public init(from decoder: any Decoder) throws {
                if let pair = try? [String](from: decoder), pair.count == 2 {
                    self.first = pair[0]
                    self.second = pair[1]
                    return
                }
                let str = try String(from: decoder)
                let parts = str.split(separator: " ", maxSplits: 1).map(String.init)
                guard parts.count == 2 else {
                    throw DecodingError.dataCorrupted(.init(
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
        public let id: Int
        public let content: String
    }
}
