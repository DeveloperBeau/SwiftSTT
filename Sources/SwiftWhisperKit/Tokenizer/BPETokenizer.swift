import Foundation

/// Byte-level BPE encoder/decoder used by Whisper.
///
/// Whisper's tokenizer follows GPT-2's design:
///
/// 1. **Byte-to-unicode mapping.** Each input byte (0 to 255) is mapped to a
///    unique unicode code point in a printable range. This avoids dealing
///    with control characters, whitespace, and multi-byte UTF-8 sequences in
///    the BPE algorithm itself. The vocabulary is built over these mapped
///    characters, not raw bytes.
/// 2. **BPE merges.** Starting from one mapped character per token, repeatedly
///    merge the adjacent pair with the lowest merge rank until no merge in the
///    table applies. The result is a list of string tokens.
/// 3. **Vocabulary lookup.** Each merged token maps to an integer ID via the
///    vocabulary table.
///
/// Decoding reverses the chain: token IDs to strings via the reverse vocab,
/// then the concatenated string is mapped back from unicode characters to
/// bytes, then interpreted as UTF-8.
public struct BPETokenizer: Sendable {

    /// Maps each byte (0...255) to its assigned unicode character.
    public static let byteToUnicode: [UInt8: Character] = makeByteToUnicode()

    /// Reverse lookup for decoding.
    public static let unicodeToByte: [Character: UInt8] = {
        var m: [Character: UInt8] = [:]
        for (b, c) in byteToUnicode { m[c] = b }
        return m
    }()

    private let vocab: [String: Int]
    private let reverseVocab: [Int: String]
    private let mergeRanks: [String: Int]

    public init(
        vocab: [String: Int],
        merges: [(String, String)]
    ) {
        self.vocab = vocab
        var reverse: [Int: String] = [:]
        reverse.reserveCapacity(vocab.count)
        for (token, id) in vocab {
            reverse[id] = token
        }
        self.reverseVocab = reverse

        var ranks: [String: Int] = [:]
        ranks.reserveCapacity(merges.count)
        for (rank, pair) in merges.enumerated() {
            ranks["\(pair.0)\u{0001}\(pair.1)"] = rank
        }
        self.mergeRanks = ranks
    }

    public init() {
        self.init(vocab: [:], merges: [])
    }

    public init(json: TokenizerJSON) {
        let merges = json.model.merges.map { ($0.first, $0.second) }
        self.init(vocab: json.model.vocab, merges: merges)
    }

    public var vocabularySize: Int { vocab.count }

    /// Look up a token ID. Returns `nil` for unknown tokens.
    public func id(for token: String) -> Int? { vocab[token] }

    /// Look up a token string. Returns `nil` for unknown IDs.
    public func token(for id: Int) -> String? { reverseVocab[id] }

    // MARK: - Encoding

    /// Encodes a chunk of text (already split by pre-tokenization) into token IDs.
    /// Bytes are mapped to unicode chars, then BPE-merged.
    public func encode(chunk text: String) -> [Int] {
        let mapped = mapBytesToUnicode(text)
        let merged = bpe(token: mapped)
        return merged.compactMap { vocab[$0] }
    }

    private func mapBytesToUnicode(_ text: String) -> String {
        var result = ""
        result.reserveCapacity(text.utf8.count)
        for byte in text.utf8 {
            if let mapped = Self.byteToUnicode[byte] {
                result.append(mapped)
            }
        }
        return result
    }

    /// Greedy BPE merge over a token's characters.
    private func bpe(token: String) -> [String] {
        guard token.count > 1 else { return [token] }
        var word: [String] = token.map { String($0) }

        while word.count > 1 {
            var bestRank = Int.max
            var bestIndex = -1
            for i in 0..<(word.count - 1) {
                let key = "\(word[i])\u{0001}\(word[i + 1])"
                if let rank = mergeRanks[key], rank < bestRank {
                    bestRank = rank
                    bestIndex = i
                }
            }
            if bestIndex == -1 { break }
            word.replaceSubrange(bestIndex...(bestIndex + 1), with: [word[bestIndex] + word[bestIndex + 1]])
        }
        return word
    }

    // MARK: - Decoding

    /// Decodes a sequence of token IDs back into text. Unknown IDs are skipped.
    public func decode(tokens: [Int]) -> String {
        let joined = tokens.compactMap { reverseVocab[$0] }.joined()
        var bytes: [UInt8] = []
        bytes.reserveCapacity(joined.count)
        for char in joined {
            if let byte = Self.unicodeToByte[char] {
                bytes.append(byte)
            }
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    // MARK: - Byte/unicode mapping construction

    /// Builds GPT-2's byte-to-unicode dictionary. Bytes that print cleanly
    /// keep their natural code point; the rest get mapped into the
    /// `0x100`+ range so every byte has a printable representative.
    private static func makeByteToUnicode() -> [UInt8: Character] {
        var bs: [Int] = []
        bs.append(contentsOf: Int(Character("!").asciiValue!)...Int(Character("~").asciiValue!))
        bs.append(contentsOf: 0xA1...0xAC)
        bs.append(contentsOf: 0xAE...0xFF)

        var cs = bs
        var n = 0
        for b in 0..<256 where !bs.contains(b) {
            bs.append(b)
            cs.append(256 + n)
            n += 1
        }

        var map: [UInt8: Character] = [:]
        for i in 0..<bs.count {
            let byte = UInt8(bs[i])
            let scalar = Unicode.Scalar(cs[i])!
            map[byte] = Character(scalar)
        }
        return map
    }
}
