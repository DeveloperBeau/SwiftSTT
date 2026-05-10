import Foundation
import SwiftWhisperCore

/// Whisper's BPE tokenizer with special-token awareness.
///
/// Wraps a ``BPETokenizer`` plus the table of Whisper-specific special tokens
/// (`<|startoftranscript|>`, language tags, `<|transcribe|>`/`<|translate|>`,
/// `<|notimestamps|>`, timestamp markers, `<|endoftext|>`).
///
/// Special tokens are matched verbatim in the input rather than going through
/// BPE. The vocabulary file's `added_tokens` section lists each special token
/// with its assigned ID, which gets respected on encode and decode.
///
/// Whisper's pre-tokenization splits text using a GPT-2-style regex. We use a
/// simplified split (whitespace-aware) that handles the common case. For
/// bit-exact reference parity with Python `whisper`, a future revision may add
/// the full regex.
public struct WhisperTokenizer: Sendable {

    private let bpe: BPETokenizer
    private let specialTokens: [String: Int]   // content -> id
    private let reverseSpecial: [Int: String]  // id -> content
    private let timestampPrefix = "<|"
    private let timestampSuffix = "|>"

    public let endOfTextToken: Int
    public let startOfTranscriptToken: Int
    public let noTimestampsToken: Int
    public let transcribeToken: Int
    public let translateToken: Int

    /// Whisper's `<|nospeech|>` marker. The decoder reads its softmax
    /// probability at the first generation step to decide whether the segment
    /// is silence and should be skipped.
    public let noSpeechToken: Int

    public init(bpe: BPETokenizer = BPETokenizer(), specialTokens: [String: Int] = [:]) {
        self.bpe = bpe
        self.specialTokens = specialTokens
        var reverse: [Int: String] = [:]
        for (content, id) in specialTokens {
            reverse[id] = content
        }
        self.reverseSpecial = reverse

        self.endOfTextToken = specialTokens["<|endoftext|>"] ?? 50_257
        self.startOfTranscriptToken = specialTokens["<|startoftranscript|>"] ?? 50_258
        self.noTimestampsToken = specialTokens["<|notimestamps|>"] ?? 50_363
        self.transcribeToken = specialTokens["<|transcribe|>"] ?? 50_359
        self.translateToken = specialTokens["<|translate|>"] ?? 50_358
        self.noSpeechToken = specialTokens["<|nospeech|>"] ?? specialTokens["<|nocaptions|>"] ?? 50_362
    }

    /// Loads a tokenizer from a HuggingFace `tokenizer.json` file on disk.
    public init(contentsOf url: URL) throws(SwiftWhisperError) {
        do {
            let data = try Data(contentsOf: url)
            let decoded = try JSONDecoder().decode(TokenizerJSON.self, from: data)
            let bpe = BPETokenizer(json: decoded)
            var specials: [String: Int] = [:]
            for added in decoded.addedTokens {
                specials[added.content] = added.id
            }
            self.init(bpe: bpe, specialTokens: specials)
        } catch {
            throw .modelLoadFailed("tokenizer.json: \(error.localizedDescription)")
        }
    }

    // MARK: - Encoding

    /// Encodes text into Whisper token IDs. Special tokens (`<|...|>`) match
    /// verbatim and are emitted as their direct ID. The remaining text is
    /// pre-tokenized by whitespace, byte-mapped, and BPE-merged.
    public func encode(text: String) -> [Int] {
        guard !specialTokens.isEmpty else {
            return bpe.encode(chunk: text)
        }
        return splitOnSpecialTokens(text).flatMap { fragment -> [Int] in
            switch fragment {
            case .special(let id):
                return [id]
            case .text(let chunk):
                return preTokenize(chunk).flatMap { bpe.encode(chunk: $0) }
            }
        }
    }

    /// Decodes a sequence of token IDs into text. Special tokens are stripped
    /// from the output by default. Use ``decode(tokens:keepingSpecials:)`` to
    /// keep them.
    public func decode(tokens: [Int]) -> String {
        decode(tokens: tokens, keepingSpecials: false)
    }

    public func decode(tokens: [Int], keepingSpecials: Bool) -> String {
        var output = ""
        var bpeRun: [Int] = []

        func flushBPE() {
            if !bpeRun.isEmpty {
                output.append(bpe.decode(tokens: bpeRun))
                bpeRun.removeAll(keepingCapacity: true)
            }
        }

        for token in tokens {
            if let special = reverseSpecial[token] {
                flushBPE()
                if keepingSpecials {
                    output.append(special)
                }
            } else {
                bpeRun.append(token)
            }
        }
        flushBPE()
        return output
    }

    // MARK: - Special token classification

    public func isSpecial(token: Int) -> Bool {
        reverseSpecial[token] != nil
    }

    /// `true` if the token is one of Whisper's `<|x.xx|>` timestamp markers
    /// (e.g. `<|0.00|>`, `<|0.50|>`, ..., `<|30.00|>`).
    public func isTimestamp(token: Int) -> Bool {
        guard let content = reverseSpecial[token] else { return false }
        guard
            content.hasPrefix(timestampPrefix),
            content.hasSuffix(timestampSuffix)
        else { return false }
        let inner = String(content.dropFirst(2).dropLast(2))
        return Double(inner) != nil
    }

    // MARK: - Internals

    private enum Fragment {
        case special(Int)
        case text(String)
    }

    /// Splits the input on any literal special-token string. Order of scanning
    /// matters for nested matches; we sort by descending length so the longest
    /// match wins when prefixes overlap.
    private func splitOnSpecialTokens(_ text: String) -> [Fragment] {
        let tokens = specialTokens.keys.sorted { $0.count > $1.count }
        var fragments: [Fragment] = []
        var remaining = text[...]

        while !remaining.isEmpty {
            var matched = false
            for token in tokens {
                if remaining.hasPrefix(token) {
                    fragments.append(.special(specialTokens[token]!))
                    remaining = remaining.dropFirst(token.count)
                    matched = true
                    break
                }
            }
            if matched { continue }

            // Find next special-token start, take everything before it as text.
            let nextSpecialStart: Substring.Index? = remaining.indices.first { idx in
                tokens.contains { remaining[idx...].hasPrefix($0) }
            }
            let endIdx = nextSpecialStart ?? remaining.endIndex
            fragments.append(.text(String(remaining[remaining.startIndex..<endIdx])))
            remaining = remaining[endIdx...]
        }
        return fragments
    }

    /// Whisper's pre-tokenizer is regex-driven. This simplified version splits
    /// on whitespace boundaries while keeping leading whitespace attached to
    /// the following word, which matches the most common BPE expectation.
    /// Good enough for the BPE merge step on most ASCII English input.
    private func preTokenize(_ text: String) -> [String] {
        guard !text.isEmpty else { return [] }
        var result: [String] = []
        var current = ""
        var sawNonSpace = false

        for char in text {
            if char.isWhitespace {
                if sawNonSpace {
                    result.append(current)
                    current = ""
                    sawNonSpace = false
                }
                current.append(char)
            } else {
                sawNonSpace = true
                current.append(char)
            }
        }
        if !current.isEmpty {
            result.append(current)
        }
        return result
    }
}
