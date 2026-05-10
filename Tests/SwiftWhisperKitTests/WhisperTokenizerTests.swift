import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

@Suite("WhisperTokenizer")
struct WhisperTokenizerTests {

    /// Builds a tokenizer with a synthetic ASCII-only vocab and standard
    /// Whisper specials. Lets us test encode/decode/special handling without
    /// needing the real 50k-entry tokenizer.json.
    static func makeFixtureTokenizer() -> WhisperTokenizer {
        var vocab: [String: Int] = [:]
        for (i, byte) in (0x21...0x7E).enumerated() {
            let char = BPETokenizer.byteToUnicode[UInt8(byte)]!
            vocab[String(char)] = i
        }
        // Add the space character mapping too so " hello" works.
        let space = BPETokenizer.byteToUnicode[UInt8(ascii: " ")]!
        vocab[String(space)] = 200

        let bpe = BPETokenizer(vocab: vocab, merges: [])
        let specials: [String: Int] = [
            "<|endoftext|>": 50_257,
            "<|startoftranscript|>": 50_258,
            "<|en|>": 50_259,
            "<|transcribe|>": 50_359,
            "<|translate|>": 50_358,
            "<|notimestamps|>": 50_363,
            "<|0.00|>": 50_364,
            "<|0.50|>": 50_414,
            "<|30.00|>": 51_864,
        ]
        return WhisperTokenizer(bpe: bpe, specialTokens: specials)
    }

    @Test("Default initializer exposes Whisper-standard special token IDs")
    func defaultSpecials() {
        let tok = WhisperTokenizer()
        #expect(tok.endOfTextToken == 50_257)
        #expect(tok.startOfTranscriptToken == 50_258)
        #expect(tok.noTimestampsToken == 50_363)
        #expect(tok.transcribeToken == 50_359)
        #expect(tok.translateToken == 50_358)
        #expect(tok.noSpeechToken == 50_362)
    }

    @Test("noSpeechToken honours an explicit override in the specials map")
    func noSpeechTokenOverride() {
        let tok = WhisperTokenizer(specialTokens: ["<|nospeech|>": 12_345])
        #expect(tok.noSpeechToken == 12_345)
    }

    @Test("noSpeechToken falls back to the legacy nocaptions name when present")
    func noSpeechTokenLegacyFallback() {
        let tok = WhisperTokenizer(specialTokens: ["<|nocaptions|>": 99_999])
        #expect(tok.noSpeechToken == 99_999)
    }

    @Test("Encodes plain text via BPE")
    func encodePlainText() {
        let tok = Self.makeFixtureTokenizer()
        let result = tok.encode(text: "hi")
        #expect(result.count == 2)
    }

    @Test("Encodes special tokens by direct ID lookup")
    func encodeSpecial() {
        let tok = Self.makeFixtureTokenizer()
        let result = tok.encode(text: "<|startoftranscript|><|en|><|transcribe|>")
        #expect(result == [50_258, 50_259, 50_359])
    }

    @Test("Mixed special + text encodes correctly")
    func encodeMixed() {
        let tok = Self.makeFixtureTokenizer()
        let result = tok.encode(text: "<|startoftranscript|>hi<|endoftext|>")
        #expect(result.first == 50_258)
        #expect(result.last == 50_257)
        #expect(result.count >= 4)
    }

    @Test("Decode strips specials by default")
    func decodeStripsSpecials() {
        let tok = Self.makeFixtureTokenizer()
        let encoded = tok.encode(text: "<|startoftranscript|>hi<|endoftext|>")
        let decoded = tok.decode(tokens: encoded)
        #expect(decoded == "hi")
    }

    @Test("Decode keeps specials when requested")
    func decodeKeepsSpecials() {
        let tok = Self.makeFixtureTokenizer()
        let encoded = tok.encode(text: "<|startoftranscript|>hi<|endoftext|>")
        let decoded = tok.decode(tokens: encoded, keepingSpecials: true)
        #expect(decoded == "<|startoftranscript|>hi<|endoftext|>")
    }

    @Test("isSpecial identifies known specials")
    func isSpecialPositive() {
        let tok = Self.makeFixtureTokenizer()
        #expect(tok.isSpecial(token: 50_257) == true)
        #expect(tok.isSpecial(token: 50_363) == true)
    }

    @Test("isSpecial returns false for normal tokens")
    func isSpecialNegative() {
        let tok = Self.makeFixtureTokenizer()
        #expect(tok.isSpecial(token: 0) == false)
        #expect(tok.isSpecial(token: 999) == false)
    }

    @Test("isTimestamp identifies timestamp tokens")
    func isTimestampPositive() {
        let tok = Self.makeFixtureTokenizer()
        #expect(tok.isTimestamp(token: 50_364) == true)  // <|0.00|>
        #expect(tok.isTimestamp(token: 50_414) == true)  // <|0.50|>
        #expect(tok.isTimestamp(token: 51_864) == true)  // <|30.00|>
    }

    @Test("isTimestamp rejects non-timestamp specials")
    func isTimestampNegative() {
        let tok = Self.makeFixtureTokenizer()
        #expect(tok.isTimestamp(token: 50_257) == false)  // endoftext
        #expect(tok.isTimestamp(token: 50_259) == false)  // language tag
    }

    @Test("Round-trip preserves ASCII text")
    func roundTrip() {
        let tok = Self.makeFixtureTokenizer()
        let original = "Hello, world!"
        let decoded = tok.decode(tokens: tok.encode(text: original))
        #expect(decoded == original)
    }

    @Test("Encoding empty string yields empty token list")
    func encodeEmpty() {
        let tok = Self.makeFixtureTokenizer()
        #expect(tok.encode(text: "") == [])
    }

    @Test("Decoding empty list yields empty string")
    func decodeEmpty() {
        let tok = Self.makeFixtureTokenizer()
        #expect(tok.decode(tokens: []) == "")
    }

    @Test("Loading tokenizer from JSON file")
    func loadFromJSON() throws {
        let json = """
            {
              "added_tokens": [
                {"id": 50257, "content": "<|endoftext|>"}
              ],
              "model": {
                "vocab": {"h": 0, "i": 1},
                "merges": ["h i"]
              }
            }
            """
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).json")
        try Data(json.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let tok = try WhisperTokenizer(contentsOf: url)
        #expect(tok.endOfTextToken == 50_257)
    }

    @Test("Loading tokenizer from missing file throws")
    func loadFromMissingFile() {
        let url = URL(fileURLWithPath: "/tmp/does-not-exist-\(UUID()).json")
        do {
            _ = try WhisperTokenizer(contentsOf: url)
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            if case .modelLoadFailed = error {
            } else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    @Test("Tokenizer.json supports array-of-pairs merges format")
    func mergesArrayFormat() throws {
        let json = """
            {
              "added_tokens": [],
              "model": {
                "vocab": {"a": 0, "b": 1},
                "merges": [["a", "b"]]
              }
            }
            """
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID()).json")
        try Data(json.utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let tok = try WhisperTokenizer(contentsOf: url)
        // Successfully constructed without throwing.
        let _ = tok.endOfTextToken
    }
}
