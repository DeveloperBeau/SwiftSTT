import Foundation
import Testing
@testable import SwiftWhisperKit

@Suite("BPETokenizer")
struct BPETokenizerTests {

    @Test("Byte-to-unicode covers all 256 bytes")
    func byteMapCoversAllBytes() {
        for byte in 0...255 {
            #expect(BPETokenizer.byteToUnicode[UInt8(byte)] != nil)
        }
    }

    @Test("Unicode-to-byte is the inverse of byte-to-unicode")
    func byteMapInvertible() {
        for byte in 0...255 {
            let mapped = BPETokenizer.byteToUnicode[UInt8(byte)]!
            #expect(BPETokenizer.unicodeToByte[mapped] == UInt8(byte))
        }
    }

    @Test("Empty tokenizer encodes nothing")
    func emptyTokenizer() {
        let bpe = BPETokenizer()
        #expect(bpe.encode(chunk: "hello") == [])
        #expect(bpe.vocabularySize == 0)
    }

    @Test("Vocab lookup round-trips")
    func vocabLookup() {
        let bpe = BPETokenizer(
            vocab: ["h": 1, "e": 2, "l": 3, "o": 4, "he": 10],
            merges: []
        )
        #expect(bpe.id(for: "h") == 1)
        #expect(bpe.token(for: 10) == "he")
        #expect(bpe.id(for: "missing") == nil)
    }

    @Test("BPE merges adjacent pairs in rank order")
    func bpeMergesByRank() {
        // vocab includes both single chars and merged pair "he"
        // merge rank 0: ("h", "e") - top priority
        let bpe = BPETokenizer(
            vocab: ["h": 1, "e": 2, "l": 3, "o": 4, "he": 10],
            merges: [("h", "e")]
        )
        // "hello" has bytes h, e, l, l, o; "h"+"e" should merge into "he".
        // Resulting tokens: ["he", "l", "l", "o"] -> [10, 3, 3, 4]
        let result = bpe.encode(chunk: "hello")
        #expect(result == [10, 3, 3, 4])
    }

    @Test("Decode reverses encode for ASCII-only single-char vocab")
    func decodeRoundTrip() {
        var vocab: [String: Int] = [:]
        for (i, byte) in (0x21...0x7E).enumerated() {
            let char = BPETokenizer.byteToUnicode[UInt8(byte)]!
            vocab[String(char)] = i
        }
        let bpe = BPETokenizer(vocab: vocab, merges: [])
        let original = "hello"
        let encoded = bpe.encode(chunk: original)
        let decoded = bpe.decode(tokens: encoded)
        #expect(decoded == original)
    }

    @Test("Decode skips unknown IDs")
    func decodeSkipsUnknown() {
        let bpe = BPETokenizer(vocab: ["h": 1], merges: [])
        let result = bpe.decode(tokens: [1, 999, 1])
        #expect(result == "hh")
    }

    @Test("Single-character input returns one token")
    func singleChar() {
        let h = BPETokenizer.byteToUnicode[UInt8(ascii: "h")]!
        let bpe = BPETokenizer(vocab: [String(h): 1], merges: [])
        #expect(bpe.encode(chunk: "h") == [1])
    }
}
