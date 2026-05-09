import Foundation
import Testing
@testable import SwiftWhisperKit
import SwiftWhisperCore

private func makeTimestampSpecials() -> [String: Int] {
    var specials: [String: Int] = [
        "<|endoftext|>": 50_257,
        "<|startoftranscript|>": 50_258,
        "<|en|>": 50_259,
        "<|translate|>": 50_358,
        "<|transcribe|>": 50_359,
        "<|notimestamps|>": 50_363,
    ]
    let firstTimestampId = 50_364
    let count = 1_501
    for step in 0..<count {
        let seconds = Double(step) * 0.02
        let formatted = String(format: "<|%.2f|>", seconds)
        specials[formatted] = firstTimestampId + step
    }
    return specials
}

private func tokenizerWithTimestamps() -> WhisperTokenizer {
    WhisperTokenizer(specialTokens: makeTimestampSpecials())
}

@Suite("WhisperDecoder timestamp parsing")
struct WhisperDecoderTimestampTests {

    @Test("Empty input returns empty array")
    func parseSegmentsEmpty() {
        let tokenizer = tokenizerWithTimestamps()
        let result = WhisperDecoder.parseSegments(tokens: [], tokenizer: tokenizer, windowOffsetSeconds: 0)
        #expect(result.isEmpty)
    }

    @Test("Single timestamp pair builds one segment")
    func parseSegmentsSinglePair() {
        let tokenizer = tokenizerWithTimestamps()
        let startTimestamp = 50_364
        let endTimestamp = 50_414
        let tokens: [WhisperToken] = [
            WhisperToken(id: startTimestamp, text: ""),
            WhisperToken(id: 5_000, text: " hello"),
            WhisperToken(id: 5_001, text: " world"),
            WhisperToken(id: endTimestamp, text: ""),
        ]
        let result = WhisperDecoder.parseSegments(tokens: tokens, tokenizer: tokenizer, windowOffsetSeconds: 0)
        #expect(result.count == 1)
        #expect(result[0].start == 0)
        #expect(abs(result[0].end - 1.0) < 0.0001)
    }

    @Test("Multiple pairs build multiple segments")
    func parseSegmentsMultiplePairs() {
        let tokenizer = tokenizerWithTimestamps()
        let tokens: [WhisperToken] = [
            WhisperToken(id: 50_364, text: ""),
            WhisperToken(id: 5_000, text: " a"),
            WhisperToken(id: 50_414, text: ""),
            WhisperToken(id: 50_414, text: ""),
            WhisperToken(id: 5_001, text: " b"),
            WhisperToken(id: 50_464, text: ""),
        ]
        let result = WhisperDecoder.parseSegments(tokens: tokens, tokenizer: tokenizer, windowOffsetSeconds: 0)
        #expect(result.count == 2)
        #expect(abs(result[0].start - 0.0) < 0.0001)
        #expect(abs(result[0].end - 1.0) < 0.0001)
        #expect(abs(result[1].start - 1.0) < 0.0001)
        #expect(abs(result[1].end - 2.0) < 0.0001)
    }

    @Test("Unmatched start timestamp is dropped")
    func parseSegmentsUnmatchedStart() {
        let tokenizer = tokenizerWithTimestamps()
        let tokens: [WhisperToken] = [
            WhisperToken(id: 50_364, text: ""),
            WhisperToken(id: 5_000, text: " hello"),
            WhisperToken(id: 50_414, text: ""),
            WhisperToken(id: 50_500, text: ""),
            WhisperToken(id: 5_001, text: " trailing"),
        ]
        let result = WhisperDecoder.parseSegments(tokens: tokens, tokenizer: tokenizer, windowOffsetSeconds: 0)
        #expect(result.count == 1)
        #expect(result[0].start == 0)
    }

    @Test("windowOffsetSeconds adds correctly to start and end")
    func parseSegmentsWindowOffset() {
        let tokenizer = tokenizerWithTimestamps()
        let tokens: [WhisperToken] = [
            WhisperToken(id: 50_364, text: ""),
            WhisperToken(id: 5_000, text: " text"),
            WhisperToken(id: 50_414, text: ""),
        ]
        let result = WhisperDecoder.parseSegments(tokens: tokens, tokenizer: tokenizer, windowOffsetSeconds: 30)
        #expect(result.count == 1)
        #expect(abs(result[0].start - 30.0) < 0.0001)
        #expect(abs(result[0].end - 31.0) < 0.0001)
    }

    @Test("Decoded text strips timestamp tokens")
    func parseSegmentsTextStripsTimestamps() {
        let tokenizer = tokenizerWithTimestamps()
        // Body intentionally contains a timestamp-like id, but parseSegments
        // stops on the first end timestamp before it reaches body content.
        let tokens: [WhisperToken] = [
            WhisperToken(id: 50_364, text: ""),
            WhisperToken(id: 5_000, text: " body"),
            WhisperToken(id: 50_414, text: ""),
        ]
        let result = WhisperDecoder.parseSegments(tokens: tokens, tokenizer: tokenizer, windowOffsetSeconds: 0)
        #expect(result.count == 1)
        #expect(!result[0].text.contains("<|"))
    }

    @Test("Tokens before first timestamp are ignored")
    func parseSegmentsTokensBeforeFirstTimestamp() {
        let tokenizer = tokenizerWithTimestamps()
        let tokens: [WhisperToken] = [
            WhisperToken(id: 5_000, text: " stray"),
            WhisperToken(id: 50_364, text: ""),
            WhisperToken(id: 5_001, text: " kept"),
            WhisperToken(id: 50_414, text: ""),
        ]
        let result = WhisperDecoder.parseSegments(tokens: tokens, tokenizer: tokenizer, windowOffsetSeconds: 0)
        #expect(result.count == 1)
    }

    @Test("Non-timestamp specials inside a segment are stripped by tokenizer.decode")
    func parseSegmentsNonTimestampSpecials() {
        let tokenizer = tokenizerWithTimestamps()
        let tokens: [WhisperToken] = [
            WhisperToken(id: 50_364, text: ""),
            WhisperToken(id: tokenizer.startOfTranscriptToken, text: ""),
            WhisperToken(id: 5_000, text: " text"),
            WhisperToken(id: 50_414, text: ""),
        ]
        let result = WhisperDecoder.parseSegments(tokens: tokens, tokenizer: tokenizer, windowOffsetSeconds: 0)
        #expect(result.count == 1)
        #expect(!result[0].text.contains("<|startoftranscript|>"))
    }

    @Test("initialPromptTokens drops noTimestamps when withoutTimestamps is false")
    func initialPromptTokensDropsNoTimestamps() {
        let tokenizer = tokenizerWithTimestamps()
        var options = DecodingOptions.default
        options.language = "en"
        options.task = .transcribe
        options.withoutTimestamps = false
        let prompt = WhisperDecoder.initialPromptTokens(options: options, tokenizer: tokenizer)
        #expect(!prompt.contains(tokenizer.noTimestampsToken))
    }

    @Test("initialPromptTokens still includes noTimestamps when default")
    func initialPromptTokensDefaultIncludesNoTimestamps() {
        let tokenizer = tokenizerWithTimestamps()
        let prompt = WhisperDecoder.initialPromptTokens(options: .default, tokenizer: tokenizer)
        #expect(prompt.contains(tokenizer.noTimestampsToken))
    }

    @Test("timestampSeconds maps token id to time")
    func timestampSecondsMapping() {
        #expect(WhisperDecoder.timestampSeconds(forTokenId: 50_364) == 0.0)
        #expect(abs(WhisperDecoder.timestampSeconds(forTokenId: 50_414) - 1.0) < 0.0001)
        #expect(abs(WhisperDecoder.timestampSeconds(forTokenId: 51_864) - 30.0) < 0.0001)
    }

    @Test("timestampSeconds clamps below first timestamp id")
    func timestampSecondsClampsBelow() {
        #expect(WhisperDecoder.timestampSeconds(forTokenId: 50_000) == 0.0)
    }
}
