import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

@Suite("WhisperDecoder repetition penalty")
struct WhisperDecoderRepetitionPenaltyTests {

    @Test("penalty 1.0 is a no-op")
    func penaltyOneNoOp() {
        var logits: [Float] = [1, 2, 3, 4]
        let original = logits
        WhisperDecoder.applyRepetitionPenalty(
            logits: &logits,
            previousTokens: [0, 2],
            penalty: 1.0
        )
        #expect(logits == original)
    }

    @Test("positive logits are divided by the penalty")
    func positiveLogitsDivided() {
        var logits: [Float] = [4, 6, 8]
        WhisperDecoder.applyRepetitionPenalty(
            logits: &logits,
            previousTokens: [0],
            penalty: 2.0
        )
        #expect(logits[0] == 2)
        #expect(logits[1] == 6)
        #expect(logits[2] == 8)
    }

    @Test("negative logits are multiplied by the penalty")
    func negativeLogitsMultiplied() {
        var logits: [Float] = [-3, -2, -1]
        WhisperDecoder.applyRepetitionPenalty(
            logits: &logits,
            previousTokens: [0, 1],
            penalty: 2.0
        )
        #expect(logits[0] == -6)
        #expect(logits[1] == -4)
        #expect(logits[2] == -1)
    }

    @Test("empty previousTokens leaves logits untouched")
    func emptyPreviousTokens() {
        var logits: [Float] = [1, 2, 3]
        let original = logits
        WhisperDecoder.applyRepetitionPenalty(
            logits: &logits,
            previousTokens: [],
            penalty: 5.0
        )
        #expect(logits == original)
    }

    @Test("out-of-range token ids are silently ignored")
    func outOfRangeIgnored() {
        var logits: [Float] = [1, 2]
        WhisperDecoder.applyRepetitionPenalty(
            logits: &logits,
            previousTokens: [-1, 99],
            penalty: 2.0
        )
        #expect(logits == [1, 2])
    }

    @Test("repeated penalty hit collapses logit further each time")
    func repeatedHitsCompound() {
        var logits: [Float] = [8]
        WhisperDecoder.applyRepetitionPenalty(
            logits: &logits,
            previousTokens: [0, 0, 0],
            penalty: 2.0
        )
        // 8 / 2 / 2 / 2 == 1.
        #expect(logits[0] == 1)
    }

    @Test("validate rejects zero penalty")
    func validateRejectsZeroPenalty() throws {
        var options = DecodingOptions.default
        options.repetitionPenalty = 0
        do {
            try WhisperDecoder.validate(options: options)
            Issue.record("expected throw")
        } catch {
            if case .invalidDecodingOption = error {
            } else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    @Test("validate rejects negative penalty")
    func validateRejectsNegativePenalty() throws {
        var options = DecodingOptions.default
        options.repetitionPenalty = -1
        do {
            try WhisperDecoder.validate(options: options)
            Issue.record("expected throw")
        } catch {
            if case .invalidDecodingOption = error {
            } else {
                Issue.record("wrong error: \(error)")
            }
        }
    }
}
