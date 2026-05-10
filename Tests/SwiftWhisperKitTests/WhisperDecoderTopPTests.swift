import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

@Suite("WhisperDecoder top-p sampling")
struct WhisperDecoderTopPTests {

    @Test("topPFilter at 1.0 returns input unchanged")
    func topPFilterDisabled() {
        let probs: [Float] = [0.5, 0.3, 0.2]
        let result = WhisperDecoder.topPFilter(probs: probs, threshold: 1.0)
        #expect(result == probs)
    }

    @Test("topPFilter keeps only the top-mass entries")
    func topPFilterKeepsTopMass() {
        let probs: [Float] = [0.5, 0.3, 0.15, 0.05]
        let result = WhisperDecoder.topPFilter(probs: probs, threshold: 0.7)
        // 0.5 + 0.3 = 0.8 >= 0.7 so the top two are kept; the rest is zeroed.
        #expect(result[2] == 0)
        #expect(result[3] == 0)
        #expect(result[0] > 0)
        #expect(result[1] > 0)
    }

    @Test("topPFilter renormalises kept probabilities to sum to 1")
    func topPFilterRenormalises() {
        let probs: [Float] = [0.5, 0.3, 0.15, 0.05]
        let result = WhisperDecoder.topPFilter(probs: probs, threshold: 0.7)
        let total = result.reduce(0, +)
        #expect(abs(total - 1.0) < 0.0001)
    }

    @Test("topPFilter with threshold below smallest prob still keeps one entry")
    func topPFilterAlwaysKeepsOne() {
        let probs: [Float] = [0.4, 0.3, 0.2, 0.1]
        let result = WhisperDecoder.topPFilter(probs: probs, threshold: 0.1)
        // Largest entry alone (0.4) is above 0.1 threshold.
        #expect(result[0] > 0)
        #expect(result[1] == 0)
    }

    @Test("topPFilter on empty input is empty")
    func topPFilterEmpty() {
        #expect(WhisperDecoder.topPFilter(probs: [], threshold: 0.5).isEmpty)
    }

    @Test("topPFilter sorts then accumulates regardless of input order")
    func topPFilterSortsBeforeAccumulating() {
        let probs: [Float] = [0.1, 0.4, 0.3, 0.2]
        let result = WhisperDecoder.topPFilter(probs: probs, threshold: 0.7)
        // After sorting descending: 0.4, 0.3, 0.2, 0.1. 0.4 + 0.3 = 0.7 hits threshold.
        // Indices 1 (0.4) and 2 (0.3) should be kept.
        #expect(result[0] == 0)
        #expect(result[3] == 0)
        #expect(result[1] > 0)
        #expect(result[2] > 0)
    }

    @Test("topPFilter ties break by lowest index")
    func topPFilterTieBreaks() {
        let probs: [Float] = [0.25, 0.25, 0.25, 0.25]
        let result = WhisperDecoder.topPFilter(probs: probs, threshold: 0.5)
        // 0.25 + 0.25 = 0.5 hits threshold; lowest two indices retained.
        #expect(result[0] > 0)
        #expect(result[1] > 0)
        #expect(result[2] == 0)
        #expect(result[3] == 0)
    }

    @Test("topPFilter with all-zero input falls back to uniform")
    func topPFilterAllZero() {
        let probs: [Float] = [0, 0, 0, 0]
        let result = WhisperDecoder.topPFilter(probs: probs, threshold: 0.5)
        // No mass to renormalise; helper returns uniform distribution.
        let total = result.reduce(0, +)
        #expect(abs(total - 1.0) < 0.0001)
    }

    @Test("validate rejects topP <= 0")
    func validateRejectsZeroTopP() throws {
        var options = DecodingOptions.default
        options.topP = 0
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

    @Test("validate rejects topP > 1")
    func validateRejectsTopPGreaterThanOne() throws {
        var options = DecodingOptions.default
        options.topP = 1.5
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

    @Test("validate accepts topP exactly equal to 1")
    func validateAcceptsTopPOne() throws {
        var options = DecodingOptions.default
        options.topP = 1.0
        try WhisperDecoder.validate(options: options)
    }
}
