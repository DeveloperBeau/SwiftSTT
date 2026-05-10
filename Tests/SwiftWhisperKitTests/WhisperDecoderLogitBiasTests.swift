import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

@Suite("WhisperDecoder logit bias")
struct WhisperDecoderLogitBiasTests {

    @Test("empty bias map is a no-op")
    func emptyBiasNoOp() {
        var logits: [Float] = [1, 2, 3]
        let original = logits
        WhisperDecoder.applyLogitBias(logits: &logits, bias: [:])
        #expect(logits == original)
    }

    @Test("positive bias raises the targeted logit")
    func positiveBiasRaises() {
        var logits: [Float] = [0, 0, 0]
        WhisperDecoder.applyLogitBias(logits: &logits, bias: [1: 5.0])
        #expect(logits == [0, 5, 0])
    }

    @Test("negative bias lowers the targeted logit")
    func negativeBiasLowers() {
        var logits: [Float] = [0, 0, 0]
        WhisperDecoder.applyLogitBias(logits: &logits, bias: [2: -3.0])
        #expect(logits == [0, 0, -3])
    }

    @Test("multiple biases stack independently")
    func multipleBiasesStack() {
        var logits: [Float] = [1, 1, 1, 1]
        WhisperDecoder.applyLogitBias(logits: &logits, bias: [0: 2, 2: -1])
        #expect(logits == [3, 1, 0, 1])
    }

    @Test("out-of-range token ids are ignored")
    func outOfRangeIgnored() {
        var logits: [Float] = [1, 2]
        WhisperDecoder.applyLogitBias(logits: &logits, bias: [-1: 5, 99: -5])
        #expect(logits == [1, 2])
    }

    @Test("bias additively combines with existing logit")
    func biasAdditive() {
        var logits: [Float] = [4, 5, 6]
        WhisperDecoder.applyLogitBias(logits: &logits, bias: [1: 10])
        #expect(logits[1] == 15)
    }

    @Test("bias of 0 is a no-op for that token")
    func biasZeroNoChange() {
        var logits: [Float] = [3, 4, 5]
        WhisperDecoder.applyLogitBias(logits: &logits, bias: [1: 0])
        #expect(logits == [3, 4, 5])
    }
}
