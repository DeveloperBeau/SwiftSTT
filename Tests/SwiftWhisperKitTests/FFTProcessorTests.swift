import Foundation
import Testing
@testable import SwiftWhisperKit

@Suite("FFTProcessor")
struct FFTProcessorTests {

    static func sine(frequency: Float, samples: Int, sampleRate: Float = 16_000) -> [Float] {
        (0..<samples).map { i in
            sinf(2 * .pi * frequency * Float(i) / sampleRate)
        }
    }

    @Test("Output length is 257 bins (FFT padded to 512)")
    func outputLength() {
        let fft = FFTProcessor()
        let frame = [Float](repeating: 0, count: 400)
        let power = fft.process(frame: frame)
        #expect(power.count == 257)
    }

    @Test("Zero-mean silence produces near-zero power")
    func silenceProducesZeroPower() {
        let fft = FFTProcessor()
        let frame = [Float](repeating: 0, count: 400)
        let power = fft.process(frame: frame)
        for v in power {
            #expect(v < 1e-6)
        }
    }

    @Test("440Hz sine peaks near bin 14")
    func sinePeakBin() {
        // 16000 / 512 = 31.25 Hz per bin. 440 / 31.25 = 14.08.
        let fft = FFTProcessor()
        let frame = Self.sine(frequency: 440, samples: 400)
        let power = fft.process(frame: frame)

        var maxBin = 0
        var maxPower: Float = 0
        for (k, p) in power.enumerated() where p > maxPower {
            maxPower = p
            maxBin = k
        }
        #expect(maxBin == 14 || maxBin == 13)
        #expect(maxPower > 0)
    }

    @Test("DC bin is small vs peak for zero-mean signal")
    func dcBinSmallForZeroMean() {
        let fft = FFTProcessor()
        let frame = Self.sine(frequency: 440, samples: 400)
        let power = fft.process(frame: frame)
        let peak = power.max()!
        #expect(power[0] < peak / 50)
    }

    @Test("1000Hz sine peaks near bin 32")
    func anotherFrequency() {
        // 1000 / 31.25 = 32.0
        let fft = FFTProcessor()
        let frame = Self.sine(frequency: 1000, samples: 400)
        let power = fft.process(frame: frame)
        var maxBin = 0
        var maxPower: Float = 0
        for (k, p) in power.enumerated() where p > maxPower {
            maxPower = p
            maxBin = k
        }
        #expect(abs(maxBin - 32) <= 1)
    }
}
