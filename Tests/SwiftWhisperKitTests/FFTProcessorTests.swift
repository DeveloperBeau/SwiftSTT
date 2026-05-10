import Foundation
import SwiftWhisperCore
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
    func outputLength() throws {
        let fft = try FFTProcessor()
        let frame = [Float](repeating: 0, count: 400)
        let power = try fft.process(frame: frame)
        #expect(power.count == 257)
    }

    @Test("Zero-mean silence produces near-zero power")
    func silenceProducesZeroPower() throws {
        let fft = try FFTProcessor()
        let frame = [Float](repeating: 0, count: 400)
        let power = try fft.process(frame: frame)
        for v in power {
            #expect(v < 1e-6)
        }
    }

    @Test("440Hz sine peaks near bin 14")
    func sinePeakBin() throws {
        // 16000 / 512 = 31.25 Hz per bin. 440 / 31.25 = 14.08.
        let fft = try FFTProcessor()
        let frame = Self.sine(frequency: 440, samples: 400)
        let power = try fft.process(frame: frame)

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
    func dcBinSmallForZeroMean() throws {
        let fft = try FFTProcessor()
        let frame = Self.sine(frequency: 440, samples: 400)
        let power = try fft.process(frame: frame)
        let peak = power.max()!
        #expect(power[0] < peak / 50)
    }

    @Test("1000Hz sine peaks near bin 32")
    func anotherFrequency() throws {
        // 1000 / 31.25 = 32.0
        let fft = try FFTProcessor()
        let frame = Self.sine(frequency: 1000, samples: 400)
        let power = try fft.process(frame: frame)
        var maxBin = 0
        var maxPower: Float = 0
        for (k, p) in power.enumerated() where p > maxPower {
            maxPower = p
            maxBin = k
        }
        #expect(abs(maxBin - 32) <= 1)
    }

    @Test("Wrong frame length throws fftFrameSizeMismatch")
    func wrongFrameLength() throws {
        let fft = try FFTProcessor()
        do {
            _ = try fft.process(frame: [Float](repeating: 0, count: 200))
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            #expect(error == .fftFrameSizeMismatch(got: 200, expected: 400))
        }
    }

    @Test("Invalid DFT count throws fftSetupFailed")
    func invalidDFTCount() {
        // 7 is not of the form f * 2^n with f in {1, 3, 5, 15}, so vDSP rejects it.
        do {
            _ = try FFTProcessor(fftSize: 7)
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            if case .fftSetupFailed = error {
                // Pass.
            } else {
                Issue.record("wrong error: \(error)")
            }
        } catch {
            Issue.record("unexpected non-SwiftWhisperError: \(error)")
        }
    }
}
