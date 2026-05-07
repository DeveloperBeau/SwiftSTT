import Foundation
import Testing
@testable import SwiftWhisperKit
import SwiftWhisperCore

@Suite("MelSpectrogram")
struct MelSpectrogramTests {

    static func sine(frequency: Float, samples: Int, amplitude: Float = 0.5, sampleRate: Int = 16_000) -> AudioChunk {
        let buf = (0..<samples).map { i in
            amplitude * sinf(2 * .pi * frequency * Float(i) / Float(sampleRate))
        }
        return AudioChunk(samples: buf, sampleRate: sampleRate, timestamp: 0)
    }

    // MARK: - Filterbank shape

    @Test("Filterbank dimensions are nMels x nBins")
    func filterbankShape() {
        let fb = MelSpectrogram.makeMelFilterbank(nMels: 80, fftSize: 512, sampleRate: 16_000)
        #expect(fb.count == 80 * 257)
    }

    @Test("Filterbank values are non-negative")
    func filterbankNonNegative() {
        let fb = MelSpectrogram.makeMelFilterbank(nMels: 80, fftSize: 512, sampleRate: 16_000)
        for v in fb {
            #expect(v >= 0)
        }
    }

    @Test("Mel scale is monotonic")
    func melMonotonic() {
        var prev: Float = -.greatestFiniteMagnitude
        for hz in stride(from: Float(0), to: 8000, by: 100) {
            let mel = MelSpectrogram.hzToMel(hz)
            #expect(mel > prev)
            prev = mel
        }
    }

    @Test("Mel <-> Hz is invertible")
    func melHzInvertible() {
        for hz in [Float(100), 440, 1000, 4000, 8000] {
            let recovered = MelSpectrogram.melToHz(MelSpectrogram.hzToMel(hz))
            #expect(abs(recovered - hz) < 1e-2)
        }
    }

    // MARK: - Frame counting + carry-over

    @Test("1 second of 16kHz audio produces ~98 frames")
    func frameCount() async throws {
        let mel = MelSpectrogram()
        let result = try await mel.process(chunk: Self.sine(frequency: 440, samples: 16_000))
        // (16000 - 400) / 160 + 1 = 98
        #expect(result.nFrames == 98)
        #expect(result.nMels == 80)
        #expect(result.frames.count == 80 * 98)
    }

    @Test("Carry-over: two half-chunks equal one full chunk in frame count")
    func carryoverEquivalence() async throws {
        let melA = MelSpectrogram()
        let resultA = try await melA.process(chunk: Self.sine(frequency: 440, samples: 16_000))

        let melB = MelSpectrogram()
        let half1 = Self.sine(frequency: 440, samples: 8_000)
        let half2Samples = (8_000..<16_000).map { i in
            Float(0.5) * sinf(2 * .pi * 440 * Float(i) / Float(16_000))
        }
        let half2 = AudioChunk(samples: half2Samples, sampleRate: 16_000, timestamp: 0)
        let r1 = try await melB.process(chunk: half1)
        let r2 = try await melB.process(chunk: half2)
        #expect(r1.nFrames + r2.nFrames == resultA.nFrames)
    }

    @Test("Reset clears leftover")
    func resetClearsLeftover() async throws {
        let mel = MelSpectrogram()
        // Feed 500 samples: 1 frame consumed, 100 leftover (after 1 hop).
        // Actually: pos 0 -> frame 0..400, then pos 160 -> needs 160..560 but only 500 -> stop. leftover = 160..500 = 340.
        _ = try await mel.process(chunk: Self.sine(frequency: 440, samples: 500))
        await mel.reset()
        // After reset, a 400-sample chunk should produce exactly 1 frame.
        let result = try await mel.process(chunk: Self.sine(frequency: 440, samples: 400))
        #expect(result.nFrames == 1)
    }

    @Test("Empty chunk produces empty result")
    func emptyChunk() async throws {
        let mel = MelSpectrogram()
        let result = try await mel.process(chunk: AudioChunk(samples: [], sampleRate: 16_000, timestamp: 0))
        #expect(result.nFrames == 0)
        #expect(result.frames.isEmpty)
    }

    // MARK: - Normalization

    @Test("Output range is exactly 2.0 wide (Whisper normalization invariant)")
    func rangeWidth() async throws {
        let mel = MelSpectrogram()
        let result = try await mel.process(chunk: Self.sine(frequency: 440, samples: 16_000, amplitude: 0.9))
        let minVal = result.frames.min()!
        let maxVal = result.frames.max()!
        // Whisper: out = (max(log_mel, log_mel.max() - 8.0) + 4.0) / 4.0
        // Range = 8.0 / 4.0 = 2.0 exactly when at least one value hits the clamp floor.
        #expect(abs((maxVal - minVal) - 2.0) < 1e-3)
    }

    @Test("All output values are finite")
    func valuesFinite() async throws {
        let mel = MelSpectrogram()
        let result = try await mel.process(chunk: Self.sine(frequency: 440, samples: 16_000, amplitude: 0.9))
        for v in result.frames {
            #expect(v.isFinite)
        }
    }

    @Test("Peak mel band for 440Hz is in lower third of bands")
    func peakBandFor440Hz() async throws {
        let mel = MelSpectrogram(nMels: 80)
        let result = try await mel.process(chunk: Self.sine(frequency: 440, samples: 16_000, amplitude: 0.9))

        // For each mel band, sum across all time frames.
        var bandTotals = [Float](repeating: 0, count: 80)
        for m in 0..<80 {
            var sum: Float = 0
            for t in 0..<result.nFrames {
                sum += result.frames[m * result.nFrames + t]
            }
            bandTotals[m] = sum
        }
        let peakBand = bandTotals.indices.max(by: { bandTotals[$0] < bandTotals[$1] }) ?? 0
        // 440 Hz mel ≈ 547, max mel ≈ 2840, so peak band ≈ 80 * 547/2840 ≈ 15.
        #expect(peakBand >= 8)
        #expect(peakBand <= 25)
    }
}
