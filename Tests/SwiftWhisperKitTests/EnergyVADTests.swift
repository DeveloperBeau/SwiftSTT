import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

@Suite("EnergyVAD")
struct EnergyVADTests {

    // MARK: - Helpers

    static func silence(samples: Int = 1024) -> AudioChunk {
        AudioChunk(samples: Array(repeating: 0, count: samples), timestamp: 0)
    }

    static func tone(
        frequency: Float = 440, amplitude: Float = 0.5, samples: Int = 1024,
        sampleRate: Int = 16_000
    ) -> AudioChunk {
        var buf = [Float](repeating: 0, count: samples)
        for i in 0..<samples {
            buf[i] = amplitude * sinf(2 * .pi * frequency * Float(i) / Float(sampleRate))
        }
        return AudioChunk(samples: buf, sampleRate: sampleRate, timestamp: 0)
    }

    // MARK: - Tests

    @Test("Empty chunk preserves current speech state")
    func emptyChunkPreservesState() async {
        let vad = EnergyVAD(warmupFrames: 0)
        let empty = AudioChunk(samples: [], sampleRate: 16_000, timestamp: 0)
        let initial = await vad.isSpeech(chunk: empty)
        #expect(initial == false)
        // After a tone, state should be true; an empty chunk leaves it true.
        _ = await vad.isSpeech(chunk: Self.tone(amplitude: 0.5))
        let after = await vad.isSpeech(chunk: empty)
        #expect(after == true)
    }

    @Test("Pure silence is never speech")
    func silenceNeverSpeech() async {
        let vad = EnergyVAD(warmupFrames: 0)
        for _ in 0..<10 {
            let result = await vad.isSpeech(chunk: Self.silence())
            #expect(result == false)
        }
    }

    @Test("Loud tone is detected as speech")
    func toneIsSpeech() async {
        let vad = EnergyVAD(warmupFrames: 0)
        _ = await vad.isSpeech(chunk: Self.tone(amplitude: 0.5))
        let result = await vad.isSpeech(chunk: Self.tone(amplitude: 0.5))
        #expect(result == true)
    }

    @Test("Hysteresis keeps speech state through brief silence")
    func hysteresisHoldsThroughSilence() async {
        let vad = EnergyVAD(hysteresisFrames: 5, warmupFrames: 0)

        // Establish speech state.
        for _ in 0..<3 {
            _ = await vad.isSpeech(chunk: Self.tone(amplitude: 0.5))
        }

        // 4 consecutive silent chunks should NOT close the segment (hysteresis = 5).
        for _ in 0..<4 {
            let stillSpeech = await vad.isSpeech(chunk: Self.silence())
            #expect(stillSpeech == true)
        }

        // 5th silent chunk crosses threshold and ends segment.
        let closed = await vad.isSpeech(chunk: Self.silence())
        #expect(closed == false)
    }

    @Test("Reset clears state")
    func resetClearsState() async {
        let vad = EnergyVAD(warmupFrames: 0)
        _ = await vad.isSpeech(chunk: Self.tone(amplitude: 0.5))
        _ = await vad.isSpeech(chunk: Self.tone(amplitude: 0.5))
        await vad.reset()
        let afterReset = await vad.isSpeech(chunk: Self.silence())
        #expect(afterReset == false)
    }

    @Test("Quiet tone below threshold is not speech")
    func quietToneNotSpeech() async {
        let vad = EnergyVAD(thresholdDB: -20.0, warmupFrames: 0)
        // Amplitude 0.001 → RMS ≈ 0.0007 → ≈ -63 dB, well below -20 dB threshold.
        let result = await vad.isSpeech(chunk: Self.tone(amplitude: 0.001))
        #expect(result == false)
    }

    @Test("Warmup period reports silence and seeds noise floor")
    func warmupSeedsFloor() async {
        let vad = EnergyVAD(thresholdDB: -60.0, marginDB: 6.0, warmupFrames: 5)
        // Feed audio that would normally trigger speech.
        for _ in 0..<5 {
            let result = await vad.isSpeech(chunk: Self.tone(amplitude: 0.5))
            #expect(result == false)
        }
    }

    @Test("Adaptive threshold doesn't latch on continuous quiet noise")
    func adaptiveThresholdAdjusts() async {
        let vad = EnergyVAD(
            thresholdDB: -60.0, marginDB: 6.0, hysteresisFrames: 3, noiseWindow: 10,
            warmupFrames: 10)

        // Warmup seeds noise floor with quiet audio.
        for _ in 0..<15 {
            _ = await vad.isSpeech(chunk: Self.tone(amplitude: 0.01))
        }
        // After warmup + noise floor adapts, same quiet level should not be speech.
        let result = await vad.isSpeech(chunk: Self.tone(amplitude: 0.01))
        #expect(result == false)
    }

    // MARK: - DSP unit tests

    @Test("RMS of constant 0.5 signal")
    func rmsConstant() {
        let samples = [Float](repeating: 0.5, count: 100)
        let rms = EnergyVAD.rms(samples)
        #expect(abs(rms - 0.5) < 1e-5)
    }

    @Test("RMS of zero signal")
    func rmsZero() {
        let samples = [Float](repeating: 0, count: 100)
        let rms = EnergyVAD.rms(samples)
        #expect(rms == 0.0)
    }

    @Test("dB conversion handles zero safely")
    func dbZeroSafe() {
        let result = EnergyVAD.dB(0)
        #expect(result < -100)  // Should be clamped, not -infinity.
        #expect(result.isFinite)
    }
}
