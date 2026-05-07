import Accelerate
import Foundation
import SwiftWhisperCore

/// Energy-based Voice Activity Detection.
///
/// Computes RMS power of each AudioChunk, compares against an adaptive noise floor
/// (rolling mean of recent RMS values), and applies hysteresis to debounce transitions
/// at segment boundaries.
public actor EnergyVAD: VoiceActivityDetector {

    /// Margin (dB) above the rolling noise floor that an RMS value must exceed
    /// to count as speech. The fixed `thresholdDB` floor still applies as a minimum.
    public static let defaultMarginDB: Float = 6.0

    private let thresholdDB: Float
    private let marginDB: Float
    private let hysteresisFrames: Int
    private let noiseWindow: Int
    private let warmupFrames: Int

    private var noiseFloorRMS: [Float] = []
    private var consecutiveSilent: Int = 0
    private var inSpeech: Bool = false
    private var framesSeen: Int = 0

    public init(
        thresholdDB: Float = -40.0,
        marginDB: Float = EnergyVAD.defaultMarginDB,
        hysteresisFrames: Int = 5,
        noiseWindow: Int = 30,
        warmupFrames: Int = 10
    ) {
        self.thresholdDB = thresholdDB
        self.marginDB = marginDB
        self.hysteresisFrames = hysteresisFrames
        self.noiseWindow = noiseWindow
        self.warmupFrames = warmupFrames
    }

    public func isSpeech(chunk: AudioChunk) async -> Bool {
        guard !chunk.samples.isEmpty else { return inSpeech }

        let rms = Self.rms(chunk.samples)
        let rmsDB = Self.dB(rms)
        framesSeen += 1

        // During warmup, always update noise floor and report silence.
        // This prevents permanent latching on continuous near-threshold audio.
        if framesSeen <= warmupFrames {
            appendNoiseFloor(rms)
            return false
        }

        let floorDB: Float = noiseFloorRMS.isEmpty
            ? thresholdDB
            : max(thresholdDB, Self.dB(Self.mean(noiseFloorRMS)) + marginDB)

        let aboveThreshold = rmsDB > floorDB

        if aboveThreshold {
            inSpeech = true
            consecutiveSilent = 0
        } else if inSpeech {
            consecutiveSilent += 1
            if consecutiveSilent >= hysteresisFrames {
                inSpeech = false
                consecutiveSilent = 0
            }
        }

        // Only update noise floor when not in speech to avoid contaminating with speech audio.
        if !inSpeech {
            appendNoiseFloor(rms)
        }

        return inSpeech
    }

    public func reset() async {
        noiseFloorRMS.removeAll(keepingCapacity: true)
        consecutiveSilent = 0
        inSpeech = false
        framesSeen = 0
    }

    private func appendNoiseFloor(_ rms: Float) {
        noiseFloorRMS.append(rms)
        if noiseFloorRMS.count > noiseWindow {
            noiseFloorRMS.removeFirst(noiseFloorRMS.count - noiseWindow)
        }
    }

    // MARK: - DSP helpers

    static func rms(_ samples: [Float]) -> Float {
        var result: Float = 0
        samples.withUnsafeBufferPointer { ptr in
            vDSP_rmsqv(ptr.baseAddress!, 1, &result, vDSP_Length(ptr.count))
        }
        return result
    }

    static func dB(_ value: Float) -> Float {
        20.0 * log10(max(value, 1e-10))
    }

    static func mean(_ values: [Float]) -> Float {
        var result: Float = 0
        values.withUnsafeBufferPointer { ptr in
            vDSP_meanv(ptr.baseAddress!, 1, &result, vDSP_Length(ptr.count))
        }
        return result
    }
}
