import Accelerate
import Foundation
import SwiftWhisperCore

/// Energy-based voice activity detector.
///
/// The cheapest possible VAD: compute the RMS of each chunk, compare against an
/// adaptive threshold, and apply hysteresis so brief gaps inside a phrase don't
/// trigger end-of-speech. No model, no allocations on the hot path beyond the
/// rolling noise floor buffer.
///
/// ## How the threshold adapts
///
/// The threshold is the larger of:
///
/// - A fixed floor (``init(thresholdDB:marginDB:hysteresisFrames:noiseWindow:warmupFrames:)``'s
///   `thresholdDB`), and
/// - The rolling mean of the recent noise floor in dB plus a margin.
///
/// The noise floor only updates when the detector is in silence, so loud speech
/// does not pollute the baseline. To bootstrap the floor before any speech can
/// be detected, the first ``init(thresholdDB:marginDB:hysteresisFrames:noiseWindow:warmupFrames:)``'s
/// `warmupFrames` chunks are unconditionally treated as silence and feed the
/// floor. Without that warmup, a single loud first chunk would lock the
/// detector into speech and the floor would never seed.
///
/// ## When to use Silero instead
///
/// Energy VAD works well on clean speech with steady background noise. On
/// recordings with music, traffic, or non-stationary noise it false-positives.
/// The neural ``SileroVAD`` will be the better choice once it is implemented;
/// see the open-question list in the implementation plan.
///
/// ## Tuning
///
/// The defaults (`-40 dB` threshold, 6 dB margin, 5-frame hysteresis, 30-frame
/// noise window) match a quiet office over an internal mic. For headset use,
/// drop the threshold to about `-50 dB`. For noisy environments raise the
/// margin to 9 or 12 dB.
public actor EnergyVAD: VoiceActivityDetector {

    /// Default amount (dB) the current RMS must exceed the rolling floor by
    /// before it counts as speech. The fixed floor still applies as a minimum.
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

    /// Creates a detector.
    ///
    /// - Parameters:
    ///   - thresholdDB: lowest threshold the detector will ever use. Anything
    ///     quieter than this counts as silence regardless of the noise floor.
    ///   - marginDB: how far above the rolling noise floor the current RMS must
    ///     sit before it counts as speech.
    ///   - hysteresisFrames: how many consecutive silent chunks must arrive
    ///     before an in-progress speech segment is closed. Prevents flicker on
    ///     short pauses inside a phrase.
    ///   - noiseWindow: length of the rolling noise-floor buffer, in chunks.
    ///   - warmupFrames: how many chunks at the start to treat as silence
    ///     unconditionally so the noise floor can seed.
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

    /// Returns `true` if the chunk is currently part of a speech segment.
    /// Stateful: identical chunks fed in different orders may give different answers.
    public func isSpeech(chunk: AudioChunk) async -> Bool {
        guard !chunk.samples.isEmpty else { return inSpeech }

        let rms = Self.rms(chunk.samples)
        let rmsDB = Self.dB(rms)
        framesSeen += 1

        if framesSeen <= warmupFrames {
            appendNoiseFloor(rms)
            return false
        }

        let floorDB: Float =
            noiseFloorRMS.isEmpty
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

        if !inSpeech {
            appendNoiseFloor(rms)
        }

        return inSpeech
    }

    /// Resets the noise floor, hysteresis counter, speech state, and warmup
    /// counter. Use after a long silence or when switching audio sources.
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

    /// Linear amplitude to dBFS. Clamps at `1e-10` to avoid `-inf` on silence.
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
