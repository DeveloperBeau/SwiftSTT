import Foundation
import SwiftWhisperCore

/// Smooths a binary speech/silence stream from a ``VoiceActivityDetector``
/// into ``SpeechBoundary`` values.
///
/// Raw VAD output flickers around the start and end of every utterance: a
/// short noise spike inside a long silence will trigger a single
/// `isSpeech == true` chunk, and a single quiet syllable inside speech will
/// trigger a single `isSpeech == false` chunk. The refiner suppresses both
/// kinds of flicker by requiring N consecutive frames of the new state before
/// committing to a transition.
///
/// ## Usage
///
/// ```swift
/// let refiner = VADBoundaryRefiner()
/// for chunk in chunks {
///     let speech = await vad.isSpeech(chunk: chunk)
///     if let boundary = await refiner.ingest(isSpeech: speech, sampleCount: chunk.samples.count) {
///         print("Segment: \(boundary.startTime)..\(boundary.endTime)")
///     }
/// }
/// if let final = await refiner.flush() {
///     print("Final segment: \(final.startTime)..\(final.endTime)")
/// }
/// ```
///
/// ## Time accounting
///
/// The refiner converts incoming sample counts to seconds using the
/// configured `sampleRate`. The boundary's `startTime` is stamped at the
/// instant the first frame of the speech run was ingested (i.e. before
/// hysteresis confirmed the transition). The `endTime` is similarly stamped
/// at the instant of the first silence frame that began the closing run.
public actor VADBoundaryRefiner {

    private let startConsecutive: Int
    private let endConsecutive: Int
    private let sampleRate: Double

    private var elapsed: TimeInterval = 0
    private var inSpeech: Bool = false
    private var consecutiveSpeech: Int = 0
    private var consecutiveSilence: Int = 0

    /// Time at which the current candidate speech run began. Stamped on the
    /// first ``ingest(isSpeech:sampleCount:)`` call that flipped from
    /// silence-with-zero-streak to silence-with-one-speech-frame, so that
    /// when the hysteresis confirms the transition the start is at the
    /// run's beginning rather than its threshold-crossing point.
    private var pendingStartTime: TimeInterval?

    /// Time at which the current candidate silence run began (during a
    /// confirmed speech segment). Stamped similarly so the end of the
    /// boundary lines up with the first silent frame of the closing run.
    private var pendingEndTime: TimeInterval?

    /// Locked-in start time for the active speech segment, set when the
    /// rising-edge hysteresis confirms a transition.
    private var currentSegmentStart: TimeInterval?

    /// Creates a refiner.
    ///
    /// - Parameters:
    ///   - startConsecutive: number of consecutive speech frames required
    ///     before a silence-to-speech transition is committed.
    ///   - endConsecutive: number of consecutive silence frames required
    ///     before a speech-to-silence transition is committed and a boundary
    ///     emitted.
    ///   - sampleRate: audio sample rate in Hz used to convert sample counts
    ///     to elapsed time.
    public init(
        startConsecutive: Int = 3,
        endConsecutive: Int = 5,
        sampleRate: Double = 16_000
    ) {
        self.startConsecutive = max(1, startConsecutive)
        self.endConsecutive = max(1, endConsecutive)
        self.sampleRate = sampleRate
    }

    /// Feeds one VAD verdict into the refiner. Returns a ``SpeechBoundary``
    /// when a falling-edge hysteresis test passes (i.e. a speech segment has
    /// just closed). Otherwise returns `nil`.
    public func ingest(isSpeech: Bool, sampleCount: Int) async -> SpeechBoundary? {
        // Stamp times before advancing elapsed, so a frame's "start time" is
        // the start of that frame in audio time.
        let frameStart = elapsed
        let duration = Double(sampleCount) / sampleRate
        elapsed += duration

        var emitted: SpeechBoundary?

        if inSpeech {
            if isSpeech {
                consecutiveSilence = 0
                pendingEndTime = nil
            } else {
                if pendingEndTime == nil {
                    pendingEndTime = frameStart
                }
                consecutiveSilence += 1
                if consecutiveSilence >= endConsecutive,
                    let segStart = currentSegmentStart,
                    let segEnd = pendingEndTime
                {
                    emitted = SpeechBoundary(startTime: segStart, endTime: segEnd)
                    inSpeech = false
                    consecutiveSpeech = 0
                    consecutiveSilence = 0
                    pendingStartTime = nil
                    pendingEndTime = nil
                    currentSegmentStart = nil
                }
            }
        } else {
            if isSpeech {
                if pendingStartTime == nil {
                    pendingStartTime = frameStart
                }
                consecutiveSpeech += 1
                if consecutiveSpeech >= startConsecutive,
                    let candidateStart = pendingStartTime
                {
                    inSpeech = true
                    currentSegmentStart = candidateStart
                    consecutiveSpeech = 0
                    consecutiveSilence = 0
                    pendingStartTime = nil
                }
            } else {
                consecutiveSpeech = 0
                pendingStartTime = nil
            }
        }

        return emitted
    }

    /// Returns an in-progress boundary if currently in speech. Closes the
    /// segment at the current elapsed time. Returns `nil` if currently in
    /// silence. The refiner remains in its current state. Call ``reset()``
    /// if a fresh observation period is needed.
    public func flush() async -> SpeechBoundary? {
        guard inSpeech, let segStart = currentSegmentStart else { return nil }
        return SpeechBoundary(startTime: segStart, endTime: elapsed)
    }

    /// Clears all state and resets elapsed time to zero.
    public func reset() async {
        elapsed = 0
        inSpeech = false
        consecutiveSpeech = 0
        consecutiveSilence = 0
        pendingStartTime = nil
        pendingEndTime = nil
        currentSegmentStart = nil
    }
}
