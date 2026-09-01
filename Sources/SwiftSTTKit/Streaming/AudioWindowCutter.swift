import Foundation
import SwiftSTTCore

/// Cuts a stream of captured PCM buffers into ``AudioWindow`` values, on
/// silence where it can and on a duration limit where it must.
public actor AudioWindowCutter {

    private let policy: StreamingWindowPolicy
    private let detector: any VoiceActivityDetector
    private let refiner: VADBoundaryRefiner

    private var pending: [Float] = []
    private var pendingContainsSpeech = false

    /// How much of ``pending`` was carried over from the previous window and
    /// so has already been transcribed once.
    private var carriedSampleCount = 0
    private var windowStartTime: TimeInterval = 0
    private var elapsedTime: TimeInterval = 0

    /// Creates a cutter.
    ///
    /// - Parameters:
    ///   - policy: window bounds.
    ///   - detector: voice activity detector. ``EnergyVAD`` by default;
    ///     ``SileroVAD`` if a converted model is available.
    ///   - refiner: boundary refiner. When `nil`, one is built at the policy's
    ///     sample rate so the two cannot disagree.
    public init(
        policy: StreamingWindowPolicy = .default,
        detector: any VoiceActivityDetector = EnergyVAD(),
        refiner: VADBoundaryRefiner? = nil
    ) {
        self.policy = policy
        self.detector = detector
        self.refiner = refiner ?? VADBoundaryRefiner(sampleRate: Double(policy.sampleRate))
    }

    /// Feeds one captured buffer in, returning a window when this buffer
    /// closed one.
    ///
    /// Returns `nil` when the window is still filling, and also when a closed
    /// window contained no speech at all — speechless audio is dropped rather
    /// than sent to the model, which hallucinates on silence.
    public func ingest(_ samples: [Float]) async -> AudioWindow? {
        guard !samples.isEmpty else { return nil }
        pending.append(contentsOf: samples)
        let isSpeech = await detector.isSpeech(
            chunk: AudioChunk(samples: samples, sampleRate: policy.sampleRate, timestamp: elapsedTime))
        if isSpeech { pendingContainsSpeech = true }
        let boundary = await refiner.ingest(isSpeech: isSpeech, sampleCount: samples.count)
        elapsedTime += Double(samples.count) / Double(policy.sampleRate)

        let pendingDuration = Double(pending.count) / Double(policy.sampleRate)
        // Carried audio does not count toward the minimum. It has been
        // transcribed once already, so a window made mostly of carry gives the
        // model almost nothing new while still paying a full decode, and the
        // little it does contain arrives with the window's edges through it:
        // the shortest windows on real speech are the ones that came back
        // mangled. The maximum still counts every sample, because that bounds
        // what is handed to the model rather than what is learned from it.
        let newAudioDuration =
            Double(pending.count - carriedSampleCount) / Double(policy.sampleRate)
        if boundary != nil, newAudioDuration >= policy.minimumWindowDuration {
            return closeWindow(cause: .silence)
        } else if pendingDuration >= policy.maximumWindowDuration {
            return closeWindow(cause: .maximumDuration)
        }
        return nil
    }

    /// Closes whatever is buffered as a ``WindowCutCause/stop`` window,
    /// ignoring ``StreamingWindowPolicy/minimumWindowDuration``.
    ///
    /// Returns `nil` if nothing is buffered or the buffer holds no speech.
    public func flush() async -> AudioWindow? {
        closeWindow(cause: .stop)
    }

    private func closeWindow(cause: WindowCutCause) -> AudioWindow? {
        // Every cut but the last carries audio forward, because the model
        // transcribes the first word of a window badly when nothing precedes
        // it and a silence boundary is not the safe place it sounds like: a
        // detector reading a low-energy moment mid-phrase cuts straight
        // through a word. Carry-over still only makes sense when the audio
        // being carried is confirmed speech, since a dropped (speechless) cut
        // has nothing worth re-covering and must consume everything it
        // accumulated so windowStartTime tracks consumed audio exactly (see
        // AudioWindowCutterTests.C6).
        let carried: [Float]
        if cause != .stop, pendingContainsSpeech {
            // StreamingWindowPolicy floors maximumWindowDuration and clamps
            // overlapDuration from it, so this should never go negative in
            // practice — guarded anyway, because `suffix(_:)` traps on a
            // negative length and this is the line that would take the
            // trap, not the policy that produced it.
            //
            // Half the closed window is a hard ceiling on top of that. A
            // silence cut can close a window shorter than overlapDuration,
            // and carrying all of it forward would leave windowStartTime
            // exactly where it was: the same audio would be re-cut and
            // re-decoded forever, never advancing (see C15's strictly
            // increasing startTime check, which is what caught this).
            let requested = max(Int(policy.overlapDuration * Double(policy.sampleRate)), 0)
            let carryCount = min(requested, pending.count / 2)
            carried = Array(pending.suffix(carryCount))
        } else {
            carried = []
        }

        let closedSamples = pending
        let closedStartTime = windowStartTime
        defer {
            windowStartTime += Double(pending.count - carried.count) / Double(policy.sampleRate)
            pending = carried
            // A forced cut severed speech mid-phrase, so its carry is content
            // the next window still owes the transcript: keep it marked so a
            // subsequent flush() with no further input can still emit it (see
            // C13). A silence cut's carry is only lead-in for whatever speech
            // comes next, and has already been transcribed in full, so it does
            // not by itself justify decoding another window.
            pendingContainsSpeech = !carried.isEmpty && cause == .maximumDuration
            carriedSampleCount = carried.count
        }

        // A stop window is the last one: audio it declines to emit is not
        // picked up later, it is lost. The detector is the reason to decline,
        // and the detector is exactly what cannot be trusted on a short
        // trailing fragment, which is where a recording's final words live. So
        // stop emits whenever it holds audio the previous window did not
        // already cover, provided a carry came with it. The carry is a slice
        // of confirmed speech, so such a window is never the silence-only
        // decode this guard exists to prevent; without one there is nothing
        // but the detector's word to go on, and silence stays dropped (C9).
        let holdsUncoveredAudio =
            cause == .stop && carriedSampleCount > 0 && closedSamples.count > carriedSampleCount
        guard pendingContainsSpeech || holdsUncoveredAudio else { return nil }
        return AudioWindow(
            samples: closedSamples,
            startTime: closedStartTime,
            sampleRate: policy.sampleRate,
            cut: cause
        )
    }
}
