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
        if boundary != nil, pendingDuration >= policy.minimumWindowDuration {
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
        // Carry-over only makes sense when the audio being carried is
        // confirmed speech: a dropped (speechless) forced cut has nothing
        // worth re-covering, and must consume everything it accumulated so
        // windowStartTime tracks consumed audio exactly (see AudioWindowCutterTests.C6).
        let carried: [Float]
        if cause == .maximumDuration, pendingContainsSpeech {
            // StreamingWindowPolicy floors maximumWindowDuration and clamps
            // overlapDuration from it, so this should never go negative in
            // practice — guarded anyway, because `suffix(_:)` traps on a
            // negative length and this is the line that would take the
            // trap, not the policy that produced it.
            let carryCount = max(Int(policy.overlapDuration * Double(policy.sampleRate)), 0)
            carried = Array(pending.suffix(carryCount))
        } else {
            carried = []
        }

        let closedSamples = pending
        let closedStartTime = windowStartTime
        defer {
            windowStartTime += Double(pending.count - carried.count) / Double(policy.sampleRate)
            pending = carried
            // A non-empty carry is, by construction above, a slice of audio
            // already confirmed as speech; keep it marked so a subsequent
            // flush() with no further input can still emit it (see C13).
            pendingContainsSpeech = !carried.isEmpty
        }

        guard pendingContainsSpeech else { return nil }
        return AudioWindow(
            samples: closedSamples,
            startTime: closedStartTime,
            sampleRate: policy.sampleRate,
            cut: cause
        )
    }
}
