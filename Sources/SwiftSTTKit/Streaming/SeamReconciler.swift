import Foundation
import SwiftSTTCore

/// Decides which segments of a decoded window to emit, given that the previous
/// window's audio may already have covered the start of this one.
///
/// Overlapping windows decode the same audio twice. Emitting both copies
/// duplicates words; emitting neither loses them. This type picks one seam
/// time per overlap and uses it as the single dividing line: the earlier
/// window emits everything ending at or before the seam, the later window
/// everything ending after it.
public struct SeamReconciler: Sendable {

    private var seamTime: TimeInterval
    private let overlapDuration: TimeInterval

    /// Creates a reconciler.
    ///
    /// - Parameter overlapDuration: seconds the next window re-covers after a
    ///   ``WindowCutCause/maximumDuration`` cut. Must match the policy driving
    ///   the cutter.
    public init(overlapDuration: TimeInterval) {
        self.overlapDuration = overlapDuration
        self.seamTime = -.infinity
    }

    /// Returns the segments of `window` that have not already been emitted,
    /// converted from window-local to absolute time, and records the seam for
    /// the next call.
    ///
    /// - Parameters:
    ///   - segments: what the model returned for `window`, in window-local time.
    ///   - window: the window those segments were decoded from.
    /// - Returns: segments in absolute time, in the order given.
    public mutating func reconcile(
        _ segments: [TranscriptionSegment],
        from window: AudioWindow
    ) -> [TranscriptionSegment] {
        let absolute = segments.map {
            TranscriptionSegment(
                text: $0.text,
                start: $0.start + window.startTime,
                end: $0.end + window.startTime
            )
        }

        let kept = absolute.filter { $0.end > seamTime }

        guard window.cut == .maximumDuration else {
            seamTime = -.infinity
            return kept
        }

        let preferredSeam = window.startTime + window.duration - overlapDuration
        let candidates = kept.map(\.end).filter { $0 >= preferredSeam }
        let newSeam = candidates.min() ?? preferredSeam
        seamTime = newSeam
        return kept.filter { $0.end <= newSeam }
    }

    /// Forgets the current seam. Use when starting a new recording session.
    public mutating func reset() {
        seamTime = -.infinity
    }
}
