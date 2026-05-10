import Foundation

/// A confirmed speech segment in audio time, produced by smoothing a binary
/// speech/silence stream from a ``VoiceActivityDetector``.
///
/// Boundaries are emitted by `VADBoundaryRefiner` once a falling edge passes
/// its hysteresis test, so values are stable: the start of a boundary is the
/// first sample that the refiner committed to a speech segment, and the end
/// is the last speech sample before the silence run that closed the segment.
public struct SpeechBoundary: Sendable, Equatable {

    /// Start time in seconds, measured from the beginning of the audio
    /// stream the refiner has been observing.
    public let startTime: TimeInterval

    /// End time in seconds. `endTime >= startTime` always holds.
    public let endTime: TimeInterval

    /// Creates a new SpeechBoundary with the supplied values.
    public init(startTime: TimeInterval, endTime: TimeInterval) {
        self.startTime = startTime
        self.endTime = endTime
    }
}
