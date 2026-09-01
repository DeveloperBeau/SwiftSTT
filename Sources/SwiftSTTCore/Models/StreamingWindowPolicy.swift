import Foundation

/// Tunable bounds on how streaming transcription cuts audio into windows.
///
/// Whisper's encoder always processes a 30 second window, so a short window
/// costs nearly as much as a long one. Every value here trades CPU for latency.
public struct StreamingWindowPolicy: Sendable, Equatable {

    /// Longest run of speech, in seconds, before a window is cut without a
    /// silence boundary. The fallback, not the primary cut.
    public let maximumWindowDuration: TimeInterval

    /// Shortest window, in seconds, a silence boundary is allowed to cut.
    /// A boundary arriving before this is ignored and audio keeps accumulating.
    public let minimumWindowDuration: TimeInterval

    /// Seconds of audio re-covered by the next window after a
    /// ``WindowCutCause/maximumDuration`` cut, so a word severed at the cut is
    /// decoded whole at least once.
    public let overlapDuration: TimeInterval

    /// Sample rate in Hz the durations above are measured against.
    public let sampleRate: Int

    /// Creates a policy, clamping values that cannot hold together.
    ///
    /// `minimumWindowDuration` is clamped to at most `maximumWindowDuration`,
    /// and `overlapDuration` to at most `maximumWindowDuration / 2`.
    public init(
        maximumWindowDuration: TimeInterval = 15,
        minimumWindowDuration: TimeInterval = 2,
        overlapDuration: TimeInterval = 1.5,
        sampleRate: Int = 16_000
    ) {
        self.maximumWindowDuration = maximumWindowDuration
        self.minimumWindowDuration = min(minimumWindowDuration, maximumWindowDuration)
        self.overlapDuration = min(overlapDuration, maximumWindowDuration / 2)
        self.sampleRate = sampleRate
    }

    /// 15 second maximum, 2 second minimum, 1.5 second overlap, 16 kHz.
    public static let `default` = StreamingWindowPolicy()
}
