import Foundation

/// A contiguous chunk of transcribed text with its time bounds.
///
/// Segments correspond to the natural boundaries Whisper finds in the audio,
/// which usually line up with sentence pauses or speaker breaths. UI code can
/// render each segment on its own line and seek the audio player using
/// ``start`` and ``end``.
public struct TranscriptionSegment: Sendable, Equatable {

    /// The transcribed text for this segment.
    public let text: String

    /// Start time in seconds, measured from the beginning of the audio stream.
    public let start: TimeInterval

    /// End time in seconds. `end > start` always holds for non-empty segments.
    public let end: TimeInterval

    public init(text: String, start: TimeInterval, end: TimeInterval) {
        self.text = text
        self.start = start
        self.end = end
    }
}
