import Foundation

/// One word with its inferred time bounds inside a ``TranscriptionSegment``.
///
/// Word timings are useful for karaoke-style highlighting, click-to-seek UIs,
/// and read-along readers. They are not used by the decoder itself.
///
/// > Important: SwiftSTT currently produces word timings by *heuristic*
/// > splitting (proportional to character count). True word-level alignment
/// > requires dynamic time warping over the decoder's cross-attention weights,
/// > which the standard Apple-exported Core ML decoders do not surface.
public struct WordTiming: Sendable, Equatable {

    /// The word as it appears in the segment's text (whitespace stripped).
    public let word: String

    /// Inferred start time in seconds, in the same time base as the parent
    /// segment's ``TranscriptionSegment/start``.
    public let start: TimeInterval

    /// Inferred end time in seconds. `end >= start` always holds.
    public let end: TimeInterval

    /// Creates a new WordTiming with the supplied values.
    public init(word: String, start: TimeInterval, end: TimeInterval) {
        self.word = word
        self.start = start
        self.end = end
    }
}
