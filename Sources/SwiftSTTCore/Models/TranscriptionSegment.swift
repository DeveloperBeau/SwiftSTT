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

    /// Creates a new TranscriptionSegment with the supplied values.
    public init(text: String, start: TimeInterval, end: TimeInterval) {
        self.text = text
        self.start = start
        self.end = end
    }
}

/// Convenience helpers for breaking a segment down into per-word slices.
extension TranscriptionSegment {

    /// Heuristic word-level timings derived by splitting ``text`` on whitespace
    /// and distributing the segment's duration across the resulting words in
    /// proportion to their character count.
    ///
    /// > Important: This is *not* DTW-derived alignment. Real per-word timings
    /// > require cross-attention analysis on the Whisper decoder, which Apple's
    /// > Core ML exports do not expose. The result is good enough for
    /// > karaoke-style UI on conversational speech and falls apart on filler
    /// > words, long pauses, or very short utterances.
    ///
    /// Returns an empty array when ``text`` is empty or whitespace-only.
    /// The last word's `end` is pinned to the segment's `end` so the cumulative
    /// sum lines up exactly without float drift.
    public func proportionalWordTimings() -> [WordTiming] {
        let words = text.split(whereSeparator: \.isWhitespace).map(String.init)
        guard !words.isEmpty else { return [] }

        let duration = end - start
        if words.count == 1 {
            return [WordTiming(word: words[0], start: start, end: end)]
        }

        let totalChars = words.reduce(0) { $0 + $1.count }
        guard totalChars > 0 else { return [] }

        var result: [WordTiming] = []
        result.reserveCapacity(words.count)
        var cursor: TimeInterval = start
        for (index, word) in words.enumerated() {
            let isLast = index == words.count - 1
            let wordEnd: TimeInterval
            if isLast {
                wordEnd = end
            } else {
                wordEnd = cursor + (Double(word.count) / Double(totalChars)) * duration
            }
            result.append(WordTiming(word: word, start: cursor, end: wordEnd))
            cursor = wordEnd
        }
        return result
    }
}
