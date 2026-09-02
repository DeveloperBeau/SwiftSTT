import Foundation
import SwiftSTTCore

/// Decides what a decoded window adds to the transcript, and what it takes
/// back.
///
/// Every window but the last carries its final
/// ``StreamingWindowPolicy/overlapDuration`` of audio forward, so consecutive
/// windows decode the same audio twice. The overlap is deliberate: a word the
/// model meets at the very start of a window, with no audio in front of it, is
/// the word it most often gets wrong. Decoding it twice is what buys the second
/// reading its context.
///
/// The two readings are compared as text, not as timestamps. A model decoding
/// a short window reports times that overrun the audio it was handed, so a
/// dividing line drawn in seconds falls in the wrong place. Nor do the readings
/// match word for word: the same audio comes back as `"runs low"` and as
/// `"runs long"`. So they are aligned on their longest common subsequence, and
/// the alignment decides two separate things.
///
/// Words the two readings agree on are corroborated, and the first window's
/// copy of them stands. The second window's copy is dropped as a duplicate.
///
/// Words the *first* window emitted after the last agreed word are not
/// corroborated by anything. They are what it made of audio that ran off the
/// end of its window, and the next window read that same audio with the rest of
/// the sentence behind it. Those words are retracted and the second reading
/// takes their place. Neither reading is better as a rule, which is why the
/// choice is not made by rule: only the tail nothing confirms is replaced, and
/// a window whose tail the next one agrees with keeps every word of it.
public struct SeamReconciler: Sendable {

    /// Where one word sits in ``emitted``.
    private struct WordPosition {
        let segment: Int
        let word: Int
        let compared: String
    }

    /// Segments handed out so far, trimmed to those the next overlap could
    /// still reach.
    private var emitted: [TranscriptionSegment] = []

    /// Where the previous window's audio ended.
    ///
    /// A window starting before that
    /// re-covers the difference, and the difference is the only place either a
    /// duplicate or an unconfirmed word can come from.
    private var previousWindowEndTime: TimeInterval?

    private let maximumOverlapWordCount: Int

    /// About double the fastest human speech, so a duration converts to a
    /// generous upper bound on the words it can hold.
    private static let fastestWordsPerSecond: Double = 12

    /// Creates a reconciler.
    ///
    /// - Parameter overlapDuration: seconds each window re-covers of the one
    ///   before it. Must match the policy driving the cutter.
    public init(overlapDuration: TimeInterval) {
        self.maximumOverlapWordCount = max(4, Int(overlapDuration * Self.fastestWordsPerSecond))
    }

    /// Returns what `window` adds to the transcript and what it withdraws,
    /// converted from window-local to absolute time.
    ///
    /// - Parameters:
    ///   - segments: what the model returned for `window`, in window-local time.
    ///   - window: the window those segments were decoded from.
    /// - Returns: an update appending this window's new segments, and
    ///   withdrawing the previous window's trailing words when this one
    ///   disagrees with them.
    public mutating func reconcile(
        _ segments: [TranscriptionSegment],
        from window: AudioWindow
    ) -> TranscriptUpdate {
        let absolute = segments.map {
            TranscriptionSegment(
                text: $0.text,
                start: $0.start + window.startTime,
                end: $0.end + window.startTime
            )
        }
        let overlapSeconds = previousWindowEndTime.map { max($0 - window.startTime, 0) } ?? 0
        previousWindowEndTime = window.startTime + window.duration

        guard overlapSeconds > 0 else { return appending(absolute) }
        return reconciling(absolute, overlapping: overlapSeconds)
    }

    private mutating func reconciling(
        _ absolute: [TranscriptionSegment],
        overlapping overlapSeconds: TimeInterval
    ) -> TranscriptUpdate {
        let span = min(
            max(Int(overlapSeconds * Self.fastestWordsPerSecond), 1), maximumOverlapWordCount)
        let positions = Self.wordPositions(in: emitted)
        let earlierStart = max(positions.count - span, 0)
        let earlier = positions[earlierStart...].map(\.compared)
        let later = Array(absolute.flatMap { Self.comparableWords(of: $0.text) }.prefix(span))
        guard !earlier.isEmpty, !later.isEmpty else { return appending(absolute) }

        let agreed = Self.longestCommonSubsequence(Array(earlier), later)
        guard let last = agreed.last else { return appending(absolute) }
        // One agreed word is only trusted when it is the very seam: the last
        // word already emitted reappearing as the first word decoded. Further
        // in, a lone match is as likely to be a common word landing twice by
        // chance, and acting on it would drop or retract real speech.
        guard agreed.count >= 2 || last.later == 0 else { return appending(absolute) }

        let fresh = Self.dropping(last.later + 1, wordsFrom: absolute)
        // Nothing new arrived, so there is no reading to put in the place of
        // anything: withdrawing the tail here would simply lose it.
        guard !fresh.isEmpty else { return appending([]) }

        let firstUnconfirmed = earlierStart + last.earlier + 1
        guard firstUnconfirmed < positions.count else { return appending(fresh) }

        let position = positions[firstUnconfirmed]
        let retracted = emitted[position.segment]
        let confirmedPart =
            position.word > 0 ? [retracted.keepingLeadingWords(position.word)] : []
        emitted.removeSubrange(position.segment...)
        return appending(confirmedPart + fresh, retractingFrom: retracted.start)
    }

    /// Records `segments` as emitted and returns them as an update.
    private mutating func appending(
        _ segments: [TranscriptionSegment],
        retractingFrom: TimeInterval? = nil
    ) -> TranscriptUpdate {
        emitted.append(contentsOf: segments)
        // Only the span the next overlap can reach is ever consulted. Keeping
        // more would grow without bound over a long recording.
        let keep = maximumOverlapWordCount * 2
        while Self.wordPositions(in: emitted).count > keep, emitted.count > 1 {
            emitted.removeFirst()
        }
        return TranscriptUpdate(retractingFrom: retractingFrom, segments: segments)
    }

    private static func wordPositions(in segments: [TranscriptionSegment]) -> [WordPosition] {
        segments.enumerated().flatMap { index, segment in
            tokens(of: segment.text).enumerated().compactMap { wordIndex, token in
                token.compared.isEmpty
                    ? nil
                    : WordPosition(segment: index, word: wordIndex, compared: token.compared)
            }
        }
    }

    /// Returns `segments` without its first `count` words.
    private static func dropping(
        _ count: Int,
        wordsFrom segments: [TranscriptionSegment]
    ) -> [TranscriptionSegment] {
        var remaining = count
        var kept: [TranscriptionSegment] = []
        for segment in segments {
            guard remaining > 0 else {
                kept.append(segment)
                continue
            }
            let wordCount = comparableWords(of: segment.text).count
            if wordCount <= remaining {
                remaining -= wordCount
            } else {
                kept.append(segment.droppingLeadingWords(remaining))
                remaining = 0
            }
        }
        return kept
    }

    /// Returns the positions of the words forming a longest common subsequence
    /// of `earlier` and `later`, in increasing order of both.
    private static func longestCommonSubsequence(
        _ earlier: [String],
        _ later: [String]
    ) -> [(earlier: Int, later: Int)] {
        // Both inputs are bounded by the words the overlap can hold, a couple
        // of dozen at most, so the textbook quadratic table is the cheapest
        // thing that is also obviously correct.
        var lengths = Array(
            repeating: Array(repeating: 0, count: later.count + 1), count: earlier.count + 1)
        for i in stride(from: earlier.count - 1, through: 0, by: -1) {
            for j in stride(from: later.count - 1, through: 0, by: -1) {
                lengths[i][j] =
                    earlier[i] == later[j]
                    ? lengths[i + 1][j + 1] + 1
                    : max(lengths[i + 1][j], lengths[i][j + 1])
            }
        }

        var pairs: [(earlier: Int, later: Int)] = []
        var i = 0
        var j = 0
        while i < earlier.count, j < later.count {
            if earlier[i] == later[j] {
                pairs.append((earlier: i, later: j))
                i += 1
                j += 1
            } else if lengths[i + 1][j] >= lengths[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }
        return pairs
    }

    /// One whitespace-separated token, in the form it is printed and in the
    /// form it is compared.
    ///
    /// Comparison folds case and drops punctuation, so
    /// `"boundary."` and `"boundary,"` count as the same word. A token of pure
    /// punctuation compares as empty and never takes part in an alignment.
    private struct Token {
        let printed: String
        let compared: String
    }

    private static func tokens(of text: String) -> [Token] {
        text.split(whereSeparator: \.isWhitespace).map {
            Token(
                printed: String($0),
                compared: String($0).lowercased().filter { $0.isLetter || $0.isNumber }
            )
        }
    }

    private static func comparableWords(of text: String) -> [String] {
        tokens(of: text).map(\.compared).filter { !$0.isEmpty }
    }

    /// Position, in the whitespace-split tokens of `text`, one past the
    /// `count`th token that carries letters or digits.
    fileprivate static func tokenIndex(past count: Int, in text: String) -> Int? {
        var seen = 0
        for (index, token) in tokens(of: text).enumerated() where !token.compared.isEmpty {
            seen += 1
            if seen == count { return index + 1 }
        }
        return nil
    }
}

extension TranscriptionSegment {

    /// Returns this segment without its first `count` words, its start advanced
    /// to where those words are estimated to end.
    fileprivate func droppingLeadingWords(_ count: Int) -> TranscriptionSegment {
        let words = text.split(whereSeparator: \.isWhitespace).map(String.init)
        guard count > 0, let cut = SeamReconciler.tokenIndex(past: count, in: text),
            cut < words.count
        else { return self }
        let timings = proportionalWordTimings()
        let advancedStart = cut <= timings.count ? timings[cut - 1].end : start
        return TranscriptionSegment(
            text: words.dropFirst(cut).joined(separator: " "),
            start: advancedStart,
            end: end
        )
    }

    /// Returns this segment cut down to its first `count` words, its end pulled
    /// back to where those words are estimated to finish.
    fileprivate func keepingLeadingWords(_ count: Int) -> TranscriptionSegment {
        let words = text.split(whereSeparator: \.isWhitespace).map(String.init)
        guard count > 0, let cut = SeamReconciler.tokenIndex(past: count, in: text),
            cut < words.count
        else { return self }
        let timings = proportionalWordTimings()
        let pulledBackEnd = cut <= timings.count ? timings[cut - 1].end : end
        return TranscriptionSegment(
            text: words.prefix(cut).joined(separator: " "),
            start: start,
            end: pulledBackEnd
        )
    }
}
