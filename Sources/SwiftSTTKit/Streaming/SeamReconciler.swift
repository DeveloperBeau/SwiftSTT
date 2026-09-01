import Foundation
import SwiftSTTCore

/// Drops the words of a decoded window that the window before it already
/// emitted.
///
/// Every window but the last carries its final
/// ``StreamingWindowPolicy/overlapDuration`` of audio forward, so consecutive
/// windows decode the same audio twice. The overlap is deliberate: a word the
/// model meets at the very start of a window, with no audio in front of it, is
/// the word it most often gets wrong. Decoding it twice is what buys the second
/// reading its context. The cost is that those words arrive twice, and removing
/// the second copy is this type's whole job.
///
/// The copy is found by matching text, not timestamps. A model decoding a short
/// window reports times that overrun the audio it was handed, so a dividing
/// line drawn in seconds falls in the wrong place. What repeats is the words,
/// so the words are what is aligned: the transcript's tail against this
/// window's opening, keeping only what comes after the last word the two
/// readings agree on.
public struct SeamReconciler: Sendable {

    /// One whitespace-separated token, in the form it is printed and in the
    /// form it is compared. Comparison folds case and drops punctuation, so
    /// `"boundary."` and `"boundary,"` count as the same word. A token of pure
    /// punctuation compares as empty and never takes part in a match.
    private struct Token {
        let printed: String
        let compared: String
    }

    /// Words already emitted, most recent last, trimmed to
    /// ``maximumOverlapWordCount``.
    private var emittedTail: [String] = []

    /// Where the previous window's audio ended. A window starting before that
    /// re-covers the difference, and the difference is the only place a
    /// duplicate can come from: text repeated anywhere else is the speaker
    /// repeating themselves, and dropping it would be a loss.
    private var previousWindowEndTime: TimeInterval?

    private let maximumOverlapWordCount: Int

    /// Creates a reconciler.
    ///
    /// - Parameter overlapDuration: seconds each window re-covers of the one
    ///   before it. Must match the policy driving the cutter.
    public init(overlapDuration: TimeInterval) {
        self.maximumOverlapWordCount = max(4, Int(overlapDuration * Self.fastestWordsPerSecond))
    }

    /// About double the fastest human speech, so a duration converts to a
    /// generous upper bound on the words it can hold.
    private static let fastestWordsPerSecond: Double = 12

    /// Returns the segments of `window` that have not already been emitted,
    /// converted from window-local to absolute time.
    ///
    /// - Parameters:
    ///   - segments: what the model returned for `window`, in window-local time.
    ///   - window: the window those segments were decoded from.
    /// - Returns: segments in absolute time, in the order given. A segment the
    ///   overlap ends partway through is emitted with its consumed words
    ///   removed and its start advanced past them.
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

        let overlap = previousWindowEndTime.map { max($0 - window.startTime, 0) } ?? 0
        let kept = overlap > 0 ? droppingOverlap(from: absolute, spanning: overlap) : absolute
        previousWindowEndTime = window.startTime + window.duration

        for segment in kept {
            emittedTail.append(contentsOf: Self.comparableWords(of: segment.text))
        }
        if emittedTail.count > maximumOverlapWordCount {
            emittedTail.removeFirst(emittedTail.count - maximumOverlapWordCount)
        }
        return kept
    }

    /// Aligns the start of `segments` against the end of ``emittedTail`` and
    /// returns `segments` without the words the alignment says are a second
    /// reading of audio already transcribed.
    ///
    /// The two readings rarely agree word for word. The model hears the same
    /// audio with different context either side of it and returns `"runs low"`
    /// where it returned `"runs long"`, or merges two words into one, or omits
    /// the overlap altogether. Requiring the readings to match exactly finds
    /// nothing on real speech, so what is looked for instead is their longest
    /// common subsequence: the words both readings agree on, in order, with
    /// disagreements skipped over. Everything up to the last agreed word is a
    /// second reading and is dropped.
    ///
    /// - Parameter overlapSeconds: how much audio this window re-covers. It
    ///   bounds the alignment, so a phrase the speaker genuinely repeats
    ///   further into the window cannot be mistaken for the carried copy.
    private func droppingOverlap(
        from segments: [TranscriptionSegment],
        spanning overlapSeconds: TimeInterval
    ) -> [TranscriptionSegment] {
        let opening = segments.flatMap { Self.comparableWords(of: $0.text) }
        let span = min(
            max(Int(overlapSeconds * Self.fastestWordsPerSecond), 1), maximumOverlapWordCount)
        let earlier = Array(emittedTail.suffix(span))
        let later = Array(opening.prefix(span))
        guard !earlier.isEmpty, !later.isEmpty else { return segments }

        let agreed = Self.longestCommonSubsequence(earlier, later)
        guard let lastAgreed = agreed.last else { return segments }
        // One agreed word is only trusted when it is the very seam: the last
        // word already emitted reappearing as the first word decoded. Further
        // in, a lone match is as likely to be a common word landing twice by
        // chance, and dropping everything before it would lose real speech.
        guard agreed.count >= 2 || lastAgreed == 0 else { return segments }

        var remaining = lastAgreed + 1
        var kept: [TranscriptionSegment] = []
        for segment in segments {
            guard remaining > 0 else {
                kept.append(segment)
                continue
            }
            let wordCount = Self.comparableWords(of: segment.text).count
            if wordCount <= remaining {
                remaining -= wordCount
            } else {
                kept.append(segment.droppingLeadingWords(remaining))
                remaining = 0
            }
        }
        return kept
    }

    /// Returns the indices into `later` of the words forming a longest common
    /// subsequence of `earlier` and `later`, in increasing order.
    private static func longestCommonSubsequence(
        _ earlier: [String],
        _ later: [String]
    ) -> [Int] {
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

        var indices: [Int] = []
        var i = 0, j = 0
        while i < earlier.count, j < later.count {
            if earlier[i] == later[j] {
                indices.append(j)
                i += 1
                j += 1
            } else if lengths[i + 1][j] >= lengths[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }
        return indices
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

private extension TranscriptionSegment {

    /// Returns this segment without its first `count` words, its start advanced
    /// to where those words are estimated to end.
    func droppingLeadingWords(_ count: Int) -> TranscriptionSegment {
        let words = text.split(whereSeparator: \.isWhitespace).map(String.init)
        guard count > 0, let cut = SeamReconciler.tokenIndex(past: count, in: text), cut < words.count
        else { return self }
        let timings = proportionalWordTimings()
        let advancedStart = cut <= timings.count ? timings[cut - 1].end : start
        return TranscriptionSegment(
            text: words.dropFirst(cut).joined(separator: " "),
            start: advancedStart,
            end: end
        )
    }
}
