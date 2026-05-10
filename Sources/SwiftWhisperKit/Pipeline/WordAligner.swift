import Foundation
import SwiftWhisperCore

/// Produces per-word time alignments for a ``TranscriptionSegment``.
///
/// ## Two paths
///
/// 1. **Proportional fallback**. With no attention matrix supplied, the
///    aligner returns ``TranscriptionSegment/proportionalWordTimings()``.
///    Good enough for karaoke-style highlighting on conversational speech;
///    gives up on filler words and very short utterances.
///
/// 2. **DTW alignment**. With a `[textFrames x audioFrames]` cross-attention
///    matrix supplied, the aligner runs dynamic time warping to find the
///    monotonic path that best aligns each text frame with an audio frame,
///    then derives word boundaries by walking the path.
///
/// ## Why the matrix is a parameter, not extracted internally
///
/// The argmaxinc Whisper Core ML exports surface only the final logits; the
/// per-layer attention weights needed for DTW alignment are not exposed by
/// `MLFeatureProvider`. Callers who export their own decoder with the
/// attention tensors named in the output dictionary can pass them through;
/// callers using the standard distribution get the proportional fallback.
public actor WordAligner {

    /// Creates a new WordAligner with the supplied values.
    public init() {}

    /// Aligns a segment's words against the audio it covers.
    ///
    /// - Parameters:
    ///   - segment: the segment to align.
    ///   - attentionMatrix: optional `[textFrames x audioFrames]` cross-
    ///     attention matrix. If `nil`, the proportional fallback is used.
    /// - Returns: per-word timings covering the segment.
    public func align(
        segment: TranscriptionSegment,
        attentionMatrix: [[Float]]? = nil
    ) async -> [WordTiming] {
        guard let matrix = attentionMatrix, !matrix.isEmpty, !matrix[0].isEmpty else {
            return segment.proportionalWordTimings()
        }

        let words = segment.text.split(whereSeparator: \.isWhitespace).map(String.init)
        guard !words.isEmpty else { return [] }
        if words.count == 1 {
            return [WordTiming(word: words[0], start: segment.start, end: segment.end)]
        }

        let textFrames = matrix.count
        let audioFrames = matrix[0].count

        // DTW is defined on a cost matrix. Convert attention (similarity) to
        // cost by negation; large attention => low cost.
        var costMatrix = [[Float]](
            repeating: [Float](repeating: 0, count: audioFrames), count: textFrames)
        for i in 0..<textFrames {
            for j in 0..<audioFrames {
                costMatrix[i][j] = -matrix[i][j]
            }
        }

        let path = Self.dtwPath(costMatrix: costMatrix)
        guard !path.isEmpty else {
            return segment.proportionalWordTimings()
        }

        // For each text frame, find the audio frame the path assigned to it.
        // If multiple audio frames map to the same text frame, take the first
        // (the boundary).
        var textToAudio = [Int](repeating: 0, count: textFrames)
        var seen = [Bool](repeating: false, count: textFrames)
        for (t, a) in path {
            if !seen[t] {
                textToAudio[t] = a
                seen[t] = true
            }
        }

        // Distribute words evenly over the textFrames axis, then look up the
        // audio frame for each word boundary.
        let segmentDuration = segment.end - segment.start
        let secondsPerAudioFrame = audioFrames > 0 ? segmentDuration / Double(audioFrames) : 0
        var timings: [WordTiming] = []
        timings.reserveCapacity(words.count)

        for (i, word) in words.enumerated() {
            let textFracStart = Double(i) / Double(words.count)
            let textFracEnd = Double(i + 1) / Double(words.count)
            let textFrameStart = min(textFrames - 1, Int(textFracStart * Double(textFrames)))
            let textFrameEnd = min(
                textFrames - 1, max(textFrameStart, Int(textFracEnd * Double(textFrames)) - 1))

            let audioStart = textToAudio[textFrameStart]
            let audioEnd = textToAudio[textFrameEnd]

            let start = segment.start + Double(audioStart) * secondsPerAudioFrame
            let end: TimeInterval =
                i == words.count - 1
                ? segment.end
                : segment.start + Double(audioEnd + 1) * secondsPerAudioFrame
            timings.append(WordTiming(word: word, start: start, end: max(start, end)))
        }
        return timings
    }

    // MARK: - DTW math

    /// Standard dynamic time warping.
    ///
    /// Given an `N x M` cost matrix, returns the lowest-cost monotonic path from `(0,0)` to `(N-1, M-1)` as a list
    /// of `(row, col)` index pairs.
    ///
    /// Uses the three classic moves at each step: down, right, diagonal.
    /// Returns an empty array if either dimension is zero.
    public static func dtwPath(costMatrix: [[Float]]) -> [(Int, Int)] {
        let n = costMatrix.count
        guard n > 0 else { return [] }
        let m = costMatrix[0].count
        guard m > 0 else { return [] }

        // Accumulator: D[i][j] = min cost from (0,0) to (i,j).
        var d = [[Float]](repeating: [Float](repeating: .infinity, count: m), count: n)
        d[0][0] = costMatrix[0][0]
        for i in 1..<n {
            d[i][0] = d[i - 1][0] + costMatrix[i][0]
        }
        for j in 1..<m {
            d[0][j] = d[0][j - 1] + costMatrix[0][j]
        }
        for i in 1..<n {
            for j in 1..<m {
                let best = min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
                d[i][j] = best + costMatrix[i][j]
            }
        }

        var path: [(Int, Int)] = []
        var i = n - 1
        var j = m - 1
        path.append((i, j))
        while i > 0 || j > 0 {
            if i == 0 {
                j -= 1
            } else if j == 0 {
                i -= 1
            } else {
                let up = d[i - 1][j]
                let left = d[i][j - 1]
                let diag = d[i - 1][j - 1]
                if diag <= up && diag <= left {
                    i -= 1
                    j -= 1
                } else if up <= left {
                    i -= 1
                } else {
                    j -= 1
                }
            }
            path.append((i, j))
        }
        return path.reversed()
    }

    /// Sliding-window median filter.
    ///
    /// Useful for smoothing attention rows before feeding them to ``dtwPath(costMatrix:)`` so a single noisy
    /// attention spike does not bend the alignment.
    ///
    /// Returns the input unchanged if `windowSize <= 1` or the input is
    /// shorter than the window.
    public static func medianFilter(values: [Float], windowSize: Int) -> [Float] {
        guard windowSize > 1 else { return values }
        guard values.count >= windowSize else { return values }
        let half = windowSize / 2
        var out = values
        for i in 0..<values.count {
            let lo = max(0, i - half)
            let hi = min(values.count, i + half + 1)
            let window = Array(values[lo..<hi]).sorted()
            out[i] = window[window.count / 2]
        }
        return out
    }
}
