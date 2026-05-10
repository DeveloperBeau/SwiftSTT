import Foundation
import Testing
@testable import SwiftWhisperKit
import SwiftWhisperCore

@Suite("WordAligner")
struct WordAlignerTests {

    // MARK: - Default fallback

    @Test("Default fallback uses proportional split when matrix is nil")
    func fallbackUsesProportionalSplit() async {
        let aligner = WordAligner()
        let segment = TranscriptionSegment(text: "hello world", start: 0, end: 1)
        let timings = await aligner.align(segment: segment, attentionMatrix: nil)
        #expect(timings.count == 2)
        #expect(timings[0].word == "hello")
        #expect(timings[1].word == "world")
        #expect(timings[1].end == 1)
    }

    @Test("Default fallback also activates on empty matrix")
    func emptyMatrixFallsBack() async {
        let aligner = WordAligner()
        let segment = TranscriptionSegment(text: "hello world", start: 0, end: 1)
        let timings = await aligner.align(segment: segment, attentionMatrix: [])
        #expect(timings.count == 2)
    }

    @Test("Empty segment text returns empty alignment")
    func emptyTextReturnsEmpty() async {
        let aligner = WordAligner()
        let segment = TranscriptionSegment(text: "", start: 0, end: 1)
        let timings = await aligner.align(segment: segment, attentionMatrix: nil)
        #expect(timings.isEmpty)
    }

    @Test("Single word returns one timing covering the segment")
    func singleWord() async {
        let aligner = WordAligner()
        let segment = TranscriptionSegment(text: "hello", start: 0.5, end: 1.5)
        let attention: [[Float]] = [[1, 0.5, 0.2]]
        let timings = await aligner.align(segment: segment, attentionMatrix: attention)
        #expect(timings.count == 1)
        #expect(timings[0].word == "hello")
        #expect(timings[0].start == 0.5)
        #expect(timings[0].end == 1.5)
    }

    @Test("Non-nil matrix produces alignment for multi-word segments")
    func nonNilMatrixProducesAlignment() async {
        let aligner = WordAligner()
        let segment = TranscriptionSegment(text: "foo bar baz", start: 0, end: 3)
        // 3 text frames, 6 audio frames; identity-ish attention so each text frame
        // aligns to a contiguous block of audio frames.
        let attention: [[Float]] = [
            [1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 1],
        ]
        let timings = await aligner.align(segment: segment, attentionMatrix: attention)
        #expect(timings.count == 3)
        // Last word's end must be pinned to the segment end.
        #expect(timings[2].end == 3)
        // Words must come out in order.
        for i in 0..<(timings.count - 1) {
            #expect(timings[i].start <= timings[i + 1].start)
        }
    }

    // MARK: - dtwPath

    @Test("dtwPath produces monotonic path on identity cost matrix")
    func dtwIdentityMonotonic() {
        let n = 4
        var matrix = [[Float]](repeating: [Float](repeating: 1, count: n), count: n)
        for i in 0..<n { matrix[i][i] = 0 }

        let path = WordAligner.dtwPath(costMatrix: matrix)
        #expect(path.first?.0 == 0)
        #expect(path.first?.1 == 0)
        #expect(path.last?.0 == n - 1)
        #expect(path.last?.1 == n - 1)
        for i in 0..<(path.count - 1) {
            let (r0, c0) = path[i]
            let (r1, c1) = path[i + 1]
            #expect(r1 >= r0)
            #expect(c1 >= c0)
        }
    }

    @Test("dtwPath handles non-square matrix")
    func dtwNonSquare() {
        // 3 rows x 5 cols.
        let matrix: [[Float]] = [
            [0, 1, 2, 3, 4],
            [1, 0, 1, 2, 3],
            [2, 1, 0, 1, 2],
        ]
        let path = WordAligner.dtwPath(costMatrix: matrix)
        #expect(path.first?.0 == 0)
        #expect(path.first?.1 == 0)
        #expect(path.last?.0 == 2)
        #expect(path.last?.1 == 4)
        // Path length is at least max(rows, cols).
        #expect(path.count >= 5)
    }

    @Test("dtwPath returns single point for 1x1 matrix")
    func dtwSinglePoint() {
        let path = WordAligner.dtwPath(costMatrix: [[42]])
        #expect(path.count == 1)
        #expect(path[0].0 == 0)
        #expect(path[0].1 == 0)
    }

    @Test("dtwPath returns empty on empty matrix")
    func dtwEmpty() {
        #expect(WordAligner.dtwPath(costMatrix: []).isEmpty)
        #expect(WordAligner.dtwPath(costMatrix: [[]]).isEmpty)
    }

    @Test("dtwPath handles 1xN matrix")
    func dtw1xN() {
        let matrix: [[Float]] = [[0, 1, 2, 3]]
        let path = WordAligner.dtwPath(costMatrix: matrix)
        #expect(path.count == 4)
        for (i, (r, c)) in path.enumerated() {
            #expect(r == 0)
            #expect(c == i)
        }
    }

    @Test("dtwPath handles Nx1 matrix")
    func dtwNx1() {
        let matrix: [[Float]] = [[0], [1], [2]]
        let path = WordAligner.dtwPath(costMatrix: matrix)
        #expect(path.count == 3)
        for (i, (r, c)) in path.enumerated() {
            #expect(c == 0)
            #expect(r == i)
        }
    }

    // MARK: - medianFilter

    @Test("medianFilter on constant input returns input")
    func medianConstant() {
        let input: [Float] = [5, 5, 5, 5, 5]
        let out = WordAligner.medianFilter(values: input, windowSize: 3)
        #expect(out == input)
    }

    @Test("medianFilter on noisy input smooths spikes")
    func medianSmoothsSpikes() {
        let input: [Float] = [1, 1, 1, 100, 1, 1, 1]
        let out = WordAligner.medianFilter(values: input, windowSize: 3)
        // The 100 spike at index 3 should be replaced with median of [1,100,1] = 1.
        #expect(out[3] == 1)
    }

    @Test("medianFilter handles empty input")
    func medianEmpty() {
        let out = WordAligner.medianFilter(values: [], windowSize: 3)
        #expect(out.isEmpty)
    }

    @Test("medianFilter handles window > input length")
    func medianWindowLargerThanInput() {
        let input: [Float] = [1, 2, 3]
        let out = WordAligner.medianFilter(values: input, windowSize: 100)
        // No filtering applied; input returned unchanged.
        #expect(out == input)
    }

    @Test("medianFilter with windowSize <= 1 returns input unchanged")
    func medianWindowOne() {
        let input: [Float] = [3, 1, 4, 1, 5]
        let out = WordAligner.medianFilter(values: input, windowSize: 1)
        #expect(out == input)
    }

    @Test("medianFilter preserves length")
    func medianPreservesLength() {
        let input: [Float] = [3, 1, 4, 1, 5, 9, 2, 6]
        let out = WordAligner.medianFilter(values: input, windowSize: 3)
        #expect(out.count == input.count)
    }
}
