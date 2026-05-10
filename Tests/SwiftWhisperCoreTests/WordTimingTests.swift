import Foundation
import Testing

@testable import SwiftWhisperCore

@Suite("WordTiming proportional split")
struct WordTimingTests {

    @Test("Empty text returns empty array")
    func emptyText() {
        let segment = TranscriptionSegment(text: "", start: 0, end: 1)
        #expect(segment.proportionalWordTimings().isEmpty)
    }

    @Test("Whitespace-only text returns empty array")
    func whitespaceOnly() {
        let segment = TranscriptionSegment(text: "   \t\n  ", start: 0, end: 2)
        #expect(segment.proportionalWordTimings().isEmpty)
    }

    @Test("Single word gets full duration")
    func singleWord() {
        let segment = TranscriptionSegment(text: "hello", start: 1, end: 3)
        let timings = segment.proportionalWordTimings()
        #expect(timings.count == 1)
        #expect(timings[0].word == "hello")
        #expect(timings[0].start == 1)
        #expect(timings[0].end == 3)
    }

    @Test("Two equal-length words split evenly")
    func twoEqualWords() {
        let segment = TranscriptionSegment(text: "foo bar", start: 0, end: 2)
        let timings = segment.proportionalWordTimings()
        #expect(timings.count == 2)
        #expect(timings[0].word == "foo")
        #expect(timings[1].word == "bar")
        #expect(abs(timings[0].end - 1.0) < 1e-9)
        #expect(abs(timings[1].start - 1.0) < 1e-9)
        #expect(timings[1].end == 2)
    }

    @Test("Words with different lengths get proportional duration")
    func proportionalDuration() {
        // "a" + "bbbb" -> 1 + 4 = 5 chars, 2-second segment -> a: 0.4 s, bbbb: 1.6 s
        let segment = TranscriptionSegment(text: "a bbbb", start: 0, end: 2)
        let timings = segment.proportionalWordTimings()
        #expect(timings.count == 2)
        #expect(abs(timings[0].end - 0.4) < 1e-9)
        #expect(abs(timings[1].start - 0.4) < 1e-9)
        #expect(abs(timings[1].end - 2.0) < 1e-9)
    }

    @Test("Timings are monotonic and contiguous (no overlap, no gap)")
    func monotonic() {
        let segment = TranscriptionSegment(text: "the quick brown fox", start: 5, end: 10)
        let timings = segment.proportionalWordTimings()
        #expect(timings.count == 4)
        for i in 0..<timings.count {
            #expect(timings[i].end >= timings[i].start)
            if i > 0 {
                #expect(abs(timings[i].start - timings[i - 1].end) < 1e-9)
            }
        }
    }

    @Test("Sum of word durations equals segment duration")
    func sumEqualsDuration() {
        let segment = TranscriptionSegment(text: "alpha beta gamma delta", start: 2, end: 7)
        let timings = segment.proportionalWordTimings()
        let total = timings.reduce(0.0) { $0 + ($1.end - $1.start) }
        #expect(abs(total - (segment.end - segment.start)) < 1e-9)
    }

    @Test("Last word ends exactly at segment end")
    func lastWordPinnedToEnd() {
        let segment = TranscriptionSegment(text: "one two three", start: 1.5, end: 3.5)
        let timings = segment.proportionalWordTimings()
        #expect(timings.last?.end == 3.5)
        #expect(timings.first?.start == 1.5)
    }
}
