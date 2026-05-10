import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperCLI

@Suite("SBVFormatter")
struct SBVFormatterTests {

    @Test("Time format is H:MM:SS.mmm with comma between start and end")
    func cueShape() {
        let formatter = SBVFormatter()
        let segment = TranscriptionSegment(
            text: "And so my fellow Americans",
            start: 0,
            end: 4.5
        )
        let block = formatter.format(segment: segment, index: 0)
        #expect(block == "0:00:00.000,0:00:04.500\nAnd so my fellow Americans\n")
    }

    @Test("Hours are not zero-padded")
    func hoursUnpadded() {
        #expect(TimeFormatter.sbvTimestamp(3661.987) == "1:01:01.987")
        #expect(TimeFormatter.sbvTimestamp(36000) == "10:00:00.000")
    }

    @Test("Concatenating two cues separates them with a blank line")
    func cuesAreSeparated() {
        let formatter = SBVFormatter()
        let s1 = TranscriptionSegment(text: "one", start: 0, end: 1)
        let s2 = TranscriptionSegment(text: "two", start: 1, end: 2)
        let combined =
            formatter.format(segment: s1, index: 0)
            + "\n"
            + formatter.format(segment: s2, index: 1)
        #expect(combined.contains("one\n\n0:00:01.000,0:00:02.000"))
    }

    @Test("Header and footer are nil; streamable")
    func flags() {
        let formatter = SBVFormatter()
        #expect(formatter.header() == nil)
        #expect(formatter.footer(segments: []) == nil)
        #expect(formatter.bufferingRequired == false)
    }
}
