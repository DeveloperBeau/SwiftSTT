import Foundation
import Testing

@testable import SwiftSTTCLI

@Suite("TimeFormatter")
struct TimeFormatterTests {

    @Test("0 seconds")
    func zero() {
        #expect(TimeFormatter.format(0) == "00:00:00")
    }

    @Test("65 seconds is 00:01:05")
    func sixtyFive() {
        #expect(TimeFormatter.format(65) == "00:01:05")
    }

    @Test("3661 seconds is 01:01:01")
    func hour() {
        #expect(TimeFormatter.format(3_661) == "01:01:01")
    }

    @Test("Negative time clamps to zero")
    func negative() {
        #expect(TimeFormatter.format(-1) == "00:00:00")
    }

    @Test("Equal start and end bumps end by 1 second")
    func equalStartEnd() {
        let (start, end) = TimeFormatter.format(start: 0, end: 0)
        #expect(start == "00:00:00")
        #expect(end == "00:00:01")
    }

    @Test("End below start still bumps to start + 1")
    func endBelowStart() {
        let (start, end) = TimeFormatter.format(start: 10, end: 5)
        #expect(start == "00:00:10")
        #expect(end == "00:00:11")
    }

    @Test("Distinct start and end format normally")
    func distinct() {
        let (start, end) = TimeFormatter.format(start: 0, end: 4)
        #expect(start == "00:00:00")
        #expect(end == "00:00:04")
    }

    @Test("Sub-second times round to nearest")
    func roundsToNearest() {
        #expect(TimeFormatter.format(0.4) == "00:00:00")
        #expect(TimeFormatter.format(0.6) == "00:00:01")
    }
}
