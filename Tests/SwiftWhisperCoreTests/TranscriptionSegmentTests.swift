import Testing

@testable import SwiftWhisperCore

@Suite("TranscriptionSegment")
struct TranscriptionSegmentTests {

    @Test("Init stores values")
    func initStoresValues() {
        let seg = TranscriptionSegment(text: "hello", start: 1.0, end: 2.5)
        #expect(seg.text == "hello")
        #expect(seg.start == 1.0)
        #expect(seg.end == 2.5)
    }

    @Test("Equality compares all fields")
    func equality() {
        let a = TranscriptionSegment(text: "x", start: 0, end: 1)
        let b = TranscriptionSegment(text: "x", start: 0, end: 1)
        let c = TranscriptionSegment(text: "y", start: 0, end: 1)
        #expect(a == b)
        #expect(a != c)
    }
}
