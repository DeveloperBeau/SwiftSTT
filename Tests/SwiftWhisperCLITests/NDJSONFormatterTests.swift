import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperCLI

@Suite("NDJSONFormatter")
struct NDJSONFormatterTests {

    @Test("Each segment is one valid JSON line")
    func segmentLineIsJSON() throws {
        let formatter = NDJSONFormatter()
        let segment = TranscriptionSegment(
            text: "And so my fellow Americans",
            start: 0,
            end: 4.5
        )
        let line = formatter.format(segment: segment, index: 0)

        #expect(!line.contains("\n"))
        let data = try #require(line.data(using: .utf8))
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(parsed?["text"] as? String == "And so my fellow Americans")
        let start = parsed?["start"] as? Double
        let end = parsed?["end"] as? Double
        #expect(start == 0)
        #expect((end ?? 0) > 4.4 && (end ?? 0) < 4.6)
    }

    @Test("Header and footer are nil; format does not buffer")
    func headerFooterFlags() {
        let formatter = NDJSONFormatter()
        #expect(formatter.header() == nil)
        #expect(formatter.footer(segments: []) == nil)
        #expect(formatter.bufferingRequired == false)
    }

    @Test("File separator emits a {\"file\":...} JSON line")
    func fileSeparatorIsJSON() throws {
        let formatter = NDJSONFormatter()
        let line = try #require(formatter.fileSeparator(path: "/tmp/clip.wav", fileIndex: 0))
        #expect(!line.contains("\n"))
        let data = try #require(line.data(using: .utf8))
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(parsed?["file"] as? String == "/tmp/clip.wav")
    }

    @Test("Multiple segments produce one line each, no shared envelope")
    func multipleSegments() throws {
        let formatter = NDJSONFormatter()
        let s1 = TranscriptionSegment(text: "a", start: 0, end: 1)
        let s2 = TranscriptionSegment(text: "b", start: 1, end: 2)
        let l1 = formatter.format(segment: s1, index: 0)
        let l2 = formatter.format(segment: s2, index: 1)

        for line in [l1, l2] {
            let data = try #require(line.data(using: .utf8))
            _ = try JSONSerialization.jsonObject(with: data)
        }
        #expect(l1 != l2)
    }
}
