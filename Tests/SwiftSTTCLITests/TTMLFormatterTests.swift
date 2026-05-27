import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTCLI

@Suite("TTMLFormatter")
struct TTMLFormatterTests {

    @Test("Document opens with XML preamble and tt namespace")
    func preamble() throws {
        let formatter = TTMLFormatter()
        let segments = [
            TranscriptionSegment(text: "hello", start: 0, end: 1)
        ]
        let doc = try #require(formatter.footer(segments: segments))
        #expect(doc.hasPrefix("<?xml version=\"1.0\" encoding=\"UTF-8\"?>"))
        #expect(doc.contains("<tt xmlns=\"http://www.w3.org/ns/ttml\""))
        #expect(doc.contains("<body>"))
        #expect(doc.contains("<div>"))
    }

    @Test("Each segment is a <p> with begin and end attributes")
    func paragraphPerSegment() throws {
        let formatter = TTMLFormatter()
        let segments = [
            TranscriptionSegment(text: "first", start: 0, end: 4.5),
            TranscriptionSegment(text: "second", start: 4.5, end: 9),
        ]
        let doc = try #require(formatter.footer(segments: segments))
        #expect(doc.contains("<p begin=\"00:00:00.000\" end=\"00:00:04.500\">first</p>"))
        #expect(doc.contains("<p begin=\"00:00:04.500\" end=\"00:00:09.000\">second</p>"))
    }

    @Test("Document closes div, body, tt")
    func closingTags() throws {
        let formatter = TTMLFormatter()
        let doc = try #require(formatter.footer(segments: []))
        #expect(doc.contains("</div>"))
        #expect(doc.contains("</body>"))
        #expect(doc.hasSuffix("</tt>"))
    }

    @Test("Special characters are XML-escaped")
    func escaping() throws {
        let formatter = TTMLFormatter()
        let segment = TranscriptionSegment(
            text: "Tom & Jerry said <hi> \"world\" 'home'",
            start: 0,
            end: 1
        )
        let doc = try #require(formatter.footer(segments: [segment]))
        #expect(doc.contains("Tom &amp; Jerry"))
        #expect(doc.contains("&lt;hi&gt;"))
        #expect(doc.contains("&quot;world&quot;"))
        #expect(doc.contains("&apos;home&apos;"))
        #expect(!doc.contains("&hi"))
    }

    @Test("Header is nil, format returns empty, buffering required")
    func flags() {
        let formatter = TTMLFormatter()
        #expect(formatter.header() == nil)
        #expect(formatter.format(segment: .init(text: "x", start: 0, end: 1), index: 0) == "")
        #expect(formatter.bufferingRequired == true)
        #expect(formatter.fileSeparator(path: "/x.wav", fileIndex: 0) == nil)
    }

    @Test("XML escaper is amp-first to avoid double-encoding")
    func ampFirst() {
        let escaped = TTMLFormatter.escapeXML("&lt;")
        #expect(escaped == "&amp;lt;")
    }
}
