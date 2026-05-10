import Foundation
import Testing
@testable import SwiftWhisperCLI
import SwiftWhisperCore

@Suite("SegmentFormatter")
struct SegmentFormatterTests {

    // MARK: - Text

    @Test("Text format matches the legacy [HH:MM:SS -> HH:MM:SS] text shape")
    func textFormat() {
        let formatter = TextFormatter()
        let segment = TranscriptionSegment(text: "hello world", start: 0, end: 4)
        let line = formatter.format(segment: segment, index: 0)
        #expect(line == "[00:00:00 -> 00:00:04] hello world")
    }

    @Test("Text format has no header or footer")
    func textHeaderFooter() {
        let formatter = TextFormatter()
        #expect(formatter.header() == nil)
        #expect(formatter.footer(segments: []) == nil)
        #expect(formatter.bufferingRequired == false)
    }

    // MARK: - SRT

    @Test("SRT cue uses 1-based sequence and comma timestamps")
    func srtCueShape() {
        let formatter = SRTFormatter()
        let segment = TranscriptionSegment(text: "And so my fellow Americans", start: 0, end: 4.5)
        let block = formatter.format(segment: segment, index: 0)
        #expect(block == "1\n00:00:00,000 --> 00:00:04,500\nAnd so my fellow Americans\n")
    }

    @Test("SRT sequence numbers increment from index + 1")
    func srtSequenceIncrements() {
        let formatter = SRTFormatter()
        let s1 = TranscriptionSegment(text: "first", start: 0, end: 1)
        let s2 = TranscriptionSegment(text: "second", start: 1, end: 2)
        let block1 = formatter.format(segment: s1, index: 0)
        let block2 = formatter.format(segment: s2, index: 1)
        #expect(block1.hasPrefix("1\n"))
        #expect(block2.hasPrefix("2\n"))
    }

    @Test("SRT has no header or footer and is stream-friendly")
    func srtFlags() {
        let formatter = SRTFormatter()
        #expect(formatter.header() == nil)
        #expect(formatter.footer(segments: []) == nil)
        #expect(formatter.bufferingRequired == false)
    }

    // MARK: - VTT

    @Test("VTT emits the WEBVTT preamble")
    func vttHeader() {
        let formatter = VTTFormatter()
        #expect(formatter.header() == "WEBVTT\n")
    }

    @Test("VTT cue uses period timestamps and trailing newline")
    func vttCueShape() {
        let formatter = VTTFormatter()
        let segment = TranscriptionSegment(text: "ask not what your country", start: 4.5, end: 9.0)
        let block = formatter.format(segment: segment, index: 0)
        #expect(block == "00:00:04.500 --> 00:00:09.000\nask not what your country\n")
    }

    @Test("VTT does not buffer and has no footer")
    func vttFlags() {
        let formatter = VTTFormatter()
        #expect(formatter.footer(segments: []) == nil)
        #expect(formatter.bufferingRequired == false)
    }

    @Test("VTT cue blocks are separated by a blank line when joined with newlines")
    func vttBlocksSeparated() {
        let formatter = VTTFormatter()
        let s1 = TranscriptionSegment(text: "one", start: 0, end: 1)
        let s2 = TranscriptionSegment(text: "two", start: 1, end: 2)
        let combined = formatter.format(segment: s1, index: 0) + "\n" + formatter.format(segment: s2, index: 1)
        #expect(combined.contains("one\n\n00:00:01.000"))
    }

    // MARK: - JSON

    @Test("JSON formatter buffers segments")
    func jsonBuffering() {
        let formatter = JSONFormatter()
        #expect(formatter.bufferingRequired == true)
        #expect(formatter.format(segment: .init(text: "x", start: 0, end: 1), index: 0) == "")
        #expect(formatter.header() == nil)
    }

    @Test("JSON footer produces valid JSON with start/end as floats and a segments array")
    func jsonFooter() throws {
        let formatter = JSONFormatter()
        let segments = [
            TranscriptionSegment(text: "And so my fellow Americans", start: 0.0, end: 4.5),
            TranscriptionSegment(text: "ask not what your country can do for you", start: 4.5, end: 9.0),
        ]
        let output = try #require(formatter.footer(segments: segments))
        let data = try #require(output.data(using: .utf8))
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let segs = try #require(parsed?["segments"] as? [[String: Any]])
        #expect(segs.count == 2)
        #expect(segs[0]["text"] as? String == "And so my fellow Americans")
        let start0 = segs[0]["start"] as? Double
        let end0 = segs[0]["end"] as? Double
        #expect(start0 == 0.0)
        #expect((end0 ?? 0).isApproximately(4.5))
    }

    @Test("JSON footer with no segments produces an empty segments array")
    func jsonFooterEmpty() throws {
        let formatter = JSONFormatter()
        let output = try #require(formatter.footer(segments: []))
        let data = try #require(output.data(using: .utf8))
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let segs = try #require(parsed?["segments"] as? [Any])
        #expect(segs.isEmpty)
    }

    // MARK: - SegmentFormatters factory

    @Test("Factory returns the correct concrete type per OutputFormat")
    func factoryDispatch() {
        #expect(SegmentFormatters.make(.text) is TextFormatter)
        #expect(SegmentFormatters.make(.srt) is SRTFormatter)
        #expect(SegmentFormatters.make(.vtt) is VTTFormatter)
        #expect(SegmentFormatters.make(.json) is JSONFormatter)
        #expect(SegmentFormatters.make(.ndjson) is NDJSONFormatter)
        #expect(SegmentFormatters.make(.ttml) is TTMLFormatter)
        #expect(SegmentFormatters.make(.sbv) is SBVFormatter)
    }

    @Test("Buffering flag is true for JSON and TTML, false for streaming formats")
    func bufferingFlags() {
        let buffering: Set<OutputFormat> = [.json, .ttml]
        for format in OutputFormat.allCases {
            let formatter = SegmentFormatters.make(format)
            if buffering.contains(format) {
                #expect(formatter.bufferingRequired == true)
            } else {
                #expect(formatter.bufferingRequired == false)
            }
        }
    }

    // MARK: - TimeFormatter milliseconds

    @Test("SRT timestamp 0s")
    func srtZero() {
        #expect(TimeFormatter.srtTimestamp(0) == "00:00:00,000")
    }

    @Test("SRT timestamp sub-second")
    func srtSubSecond() {
        #expect(TimeFormatter.srtTimestamp(0.5) == "00:00:00,500")
        #expect(TimeFormatter.srtTimestamp(1.234) == "00:00:01,234")
    }

    @Test("SRT timestamp at hour boundary")
    func srtHour() {
        #expect(TimeFormatter.srtTimestamp(3661.987) == "01:01:01,987")
    }

    @Test("SRT timestamp negative clamps to zero")
    func srtNegative() {
        #expect(TimeFormatter.srtTimestamp(-5) == "00:00:00,000")
    }

    @Test("VTT timestamp uses period separator")
    func vttSeparator() {
        #expect(TimeFormatter.vttTimestamp(4.5) == "00:00:04.500")
    }

    @Test("VTT timestamp at hour boundary")
    func vttHour() {
        #expect(TimeFormatter.vttTimestamp(3661.987) == "01:01:01.987")
    }

    @Test("Millisecond rounding rounds half-millis to nearest")
    func millisecondRounding() {
        #expect(TimeFormatter.srtTimestamp(0.0006) == "00:00:00,001")
        #expect(TimeFormatter.srtTimestamp(0.0004) == "00:00:00,000")
    }

    @Test("Millisecond formatter wraps around the second boundary correctly")
    func millisecondWrap() {
        #expect(TimeFormatter.srtTimestamp(0.999) == "00:00:00,999")
        #expect(TimeFormatter.srtTimestamp(1.000) == "00:00:01,000")
    }
}

private extension Double {
    func isApproximately(_ other: Double, tolerance: Double = 0.001) -> Bool {
        abs(self - other) < tolerance
    }
}
