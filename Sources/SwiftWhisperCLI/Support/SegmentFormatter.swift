import ArgumentParser
import Foundation
import SwiftWhisperCore

/// Output formats supported by the `transcribe` and `transcribe-mic`
/// subcommands. Selected via `--format` on the CLI.
public enum OutputFormat: String, ExpressibleByArgument, Sendable, CaseIterable {
    case text
    case srt
    case vtt
    case json
    case ndjson
    case ttml
    case sbv
}

/// A formatter that turns ``TranscriptionSegment`` values into per-format text
/// suitable for stdout. Stream-friendly formats (text, srt, vtt, ndjson, sbv)
/// print as segments arrive; buffered formats (json, ttml) collect all segments
/// and emit the document once via ``footer(segments:)``.
nonisolated public protocol SegmentFormatter: Sendable {

    /// Header line emitted once before any segments. `nil` for formats that
    /// have no preamble.
    nonisolated func header() -> String?

    /// Per-segment representation. The `index` is zero-based; SRT uses
    /// `index + 1` for its sequence numbers.
    nonisolated func format(segment: TranscriptionSegment, index: Int) -> String

    /// Footer emitted after all segments. JSON and TTML use this to flush the
    /// buffered document. `nil` for stream formats.
    nonisolated func footer(segments: [TranscriptionSegment]) -> String?

    /// Inter-file separator emitted before each subsequent file's segments in
    /// batch mode. Default implementation returns `# basename` (matches the
    /// existing text/srt/vtt convention). Returns `nil` to suppress.
    nonisolated func fileSeparator(path: String, fileIndex: Int) -> String?

    /// `true` when ``format(segment:index:)`` should not be printed live and
    /// the caller must collect every segment before calling
    /// ``footer(segments:)``. JSON and TTML require buffering.
    nonisolated var bufferingRequired: Bool { get }
}

extension SegmentFormatter {

    nonisolated public func fileSeparator(path: String, fileIndex: Int) -> String? {
        let name = (path as NSString).lastPathComponent
        return fileIndex == 0 ? "# \(name)" : "\n# \(name)"
    }
}

/// Builds the right ``SegmentFormatter`` for the supplied ``OutputFormat``.
public enum SegmentFormatters {

    nonisolated public static func make(_ format: OutputFormat) -> any SegmentFormatter {
        switch format {
        case .text: TextFormatter()
        case .srt: SRTFormatter()
        case .vtt: VTTFormatter()
        case .json: JSONFormatter()
        case .ndjson: NDJSONFormatter()
        case .ttml: TTMLFormatter()
        case .sbv: SBVFormatter()
        }
    }
}

// MARK: - Text

/// Matches the original M7 transcription output: `[HH:MM:SS -> HH:MM:SS] text`.
nonisolated public struct TextFormatter: SegmentFormatter {

    public init() {}

    nonisolated public func header() -> String? { nil }

    nonisolated public func format(segment: TranscriptionSegment, index: Int) -> String {
        let (start, end) = TimeFormatter.format(start: segment.start, end: segment.end)
        return "[\(start) -> \(end)] \(segment.text)"
    }

    nonisolated public func footer(segments: [TranscriptionSegment]) -> String? { nil }

    nonisolated public var bufferingRequired: Bool { false }
}

// MARK: - SRT

/// SubRip subtitle format. Each cue is a 1-based sequence number, a comma-
/// separated `HH:MM:SS,mmm --> HH:MM:SS,mmm` timestamp line, the text, and a
/// blank line.
nonisolated public struct SRTFormatter: SegmentFormatter {

    public init() {}

    nonisolated public func header() -> String? { nil }

    nonisolated public func format(segment: TranscriptionSegment, index: Int) -> String {
        let start = TimeFormatter.srtTimestamp(segment.start)
        let end = TimeFormatter.srtTimestamp(segment.end)
        return "\(index + 1)\n\(start) --> \(end)\n\(segment.text)\n"
    }

    nonisolated public func footer(segments: [TranscriptionSegment]) -> String? { nil }

    nonisolated public var bufferingRequired: Bool { false }
}

// MARK: - VTT

/// WebVTT subtitle format. `WEBVTT` preamble, then period-separated
/// timestamps and a trailing blank line per cue.
nonisolated public struct VTTFormatter: SegmentFormatter {

    public init() {}

    nonisolated public func header() -> String? { "WEBVTT\n" }

    nonisolated public func format(segment: TranscriptionSegment, index: Int) -> String {
        let start = TimeFormatter.vttTimestamp(segment.start)
        let end = TimeFormatter.vttTimestamp(segment.end)
        return "\(start) --> \(end)\n\(segment.text)\n"
    }

    nonisolated public func footer(segments: [TranscriptionSegment]) -> String? { nil }

    nonisolated public var bufferingRequired: Bool { false }
}

// MARK: - JSON

/// Structured JSON output. Buffers all segments and emits a single document on
/// ``footer(segments:)``. `start` and `end` are encoded as `Float` seconds.
nonisolated public struct JSONFormatter: SegmentFormatter {

    public init() {}

    nonisolated public func header() -> String? { nil }

    nonisolated public func format(segment: TranscriptionSegment, index: Int) -> String { "" }

    nonisolated public func footer(segments: [TranscriptionSegment]) -> String? {
        let payload = SingleFilePayload(
            segments: segments.map(JSONSegmentPayload.init(segment:))
        )
        return Self.encode(payload)
    }

    nonisolated public var bufferingRequired: Bool { true }

    nonisolated static func encode(_ value: some Encodable) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        guard let data = try? encoder.encode(value) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

// MARK: - NDJSON

/// Newline-delimited JSON. Each segment becomes a single-line JSON object
/// terminated by `\n`. Streamable; the segment line is yielded as soon as it
/// is decoded.
///
/// Batch mode emits a `{"file":"path"}` line before each file's segments so
/// downstream consumers can group results without inspecting timestamps.
nonisolated public struct NDJSONFormatter: SegmentFormatter {

    public init() {}

    nonisolated public func header() -> String? { nil }

    nonisolated public func format(segment: TranscriptionSegment, index: Int) -> String {
        let payload = JSONSegmentPayload(segment: segment)
        return Self.encodeLine(payload) ?? ""
    }

    nonisolated public func footer(segments: [TranscriptionSegment]) -> String? { nil }

    nonisolated public func fileSeparator(path: String, fileIndex: Int) -> String? {
        let payload = NDJSONFileMarker(file: path)
        return Self.encodeLine(payload)
    }

    nonisolated public var bufferingRequired: Bool { false }

    nonisolated static func encodeLine(_ value: some Encodable) -> String? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(value) else { return nil }
        return String(data: data, encoding: .utf8)
    }
}

nonisolated struct NDJSONFileMarker: Encodable, Sendable {
    let file: String
}

// MARK: - TTML

/// Timed Text Markup Language output. Buffers all segments and emits a full
/// XML document on ``footer(segments:)``.
///
/// Each segment becomes a `<p begin="..." end="...">text</p>` inside a single
/// `<div>`. Special characters in text are escaped against the five XML
/// predefined entities.
nonisolated public struct TTMLFormatter: SegmentFormatter {

    public init() {}

    nonisolated public func header() -> String? { nil }

    nonisolated public func format(segment: TranscriptionSegment, index: Int) -> String { "" }

    nonisolated public func footer(segments: [TranscriptionSegment]) -> String? {
        var lines: [String] = []
        lines.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
        lines.append("<tt xmlns=\"http://www.w3.org/ns/ttml\" xml:lang=\"en\">")
        lines.append("  <body>")
        lines.append("    <div>")
        for segment in segments {
            let begin = TimeFormatter.ttmlTimestamp(segment.start)
            let end = TimeFormatter.ttmlTimestamp(segment.end)
            let text = Self.escapeXML(segment.text)
            lines.append("      <p begin=\"\(begin)\" end=\"\(end)\">\(text)</p>")
        }
        lines.append("    </div>")
        lines.append("  </body>")
        lines.append("</tt>")
        return lines.joined(separator: "\n")
    }

    nonisolated public func fileSeparator(path: String, fileIndex: Int) -> String? {
        nil
    }

    nonisolated public var bufferingRequired: Bool { true }

    /// Escapes the five XML predefined entities. Anything else is passed
    /// through. Order matters: `&` must be escaped first or it would double-
    /// escape the entities written for the other characters.
    nonisolated static func escapeXML(_ string: String) -> String {
        var result = string
        result = result.replacingOccurrences(of: "&", with: "&amp;")
        result = result.replacingOccurrences(of: "<", with: "&lt;")
        result = result.replacingOccurrences(of: ">", with: "&gt;")
        result = result.replacingOccurrences(of: "\"", with: "&quot;")
        result = result.replacingOccurrences(of: "'", with: "&apos;")
        return result
    }
}

// MARK: - SBV

/// YouTube SubViewer (SBV) format. Each cue is a `H:MM:SS.mmm,H:MM:SS.mmm`
/// time line, the text, and a trailing blank line. Streamable.
nonisolated public struct SBVFormatter: SegmentFormatter {

    public init() {}

    nonisolated public func header() -> String? { nil }

    nonisolated public func format(segment: TranscriptionSegment, index: Int) -> String {
        let start = TimeFormatter.sbvTimestamp(segment.start)
        let end = TimeFormatter.sbvTimestamp(segment.end)
        return "\(start),\(end)\n\(segment.text)\n"
    }

    nonisolated public func footer(segments: [TranscriptionSegment]) -> String? { nil }

    nonisolated public var bufferingRequired: Bool { false }
}

// MARK: - JSON shared payloads

/// Encodable shape for `{"segments": [...]}`.
nonisolated struct SingleFilePayload: Encodable, Sendable {
    let segments: [JSONSegmentPayload]
}

/// Encodable shape for `{"files": [{"path": ..., "segments": [...]}, ...]}`.
nonisolated struct BatchFilePayload: Encodable, Sendable {
    nonisolated struct File: Encodable, Sendable {
        let path: String
        let segments: [JSONSegmentPayload]
    }
    let files: [File]
}

nonisolated struct JSONSegmentPayload: Encodable, Sendable {
    let start: Float
    let end: Float
    let text: String

    init(segment: TranscriptionSegment) {
        self.start = Float(segment.start)
        self.end = Float(segment.end)
        self.text = segment.text
    }
}
