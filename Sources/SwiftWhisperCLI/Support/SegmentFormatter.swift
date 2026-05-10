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
}

/// A formatter that turns ``TranscriptionSegment`` values into per-format text
/// suitable for stdout. Stream-friendly formats (text, srt, vtt) print as
/// segments arrive; buffered formats (json) collect all segments and emit the
/// document once via ``footer(segments:)``.
nonisolated public protocol SegmentFormatter: Sendable {

    /// Header line emitted once before any segments. `nil` for formats that
    /// have no preamble (text, srt, json).
    nonisolated func header() -> String?

    /// Per-segment representation. The `index` is zero-based; SRT uses
    /// `index + 1` for its sequence numbers.
    nonisolated func format(segment: TranscriptionSegment, index: Int) -> String

    /// Footer emitted after all segments. JSON uses this to flush the buffered
    /// document. `nil` for stream formats.
    nonisolated func footer(segments: [TranscriptionSegment]) -> String?

    /// `true` when ``format(segment:index:)`` should not be printed live and
    /// the caller must collect every segment before calling
    /// ``footer(segments:)``. Only JSON requires buffering.
    nonisolated var bufferingRequired: Bool { get }
}

/// Builds the right ``SegmentFormatter`` for the supplied ``OutputFormat``.
public enum SegmentFormatters {

    nonisolated public static func make(_ format: OutputFormat) -> any SegmentFormatter {
        switch format {
        case .text: TextFormatter()
        case .srt: SRTFormatter()
        case .vtt: VTTFormatter()
        case .json: JSONFormatter()
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
