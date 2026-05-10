import Foundation

/// How the streaming pipeline turns decoder tokens into ``TranscriptionSegment`` values.
///
/// Two strategies, each with different trade-offs:
///
/// - ``stableTokens`` runs the local-agreement policy on the raw token stream and
///   emits one segment per agreement step. Time bounds come from the rolling mel
///   window (decoder-time, not Whisper's predicted timestamps). Best when the
///   downstream UI cares about latency-of-confirmation more than precise audio
///   alignment.
/// - ``timestampSegments`` lets Whisper emit `<|t.tt|>` markers and uses
///   ``WhisperDecoder/parseSegments(tokens:tokenizer:windowOffsetSeconds:)`` to
///   split on them. Each output segment carries Whisper's own start/end times.
///   Best when the UI plays back audio and wants accurate seek points.
public enum SegmentEmissionMode: Sendable, Equatable {

    /// Local-agreement on raw tokens. Time bounds come from the rolling mel cursor.
    case stableTokens

    /// Whisper-emitted `<|t.tt|>` timestamp pairs split into per-segment slices.
    case timestampSegments
}
