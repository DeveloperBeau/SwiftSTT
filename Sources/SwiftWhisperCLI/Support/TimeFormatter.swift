import Foundation

/// Formats `TimeInterval` values into the `HH:MM:SS` strings used in CLI
/// transcription output. ASCII only.
///
/// The formatter rounds to the nearest second. The pair-formatter
/// (``format(start:end:)``) bumps the end time up by one second whenever
/// rounding would collapse it onto the start, so output never reads
/// `[00:00:00 -> 00:00:00]`.
enum TimeFormatter {

    nonisolated static func format(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds.rounded()))
        let h = total / 3_600
        let m = (total % 3_600) / 60
        let s = total % 60
        return String(format: "%02d:%02d:%02d", h, m, s)
    }

    nonisolated static func format(start: TimeInterval, end: TimeInterval) -> (String, String) {
        let startStr = format(start)
        var endRounded = max(0, Int(end.rounded()))
        let startRounded = max(0, Int(start.rounded()))
        if endRounded <= startRounded {
            endRounded = startRounded + 1
        }
        let h = endRounded / 3_600
        let m = (endRounded % 3_600) / 60
        let s = endRounded % 60
        let endStr = String(format: "%02d:%02d:%02d", h, m, s)
        return (startStr, endStr)
    }

    /// Formats `seconds` as `HH:MM:SS,mmm` for SubRip (SRT). Negative inputs
    /// clamp to zero.
    nonisolated static func srtTimestamp(_ seconds: TimeInterval) -> String {
        let parts = millisecondParts(seconds)
        return String(format: "%02d:%02d:%02d,%03d", parts.h, parts.m, parts.s, parts.ms)
    }

    /// Formats `seconds` as `HH:MM:SS.mmm` for WebVTT. Negative inputs clamp
    /// to zero.
    nonisolated static func vttTimestamp(_ seconds: TimeInterval) -> String {
        let parts = millisecondParts(seconds)
        return String(format: "%02d:%02d:%02d.%03d", parts.h, parts.m, parts.s, parts.ms)
    }

    private nonisolated static func millisecondParts(
        _ seconds: TimeInterval
    ) -> (h: Int, m: Int, s: Int, ms: Int) {
        let safe = max(0, seconds)
        let totalMillis = Int((safe * 1_000).rounded())
        let ms = totalMillis % 1_000
        let totalSeconds = totalMillis / 1_000
        let h = totalSeconds / 3_600
        let m = (totalSeconds % 3_600) / 60
        let s = totalSeconds % 60
        return (h, m, s, ms)
    }
}
