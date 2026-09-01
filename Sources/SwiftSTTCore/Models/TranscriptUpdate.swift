import Foundation

/// A change to the running transcript.
///
/// Streaming transcription cannot always be additive. A window cut through
/// speech is decoded without the audio that follows it, and the model's
/// reading of those last few words is a guess it revises once it hears the
/// rest. Emitting the guess and never correcting it leaves the wrong word in
/// the transcript; holding it back until the next window confirms it delays
/// every word by a window. This type is how the guess is emitted at once and
/// corrected later.
public struct TranscriptUpdate: Sendable, Equatable {

    /// Segments previously emitted with ``TranscriptionSegment/start`` at or
    /// after this time are no longer part of the transcript, and `segments`
    /// replaces them. `nil` when nothing is withdrawn, which is the common
    /// case: only a window's final unconfirmed words are ever retracted.
    public let retractingFrom: TimeInterval?

    /// Segments to append once the retraction, if any, has been applied.
    public let segments: [TranscriptionSegment]

    /// Creates an update.
    public init(retractingFrom: TimeInterval? = nil, segments: [TranscriptionSegment]) {
        self.retractingFrom = retractingFrom
        self.segments = segments
    }

    /// Applies this update to a transcript held in order of ``TranscriptionSegment/start``.
    public func apply(to transcript: inout [TranscriptionSegment]) {
        if let retractingFrom {
            transcript.removeAll { $0.start >= retractingFrom }
        }
        transcript.append(contentsOf: segments)
    }
}
