import Foundation

/// A snapshot of the streaming transcription's current state.
///
/// Real-time speech-to-text needs to expose two kinds of text at once:
///
/// - ``text`` is the part the system is confident about. The local-agreement
///   policy has confirmed these tokens by seeing them in two successive decodes,
///   so they will not change.
/// - ``hypothesis`` is the model's best guess for the audio that has come in
///   since the last confirmation. It can change between updates while the model
///   waits for more context.
///
/// Renderers usually display ``text`` in a stable colour and ``hypothesis`` in
/// a faded one, then merge the two when ``isFinal`` is `true`.
public struct TranscriptionResult: Sendable, Equatable {

    /// Confirmed text that is not expected to change.
    public let text: String

    /// Tentative text that may be revised in the next update.
    ///
    /// Empty once the segment is final.
    public let hypothesis: String

    /// `true` once the audio segment has ended (silence detected by VAD or stream
    /// closed) and ``hypothesis`` has been folded into ``text``.
    public let isFinal: Bool

    /// Detected language code (for example `"en"`, `"fr"`). `nil` when language
    /// detection has not run yet.
    public let language: String?

    /// Per-segment breakdown with timing information.
    ///
    /// Empty until the first segment has been finalised.
    public let segments: [TranscriptionSegment]

    /// Creates a new TranscriptionResult with the supplied values.
    public init(
        text: String,
        hypothesis: String = "",
        isFinal: Bool = false,
        language: String? = nil,
        segments: [TranscriptionSegment] = []
    ) {
        self.text = text
        self.hypothesis = hypothesis
        self.isFinal = isFinal
        self.language = language
        self.segments = segments
    }
}
