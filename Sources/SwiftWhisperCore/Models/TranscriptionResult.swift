import Foundation

public struct TranscriptionResult: Sendable, Equatable {
    public let text: String
    public let hypothesis: String
    public let isFinal: Bool
    public let language: String?
    public let segments: [TranscriptionSegment]

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
