import Foundation

public struct WhisperToken: Sendable, Equatable {
    public let id: Int
    public let text: String
    public let probability: Float
    public let timestamp: TimeInterval?

    public init(id: Int, text: String, probability: Float = 1.0, timestamp: TimeInterval? = nil) {
        self.id = id
        self.text = text
        self.probability = probability
        self.timestamp = timestamp
    }
}
