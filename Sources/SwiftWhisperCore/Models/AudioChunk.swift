import Foundation

public struct AudioChunk: Sendable, Equatable {
    public let samples: [Float]
    public let sampleRate: Int
    public let timestamp: TimeInterval

    public init(samples: [Float], sampleRate: Int = 16_000, timestamp: TimeInterval) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.timestamp = timestamp
    }
}
