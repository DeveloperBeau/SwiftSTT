import Foundation
import SwiftWhisperCore

public actor EnergyVAD: VoiceActivityDetector {
    public init(thresholdDB: Float = -40.0, hysteresisFrames: Int = 5) {}

    public func isSpeech(chunk: AudioChunk) async -> Bool {
        false
    }

    public func reset() async {}
}
