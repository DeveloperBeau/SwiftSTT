import Foundation
import SwiftWhisperCore

public actor SileroVAD: VoiceActivityDetector {
    public init() {}

    public func isSpeech(chunk: AudioChunk) async -> Bool {
        false
    }

    public func reset() async {}
}
