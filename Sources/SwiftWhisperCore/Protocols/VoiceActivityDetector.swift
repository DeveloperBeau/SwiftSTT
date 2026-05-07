import Foundation

public protocol VoiceActivityDetector: Actor {
    func isSpeech(chunk: AudioChunk) async -> Bool
    func reset() async
}
