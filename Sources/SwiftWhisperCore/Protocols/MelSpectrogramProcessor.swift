import Foundation

public protocol MelSpectrogramProcessor: Actor {
    func process(chunk: AudioChunk) async throws(SwiftWhisperError) -> MelSpectrogramResult
    func reset() async
}
