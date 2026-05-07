import Foundation

public protocol Transcriber: Actor {
    var resultStream: AsyncStream<TranscriptionResult> { get }
    func transcribe(audioStream: AsyncStream<AudioChunk>) async throws(SwiftWhisperError)
    func stop() async
}
