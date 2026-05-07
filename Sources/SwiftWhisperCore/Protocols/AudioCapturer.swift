import Foundation

public protocol AudioCapturer: Actor {
    var audioStream: AsyncStream<AudioChunk> { get }
    func startCapture() async throws(SwiftWhisperError)
    func stopCapture() async
}
