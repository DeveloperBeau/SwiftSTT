import Foundation
import SwiftWhisperCore

public actor AVAudioCapture: AudioCapturer {
    public nonisolated let audioStream: AsyncStream<AudioChunk>
    private let continuation: AsyncStream<AudioChunk>.Continuation

    public init() {
        let (stream, continuation) = AsyncStream<AudioChunk>.makeStream()
        self.audioStream = stream
        self.continuation = continuation
    }

    public func startCapture() async throws(SwiftWhisperError) {
        throw .notImplemented
    }

    public func stopCapture() async {
        continuation.finish()
    }
}
