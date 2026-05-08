import Foundation

/// Anything that produces a stream of ``AudioChunk`` values.
///
/// The concrete implementation in ``SwiftWhisperKit/AVAudioCapture`` reads from the
/// microphone, but tests and offline tools can supply other implementations that
/// read from files or replay fixtures. The protocol is constrained to `Actor` so
/// that capture state (running flag, engine reference, observers) can never be
/// mutated from two contexts at once.
///
/// The expected lifecycle is:
///
/// ```swift
/// let capture: any AudioCapturer = AVAudioCapture()
///
/// Task {
///     for await chunk in await capture.audioStream {
///         // forward to VAD or mel pipeline
///     }
/// }
///
/// try await capture.startCapture()
/// // ... later
/// await capture.stopCapture()
/// ```
///
/// Implementations should call `continuation.finish()` when the source ends so
/// consumers see the stream terminate cleanly.
public protocol AudioCapturer: Actor {

    /// Async stream of audio chunks. Available immediately after `init`; chunks
    /// only start arriving once ``startCapture()`` has been called and granted.
    var audioStream: AsyncStream<AudioChunk> { get }

    /// Starts the capture source. Throws ``SwiftWhisperError/micPermissionDenied``
    /// if the user has not granted microphone access, or
    /// ``SwiftWhisperError/audioCaptureFailed(_:)`` if the underlying engine
    /// could not be started.
    func startCapture() async throws(SwiftWhisperError)

    /// Stops the capture source, removes any installed taps, and finishes the stream.
    /// Idempotent: calling more than once has no extra effect.
    func stopCapture() async
}
