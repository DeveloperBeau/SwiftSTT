import Foundation
import SwiftWhisperCore

/// Streams ``AudioChunk`` values from an injected ``AudioInputProvider``.
///
/// Defaults to ``AVMicrophoneInput`` (real microphone) but accepts any provider,
/// which is what makes the actor testable. The actor itself only handles the
/// state machine and the AsyncStream lifecycle; the heavy AVFoundation lifting
/// lives in ``AVMicrophoneInput``.
///
/// Cleanup runs automatically when the stream consumer cancels (via
/// `onTermination`) or when `stopCapture()` is called explicitly.
///
/// ## Example
///
/// ```swift
/// let capture = AVAudioCapture()
///
/// Task {
///     for await chunk in capture.audioStream {
///         print("got \(chunk.samples.count) samples at \(chunk.timestamp)s")
///     }
/// }
///
/// try await capture.startCapture()
/// // ... later
/// await capture.stopCapture()
/// ```
public actor AVAudioCapture: AudioCapturer {

    /// Async stream of resampled audio chunks.
    public nonisolated let audioStream: AsyncStream<AudioChunk>

    private nonisolated let continuation: AsyncStream<AudioChunk>.Continuation
    private let provider: any AudioInputProvider

    private let targetSampleRate: Double
    private let bufferDurationSeconds: Double
    private var isCapturing: Bool = false
    private var startTimestamp: TimeInterval = 0

    /// Creates a capture actor.
    ///
    /// - Parameters:
    ///   - provider: source of raw audio samples. Defaults to
    ///     ``AVMicrophoneInput`` (real microphone).
    ///   - targetSampleRate: rate to resample to. 16 000 Hz matches Whisper.
    ///   - bufferDurationSeconds: requested provider buffer length. The
    ///     default of 64 ms balances callback frequency against latency.
    public init(
        provider: any AudioInputProvider = AVMicrophoneInput(),
        targetSampleRate: Double = 16_000,
        bufferDurationSeconds: Double = 0.064
    ) {
        self.provider = provider
        self.targetSampleRate = targetSampleRate
        self.bufferDurationSeconds = bufferDurationSeconds

        let (stream, continuation) = AsyncStream<AudioChunk>.makeStream()
        self.audioStream = stream
        self.continuation = continuation

        let providerForCleanup = provider
        continuation.onTermination = { @Sendable _ in
            Task { await providerForCleanup.stop() }
        }
    }

    /// Asks the provider to start producing audio. Forwards each delivered
    /// buffer onto ``audioStream`` as an ``SwiftWhisperCore/AudioChunk``.
    ///
    /// Throws whatever the provider throws. For ``AVMicrophoneInput`` that
    /// includes ``SwiftWhisperCore/SwiftWhisperError/micPermissionDenied``,
    /// ``SwiftWhisperCore/SwiftWhisperError/audioConversionFailed``, and
    /// ``SwiftWhisperCore/SwiftWhisperError/audioCaptureFailed(_:)``.
    ///
    /// Calling more than once without ``stopCapture()`` is a no-op.
    public func startCapture() async throws(SwiftWhisperError) {
        guard !isCapturing else { return }

        let started = Date().timeIntervalSince1970
        startTimestamp = started
        let cont = continuation
        let target = targetSampleRate

        try await provider.start(
            targetSampleRate: targetSampleRate,
            bufferDurationSeconds: bufferDurationSeconds
        ) { @Sendable samples in
            let timestamp = Date().timeIntervalSince1970 - started
            cont.yield(
                AudioChunk(samples: samples, sampleRate: Int(target), timestamp: timestamp)
            )
        }
        isCapturing = true
    }

    /// Stops the provider and finishes ``audioStream``. Idempotent.
    public func stopCapture() async {
        guard isCapturing else { return }
        await provider.stop()
        isCapturing = false
        continuation.finish()
    }
}
