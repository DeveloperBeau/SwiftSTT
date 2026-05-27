import Foundation

/// Source of raw mono Float32 audio sample buffers, resampled to a requested rate.
///
/// Sits one layer below ``AudioCapturer`` and is what makes ``AudioCapturer``
/// implementations testable. The default production provider wraps
/// `AVAudioEngine` with format conversion. Tests can supply a fake provider
/// that synchronously delivers a fixed sequence of buffers, which lets the
/// capturer's lifecycle and stream semantics be exercised without a microphone.
public protocol AudioInputProvider: Sendable {

    /// Starts the input. Calls `onChunk` for each captured buffer once it has
    /// been resampled to `targetSampleRate` mono Float32.
    ///
    /// - Parameters:
    ///   - targetSampleRate: rate to resample audio to (Hz).
    ///   - bufferDurationSeconds: requested buffer length. Implementations
    ///     may round to the nearest hardware-friendly size.
    ///   - onChunk: handler called on a Sendable executor (typically a real-
    ///     time audio thread for the production provider). Must not block.
    /// - Throws: ``SwiftSTTError`` if the input cannot be started or audio
    ///   format conversion fails.
    func start(
        targetSampleRate: Double,
        bufferDurationSeconds: Double,
        onChunk: @Sendable @escaping ([Float]) -> Void
    ) async throws(SwiftSTTError)

    /// Stops the input and releases its resources. Idempotent.
    func stop() async
}
