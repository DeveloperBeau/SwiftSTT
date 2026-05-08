import Foundation

/// Output of a mel-scale spectrogram computation.
///
/// Whisper expects log-mel features arranged as a `[nMels x nFrames]` matrix. To keep
/// allocation simple and to match Core ML `MLMultiArray` row-major layout, the values
/// are stored in a single flat ``frames`` buffer. To read the value at mel band `m`
/// and time frame `t`, use `frames[m * nFrames + t]`.
///
/// Two factors fix the dimensions in practice:
///
/// - `nMels` is 80 for the small and base Whisper models, 128 for large.
/// - `nFrames` depends on input length. With a 400-sample frame and 160-sample hop
///   at 16 kHz, one second of audio produces 98 frames.
///
/// Values come out normalised by the same formula Whisper uses
/// (`(max(log_mel, log_mel.max() - 8.0) + 4.0) / 4.0`), so the spread is exactly 2.0
/// regardless of input loudness.
public struct MelSpectrogramResult: Sendable, Equatable {

    /// Flat row-major buffer of length `nMels * nFrames`. Index with
    /// `frames[m * nFrames + t]` to read mel band `m` at frame `t`.
    public let frames: [Float]

    /// Number of mel filterbank bands. 80 by default, 128 for the large model.
    public let nMels: Int

    /// Number of time frames. Equal to `(samples - frameLength) / hopLength + 1`
    /// once enough audio has accumulated.
    public let nFrames: Int

    /// Throws ``SwiftWhisperError/invalidMelDimensions(framesCount:expected:)``
    /// if `frames.count != nMels * nFrames`.
    public init(frames: [Float], nMels: Int, nFrames: Int) throws(SwiftWhisperError) {
        let expected = nMels * nFrames
        guard frames.count == expected else {
            throw .invalidMelDimensions(framesCount: frames.count, expected: expected)
        }
        self.frames = frames
        self.nMels = nMels
        self.nFrames = nFrames
    }
}
