import Foundation

/// A short slice of PCM audio flowing through the transcription pipeline.
///
/// The pipeline standardises on 16 kHz mono Float32 because that is the rate Whisper
/// models expect. Capture sources that record at 44.1 or 48 kHz should resample before
/// emitting chunks, which is what `AVAudioCapture` in `SwiftWhisperKit` does.
///
/// Chunks are deliberately small (typically 1024 samples, around 64 ms) so that the
/// downstream voice activity detector and mel spectrogram can react with low latency.
///
/// ## Example
///
/// ```swift
/// let chunk = AudioChunk(
///     samples: [0.0, 0.1, -0.05, 0.2],
///     sampleRate: 16_000,
///     timestamp: 0.064
/// )
/// ```
public struct AudioChunk: Sendable, Equatable {

    /// PCM samples in the range `-1.0 ... 1.0`.
    ///
    /// Length is not fixed but should stay small enough to keep streaming latency bounded.
    public let samples: [Float]

    /// Sample rate in Hz. The pipeline assumes 16 000 throughout; other values are
    /// accepted on the model so callers can tag pre-resampled audio without losing
    /// the original rate.
    public let sampleRate: Int

    /// Wall-clock offset of the first sample in this chunk, measured from the moment
    /// capture started.
    ///
    /// Used to align tokens with their original audio position.
    public let timestamp: TimeInterval

    /// Creates a new AudioChunk with the supplied values.
    public init(samples: [Float], sampleRate: Int = 16_000, timestamp: TimeInterval) {
        self.samples = samples
        self.sampleRate = sampleRate
        self.timestamp = timestamp
    }
}
