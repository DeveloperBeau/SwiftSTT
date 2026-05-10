import Foundation

/// Converts raw audio chunks into the log-mel spectrogram features Whisper expects.
///
/// The Whisper encoder takes 80-band (or 128-band on the large model) log-mel
/// features at 100 frames per second. This protocol is the boundary between the
/// audio side of the pipeline and the model side.
///
/// Implementations must carry over partial frames between calls. Audio chunks
/// are typically 1024 samples but the spectrogram works on 400-sample frames
/// with a 160-sample hop, so chunk boundaries almost never line up with frame
/// boundaries. Without carry-over, samples at the seams would be silently
/// dropped, which would shift every downstream timestamp.
public protocol MelSpectrogramProcessor: Actor {

    /// Computes mel features for the given chunk plus any leftover samples from
    /// the previous call. Throws if the input format is invalid or DSP setup fails.
    func process(chunk: AudioChunk) async throws(SwiftWhisperError) -> MelSpectrogramResult

    /// Drops any carried-over samples. Call after a stream restart so the next
    /// frame starts at sample zero rather than at an offset from the previous run.
    func reset() async

    /// Number of mel frames currently held in the rolling buffer.
    func currentFrameCount() async -> Int

    /// Returns a flattened copy of the rolling buffer in `[nMels x nFrames]`
    /// row-major layout. Throws if the rolling buffer's bookkeeping is
    /// inconsistent (defensive: should be unreachable on a correctly-implemented
    /// processor).
    func snapshot() async throws(SwiftWhisperError) -> MelSpectrogramResult

    /// Drops the first `framesConsumed` columns from the rolling buffer. Used
    /// by the pipeline to slide the encoder window after a successful decode.
    func advance(framesConsumed: Int) async
}
