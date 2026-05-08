import CoreML
import SwiftWhisperCore

/// Runs the Whisper text decoder autoregressively until end-of-text.
///
/// The decoder is the most complex component in Whisper. It consumes the
/// encoder's embedding plus a growing sequence of previously generated tokens,
/// producing one new token per forward pass. Implementations are responsible for:
///
/// - Maintaining the KV cache across steps so the per-token cost stays linear.
/// - Applying logit filters (suppress blank, suppress non-speech, timestamp rules).
/// - Sampling the next token (greedy, beam search, or temperature sampling
///   based on ``DecodingOptions``).
/// - Stopping at the end-of-text token.
public protocol TokenDecoding: Actor {

    /// Decodes a full token sequence from the encoder output.
    ///
    /// - Parameters:
    ///   - encoderOutput: the embedding produced by ``AudioEncoding/encode(spectrogram:)``.
    ///   - options: sampling parameters and language hints.
    /// - Returns: tokens in generation order, ending with the end-of-text token
    ///   (which callers usually drop before display).
    func decode(
        encoderOutput: MLMultiArray,
        options: DecodingOptions
    ) async throws(SwiftWhisperError) -> [WhisperToken]
}
