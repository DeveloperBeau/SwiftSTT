import CoreML
import SwiftWhisperCore

/// Core ML implementation of the Whisper text decoder.
///
/// The decoder is autoregressive: each forward pass produces logits over the
/// vocabulary for one new token, conditioned on the encoder embedding plus all
/// previously generated tokens. To keep per-token cost flat instead of growing
/// quadratically, the implementation maintains a key-value cache via Core ML's
/// `MLState` (iOS 18 / macOS 15 minimum).
///
/// ## Sampling
///
/// Two strategies live here, picked based on `DecodingOptions`:
///
/// - **Greedy** (`temperature == 0`, `beamSize == 1`): always pick `argmax`.
///   Fastest. Good enough for clean speech.
/// - **Beam search** (`beamSize > 1`): keep `beamSize` candidate sequences and
///   score by cumulative log-probability. Better accuracy on hard audio at the
///   cost of `beamSize` times more model invocations.
///
/// Logit filters (suppress blank, suppress non-speech tokens, timestamp rules)
/// run before sampling, in the order documented in the OpenAI Whisper paper.
///
/// > Important: Stub. Real implementation lands in milestone M5.
public actor WhisperDecoder: TokenDecoding {
    public init() {}

    public func decode(
        encoderOutput: MLMultiArray,
        options: DecodingOptions
    ) async throws(SwiftWhisperError) -> [WhisperToken] {
        throw .notImplemented
    }
}
