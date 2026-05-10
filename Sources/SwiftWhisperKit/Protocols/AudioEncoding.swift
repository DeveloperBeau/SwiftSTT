import CoreML
import SwiftWhisperCore

/// Runs the Whisper audio encoder forward pass.
///
/// This protocol lives in `SwiftWhisperKit` rather than Core because its
/// signature mentions `MLMultiArray`, which is a Core ML type. Keeping Core free
/// of Apple framework imports lets the Core target run on Linux for testing,
/// which is occasionally useful for the pipeline's pure-logic parts.
///
/// The encoder takes a 30-second mel spectrogram (padded or trimmed to exactly
/// 3000 frames) and produces a fixed-size embedding tensor. For the small model
/// the output shape is `[1, 1500, 384]`; the base model is `[1, 1500, 512]`.
/// That tensor feeds straight into ``TokenDecoding``.
public protocol AudioEncoding: Actor {

    /// Encodes a mel spectrogram into an embedding tensor.
    ///
    /// - Parameter spectrogram: log-mel features in the layout that
    ///   `MelSpectrogramResult` documents. Caller is responsible for
    ///   padding or trimming to the model's expected frame count.
    /// - Returns: encoder output as a Core ML `MLMultiArray`, ready to feed
    ///   into the decoder.
    /// - Throws: ``SwiftWhisperError`` if Core ML inference fails.
    func encode(spectrogram: MelSpectrogramResult) async throws(SwiftWhisperError) -> MLMultiArray
}
