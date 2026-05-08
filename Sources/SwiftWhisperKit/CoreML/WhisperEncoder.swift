import CoreML
import SwiftWhisperCore

/// Core ML implementation of the Whisper audio encoder.
///
/// Wraps an `MLModel` loaded by ``ModelLoader``. Each forward pass:
///
/// 1. Pads or trims the input mel spectrogram to exactly 3000 frames (30 s).
/// 2. Packs the mel features into an `MLMultiArray` with the model's expected
///    shape (`[1, nMels, 3000]` for the standard models).
/// 3. Runs `model.prediction(from:)`, which executes on the Neural Engine when
///    available.
/// 4. Returns the encoder embedding tensor.
///
/// Encode latency on Apple Silicon is roughly 50 ms for the base model on the
/// ANE, climbing to 300 ms for the large turbo model. Real-time factor on
/// recent hardware sits at 0.05 to 0.1, so encoding rarely dominates the
/// pipeline cost.
///
/// > Important: Stub. Real implementation lands in milestone M4.
public actor WhisperEncoder: AudioEncoding {
    public init() {}

    public func encode(spectrogram: MelSpectrogramResult) async throws(SwiftWhisperError) -> MLMultiArray {
        throw .notImplemented
    }
}
