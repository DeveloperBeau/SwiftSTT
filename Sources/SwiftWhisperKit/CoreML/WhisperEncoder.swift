@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore

/// Core ML implementation of the Whisper audio encoder.
///
/// Each forward pass:
///
/// 1. Pads or trims the input mel spectrogram to exactly 3000 frames (30 s).
/// 2. Packs the mel features into an `MLMultiArray` with shape
///    `[1, nMels, 3000]` under the feature name `melspectrogram_features`,
///    matching the argmaxinc-converted models on HuggingFace.
/// 3. Runs the model via the injected ``CoreMLModelRunner``.
/// 4. Extracts the encoder embedding tensor from the output feature
///    `encoder_output_embeds`.
///
/// Encode latency on Apple Silicon is roughly 50 ms for the base model on the
/// ANE, climbing to 300 ms for the large turbo model.
public actor WhisperEncoder: AudioEncoding {

    /// Number of mel frames the encoder expects per call.
    public static let expectedFrames: Int = 3_000
    /// Name of the mel-spectrogram input feature.
    public static let inputFeatureName: String = "melspectrogram_features"
    /// Name of the encoder-output feature.
    public static let outputFeatureName: String = "encoder_output_embeds"

    private let runner: any CoreMLModelRunner

    /// Creates a new WhisperEncoder with the supplied values.
    public init(runner: any CoreMLModelRunner) {
        self.runner = runner
    }

    /// Encodes the input.
    public func encode(spectrogram: MelSpectrogramResult) async throws(SwiftWhisperError)
        -> MLMultiArray
    {
        let padded = try Self.padOrTrim(spectrogram, toFrames: Self.expectedFrames)
        let input = try Self.buildFeatureProvider(mel: padded)
        let output = try await runner.predict(features: input)
        return try Self.extractEmbeddings(from: output)
    }

    // MARK: - Pure helpers (visible to tests)

    /// Pads with zeros or trims to exactly `frames` time steps.
    ///
    /// Layout assumes row-major `[nMels x nFrames]` per ``MelSpectrogramResult``.
    static func padOrTrim(
        _ spectrogram: MelSpectrogramResult,
        toFrames frames: Int
    ) throws(SwiftWhisperError) -> MelSpectrogramResult {
        let nMels = spectrogram.nMels
        let currentFrames = spectrogram.nFrames

        if currentFrames == frames {
            return spectrogram
        }

        var out = [Float](repeating: 0, count: nMels * frames)
        let copyFrames = min(currentFrames, frames)
        for m in 0..<nMels {
            for t in 0..<copyFrames {
                out[m * frames + t] = spectrogram.frames[m * currentFrames + t]
            }
        }
        return try MelSpectrogramResult(frames: out, nMels: nMels, nFrames: frames)
    }

    /// Packs the mel buffer into an `MLMultiArray` for the encoder.
    ///
    /// Uses shape `[1, nMels, 1, nFrames]` and wraps it as a feature provider.
    /// The argmaxinc Core ML conversions of Whisper expect a rank-4 input with
    /// a singleton "channel" dimension between nMels and nFrames. The flat
    /// memory layout is unchanged versus rank 3 because the inserted axis
    /// has length 1.
    static func buildFeatureProvider(
        mel: MelSpectrogramResult
    ) throws(SwiftWhisperError) -> any MLFeatureProvider {
        let array: MLMultiArray
        do {
            array = try MLMultiArray(
                shape: [1, NSNumber(value: mel.nMels), 1, NSNumber(value: mel.nFrames)],
                dataType: .float32
            )
        } catch {
            throw .modelLoadFailed("MLMultiArray init: \(error.localizedDescription)")
        }
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: mel.frames.count)
        for i in 0..<mel.frames.count {
            pointer[i] = mel.frames[i]
        }
        do {
            return try MLDictionaryFeatureProvider(dictionary: [
                inputFeatureName: array
            ])
        } catch {
            throw .modelLoadFailed("feature provider: \(error.localizedDescription)")
        }
    }

    /// Pulls the encoder output tensor out of the model's response.
    static func extractEmbeddings(
        from output: any MLFeatureProvider
    ) throws(SwiftWhisperError) -> MLMultiArray {
        guard let value = output.featureValue(for: outputFeatureName) else {
            throw .decoderFailure("missing feature '\(outputFeatureName)' in encoder output")
        }
        guard let array = value.multiArrayValue else {
            throw .decoderFailure("feature '\(outputFeatureName)' is not an MLMultiArray")
        }
        return array
    }
}
