@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore

/// Loads compiled Core ML Whisper models from disk into memory.
///
/// The encoder runs on the Neural Engine when available (compute units = `.all`).
/// First-run ANE compilation can take 30 to 120 seconds for large models; the
/// OS caches the result for subsequent loads.
public actor ModelLoader {

    /// Creates a new ModelLoader with the supplied values.
    public init() {}

    /// Loads a compiled audio encoder model from an `AudioEncoder.mlmodelc` directory.
    public func loadEncoder(at url: URL) async throws(SwiftWhisperError) -> MLModel {
        try await loadModel(at: url, label: "encoder")
    }

    /// Loads a compiled text decoder model from a `TextDecoder.mlmodelc` directory.
    public func loadDecoder(at url: URL) async throws(SwiftWhisperError) -> MLModel {
        try await loadModel(at: url, label: "decoder")
    }

    /// Loads both encoder and decoder from a ``ModelBundle`` sequentially.
    public func loadBundle(_ bundle: ModelBundle) async throws(SwiftWhisperError) -> LoadedModels {
        let encoder = try await loadEncoder(at: bundle.encoderURL)
        let decoder = try await loadDecoder(at: bundle.decoderURL)
        return LoadedModels(encoder: encoder, decoder: decoder, tokenizerURL: bundle.tokenizerURL)
    }

    private func loadModel(at url: URL, label: String) async throws(SwiftWhisperError) -> MLModel {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw .modelLoadFailed("\(label) not found at \(url.path)")
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        do {
            return try await MLModel.load(contentsOf: url, configuration: config)
        } catch {
            throw .modelLoadFailed("\(label): \(error.localizedDescription)")
        }
    }
}

/// Loaded encoder and decoder pair, ready to wire into the pipeline.
public struct LoadedModels {
    /// URL of the compiled encoder `.mlmodelc`.
    public let encoder: MLModel
    /// URL of the compiled decoder `.mlmodelc`.
    public let decoder: MLModel
    /// URL of the tokenizer JSON file.
    public let tokenizerURL: URL
}
