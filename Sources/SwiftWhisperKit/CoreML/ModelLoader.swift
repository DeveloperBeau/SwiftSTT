import CoreML
import Foundation
import SwiftWhisperCore

/// Loads compiled Core ML Whisper models from disk.
///
/// Whisper ships as a pair of models: an audio encoder (`AudioEncoder.mlmodelc`)
/// and a text decoder (`TextDecoder.mlmodelc`). This loader handles both. Models
/// load with `MLModelConfiguration.computeUnits = .all` so the Apple Neural
/// Engine takes the encoder where available.
///
/// ## First-run latency
///
/// The Neural Engine compiles `.mlmodelc` packages on first use per device.
/// For the small model this takes 30 to 60 seconds; the large turbo model can
/// take two minutes. Surface this to the user with a progress indicator on
/// first launch. Subsequent loads are instant because the compiled artefact is
/// cached by the OS.
///
/// > Important: Stub. Real loader lands in milestone M4.
public actor ModelLoader {
    public init() {}

    /// Loads a compiled audio encoder model.
    ///
    /// - Parameter url: file URL to an `AudioEncoder.mlmodelc` directory.
    /// - Returns: the compiled `MLModel`, ready to wire into ``WhisperEncoder``.
    public func loadEncoder(at url: URL) async throws(SwiftWhisperError) -> MLModel {
        throw .notImplemented
    }

    /// Loads a compiled text decoder model.
    ///
    /// - Parameter url: file URL to a `TextDecoder.mlmodelc` directory.
    /// - Returns: the compiled `MLModel`, ready to wire into ``WhisperDecoder``.
    public func loadDecoder(at url: URL) async throws(SwiftWhisperError) -> MLModel {
        throw .notImplemented
    }
}
