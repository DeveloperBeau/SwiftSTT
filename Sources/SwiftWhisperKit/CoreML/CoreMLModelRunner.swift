@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore

/// Abstracts a single Core ML forward pass so the encoder and decoder can be
/// tested without instantiating a real `MLModel`.
///
/// The production implementation is ``MLModelRunner``, which wraps an
/// `MLModel` and forwards to `prediction(from:)`. Tests inject a mock that
/// returns canned `MLFeatureProvider` values.
public protocol CoreMLModelRunner: Sendable {

    /// Runs one forward pass.
    func predict(
        features: MLFeatureProvider
    ) async throws(SwiftWhisperError) -> MLFeatureProvider
}

/// Production runner that delegates to a Core ML `MLModel`.
public struct MLModelRunner: CoreMLModelRunner {

    private let model: MLModel

    public init(model: MLModel) {
        self.model = model
    }

    public func predict(
        features: MLFeatureProvider
    ) async throws(SwiftWhisperError) -> MLFeatureProvider {
        do {
            return try await model.prediction(from: features)
        } catch {
            throw .decoderFailure("prediction: \(error.localizedDescription)")
        }
    }
}
