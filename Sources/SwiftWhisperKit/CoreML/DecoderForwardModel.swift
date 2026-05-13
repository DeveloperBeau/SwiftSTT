@preconcurrency import CoreML
import Foundation

/// Internal seam that lets ``StatelessDecoderRunner`` be tested against a
/// scripted forward pass without instantiating a real Core ML model.
///
/// Apple's `MLModel` is a concrete class with no public init, so we wrap the
/// single method we need behind a protocol.
protocol DecoderForwardModel: Sendable {

    /// Runs one forward pass and returns the produced feature provider.
    func prediction(
        from features: any MLFeatureProvider
    ) async throws -> any MLFeatureProvider
}

extension MLModel: @retroactive @unchecked Sendable {}

extension MLModel: DecoderForwardModel {

    func prediction(
        from features: any MLFeatureProvider
    ) async throws -> any MLFeatureProvider {
        try await self.prediction(from: features, options: MLPredictionOptions())
    }
}
