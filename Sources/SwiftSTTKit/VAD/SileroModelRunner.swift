@preconcurrency import CoreML
import Foundation
import SwiftSTTCore
import Synchronization

/// Abstracts a stateful Core ML forward pass for Silero VAD.
///
/// Silero VAD is a recurrent network; it reads and writes a `[2, 1, 64]`
/// state tensor on every window. ``resetState()`` re-zeroes that tensor
/// between audio streams; ``predict(features:)`` runs one window through
/// the model.
///
/// The production implementation is ``MLStateModelRunner``. Tests may
/// inject a mock that records calls and returns canned `MLFeatureProvider`
/// values.
public protocol StatefulCoreMLModelRunner: Sendable {

    /// Discards any existing state and prepares a fresh zero state for the
    /// next audio stream.
    func resetState() async

    /// Runs one forward pass against the current state, mutating the state
    /// tensor in-place.
    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftSTTError) -> any MLFeatureProvider
}

/// Production runner that delegates to a Core ML `MLModel` carrying an
/// `MLState` for the Silero VAD recurrent state.
///
/// Concurrency: `MLState` is not `Sendable`, so this type is
/// `@unchecked Sendable` and guards the state with a `Mutex`.
public final class MLStateModelRunner: StatefulCoreMLModelRunner, @unchecked Sendable {

    private let model: MLModel
    private let state: Mutex<MLState?>

    /// Creates a new runner wrapping `model`.
    public init(model: MLModel) {
        self.model = model
        self.state = Mutex<MLState?>(nil)
    }

    /// Allocates a fresh `MLState`, replacing any previous one.
    public func resetState() async {
        let newState = model.makeState()
        state.withLock { $0 = newState }
    }

    /// Runs prediction using the current state.
    public func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftSTTError) -> any MLFeatureProvider {
        let cached = state.withLock { $0 }
        guard let cached else {
            throw .decoderFailure(
                "SileroMLStateRunner: state not initialised; call resetState() first")
        }
        do {
            return try await model.prediction(from: features, using: cached)
        } catch {
            throw .decoderFailure("Silero VAD prediction: \(error.localizedDescription)")
        }
    }
}
