@preconcurrency import CoreML
import Foundation
import Synchronization
import SwiftWhisperCore

/// Abstracts a stateful Core ML forward pass for the Whisper decoder.
///
/// The Whisper decoder is autoregressive and reuses a key-value cache between
/// per-token forward passes. Core ML's `MLState` (iOS 18 / macOS 15) holds
/// that cache opaquely on behalf of the model. This protocol owns one state
/// per runner instance and exposes two operations: ``resetState()`` to
/// rebuild the cache between transcriptions, and ``predict(features:)`` to
/// run a single decoding step that mutates the cache.
///
/// The production implementation is ``MLStateModelRunner``, which wraps an
/// `MLModel`. Tests inject a mock that records calls and returns canned
/// `MLFeatureProvider` values.
public protocol StatefulCoreMLModelRunner: Sendable {

    /// Discards any existing cache and prepares a fresh state for the next
    /// decoding session.
    func resetState() async

    /// Runs one forward pass against the current state, mutating the cache.
    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider
}

/// A `StatefulCoreMLModelRunner` that can spawn an independent runner sharing
/// the same underlying model but holding its own fresh KV cache. Used by beam
/// search so each beam can advance its cache independently instead of
/// re-prefilling on every step.
///
/// Apple does not expose `MLState` copy, so a freshly branched runner starts
/// with an empty cache. Callers that need to pick up where the source runner
/// left off must replay the relevant prefix into the branched runner.
public protocol BranchableStatefulRunner: StatefulCoreMLModelRunner {

    /// Returns a new runner that shares the same model but holds an
    /// independent, freshly-allocated state. Mutations on the parent runner do
    /// not affect the branch and vice versa.
    func branch() async throws(SwiftWhisperError) -> any BranchableStatefulRunner
}

/// Production runner that delegates to a Core ML `MLModel` carrying an
/// `MLState` for its KV cache.
///
/// Concurrency: `MLState` is not `Sendable`, so the type is
/// `@unchecked Sendable` and uses `Mutex` to guard the cache. The production
/// pattern matches `MLModelRunner` in spirit, with the addition that the
/// state must be initialised via ``resetState()`` before the first
/// ``predict(features:)`` call.
public final class MLStateModelRunner: BranchableStatefulRunner, @unchecked Sendable {

    private let model: MLModel
    private let state: Mutex<MLState?>

    public init(model: MLModel) {
        self.model = model
        self.state = Mutex<MLState?>(nil)
    }

    public func branch() async throws(SwiftWhisperError) -> any BranchableStatefulRunner {
        let clone = MLStateModelRunner(model: model)
        await clone.resetState()
        return clone
    }

    public func resetState() async {
        // Build the state outside the lock so the (potentially expensive)
        // allocation does not stall other actors waiting on the mutex.
        let newState = model.makeState()
        state.withLock { $0 = newState }
    }

    public func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        let cached = state.withLock { $0 }
        guard let cached else {
            throw .decoderFailure("MLStateModelRunner: state not initialised; call resetState() first")
        }
        do {
            return try await model.prediction(from: features, using: cached)
        } catch {
            throw .decoderFailure("decoder prediction: \(error.localizedDescription)")
        }
    }
}
