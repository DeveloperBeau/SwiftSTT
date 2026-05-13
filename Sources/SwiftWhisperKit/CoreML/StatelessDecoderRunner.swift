@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Synchronization

/// Required input names for the argmaxinc stateless Whisper decoder.
enum StatelessDecoderInputs {
    static let inputIDs = "input_ids"
    static let cacheLength = "cache_length"
    static let keyCache = "key_cache"
    static let valueCache = "value_cache"
    static let kvCacheUpdateMask = "kv_cache_update_mask"
    static let encoderEmbeds = "encoder_output_embeds"
    static let decoderKeyPaddingMask = "decoder_key_padding_mask"

    static let required: [String] = [
        inputIDs, cacheLength, keyCache, valueCache,
        kvCacheUpdateMask, encoderEmbeds, decoderKeyPaddingMask,
    ]
}

/// Required output names for the same decoder.
enum StatelessDecoderOutputs {
    static let logits = "logits"
    static let keyCacheUpdates = "key_cache_updates"
    static let valueCacheUpdates = "value_cache_updates"
}

/// Runner for the stateless Whisper decoder.
///
/// The KV cache lives in two `MLMultiArray` buffers that are passed as inputs
/// and updated from the model's `key_cache_updates`/`value_cache_updates`
/// outputs each step.
///
/// Conforms to ``StatefulCoreMLModelRunner`` so callers (notably
/// ``WhisperDecoder``) do not have to know which decoder shape they are
/// dealing with.
final class StatelessDecoderRunner: StatefulCoreMLModelRunner, @unchecked Sendable {

    private let model: any DecoderForwardModel
    private let cache: Mutex<DecoderKVCache>

    /// Validates that the supplied input names cover every required feature.
    ///
    /// Then allocates the KV cache from the given shape parameters.
    init(
        model: any DecoderForwardModel,
        inputNames: Set<String>,
        layerWidth: Int,
        capacity: Int
    ) throws(SwiftWhisperError) {
        for name in StatelessDecoderInputs.required where !inputNames.contains(name) {
            throw .modelLoadFailed("stateless decoder missing required input '\(name)'")
        }
        self.model = model
        let kvCache = try DecoderKVCache(layerWidth: layerWidth, capacity: capacity)
        self.cache = Mutex<DecoderKVCache>(kvCache)
    }

    func resetState() async {
        cache.withLock { $0.reset() }
    }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        // Filled in by Task 4.
        throw .notImplemented
    }
}
