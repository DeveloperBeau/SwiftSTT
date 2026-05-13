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
    private let updateMaskBuffer: MLMultiArray
    private let paddingMaskBuffer: MLMultiArray

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
        do {
            self.updateMaskBuffer = try MLMultiArray(
                shape: [1, NSNumber(value: capacity)],
                dataType: .float16
            )
            self.paddingMaskBuffer = try MLMultiArray(
                shape: [1, NSNumber(value: capacity)],
                dataType: .float16
            )
        } catch {
            throw .modelLoadFailed("mask buffer allocation: \(error.localizedDescription)")
        }
    }

    func resetState() async {
        cache.withLock { $0.reset() }
    }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        let tokenIDs = try requireMultiArray(features, name: StatelessDecoderInputs.inputIDs)
        let cacheLengthArr = try requireMultiArray(
            features,
            name: StatelessDecoderInputs.cacheLength
        )
        let encoder = try requireMultiArray(
            features,
            name: StatelessDecoderInputs.encoderEmbeds
        )

        let (keyCache, valueCache, currentLength, capacity) = cache.withLock {
            ($0.key, $0.value, $0.length, $0.capacity)
        }

        fillUpdateMask(currentLength: currentLength, capacity: capacity)
        fillPaddingMask(currentLength: currentLength, capacity: capacity)

        let provider: any MLFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(dictionary: [
                StatelessDecoderInputs.inputIDs: tokenIDs,
                StatelessDecoderInputs.cacheLength: cacheLengthArr,
                StatelessDecoderInputs.encoderEmbeds: encoder,
                StatelessDecoderInputs.keyCache: keyCache,
                StatelessDecoderInputs.valueCache: valueCache,
                StatelessDecoderInputs.kvCacheUpdateMask: updateMaskBuffer,
                StatelessDecoderInputs.decoderKeyPaddingMask: paddingMaskBuffer,
            ])
        } catch {
            throw .decoderFailure("feature provider: \(error.localizedDescription)")
        }

        let output: any MLFeatureProvider
        do {
            output = try await model.prediction(from: provider)
        } catch {
            throw .decoderFailure("model prediction: \(error.localizedDescription)")
        }

        let keyUpdate = try requireOutputArray(
            output,
            name: StatelessDecoderOutputs.keyCacheUpdates
        )
        let valueUpdate = try requireOutputArray(
            output,
            name: StatelessDecoderOutputs.valueCacheUpdates
        )
        do {
            try cache.withLock { cache in
                try cache.splice(keyUpdate: keyUpdate, valueUpdate: valueUpdate)
            }
        } catch let error as SwiftWhisperError {
            throw error
        } catch {
            throw .decoderFailure("cache splice: \(error.localizedDescription)")
        }

        return output
    }

    private func requireMultiArray(
        _ provider: any MLFeatureProvider,
        name: String
    ) throws(SwiftWhisperError) -> MLMultiArray {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw .decoderFailure("missing input feature '\(name)'")
        }
        return array
    }

    private func requireOutputArray(
        _ provider: any MLFeatureProvider,
        name: String
    ) throws(SwiftWhisperError) -> MLMultiArray {
        guard let array = provider.featureValue(for: name)?.multiArrayValue else {
            throw .decoderFailure("missing output feature '\(name)'")
        }
        return array
    }

    private func fillUpdateMask(currentLength: Int, capacity: Int) {
        let ptr = updateMaskBuffer.dataPointer.bindMemory(to: UInt16.self, capacity: capacity)
        for i in 0..<capacity {
            ptr[i] = (i == currentLength) ? 0x3C00 : 0x0000  // 1.0 / 0.0
        }
    }

    private func fillPaddingMask(currentLength: Int, capacity: Int) {
        let ptr = paddingMaskBuffer.dataPointer.bindMemory(to: UInt16.self, capacity: capacity)
        let attendable = currentLength + 1
        for i in 0..<capacity {
            ptr[i] = (i < attendable) ? 0x0000 : 0xFC00  // 0.0 / -inf in Float16
        }
    }
}
