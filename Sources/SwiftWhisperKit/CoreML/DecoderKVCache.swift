@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore

/// Swift-side KV cache for the stateless Whisper decoder.
///
/// Owns the `key_cache` and `value_cache` `MLMultiArray` buffers that get
/// re-fed to the model on every decoding step.
struct DecoderKVCache {

    /// Number of layer-channel features per token slot (e.g. 1536 for tiny).
    let layerWidth: Int

    /// Maximum number of tokens the cache holds (e.g. 224).
    let capacity: Int

    /// Cache buffer for key projections, shape `[1, layerWidth, 1, capacity]`.
    let key: MLMultiArray

    /// Cache buffer for value projections, shape `[1, layerWidth, 1, capacity]`.
    let value: MLMultiArray

    /// Number of tokens currently in the cache.
    var length: Int

    /// Allocates fresh, zeroed cache buffers.
    ///
    /// The shape parameters are read from the model's
    /// `inputDescriptionsByName` by the caller
    /// (`WhisperDecoder.init(model:tokenizer:)`); kept as plain `Int`s here
    /// so the struct is testable without real CoreML descriptions.
    init(
        layerWidth: Int,
        capacity: Int
    ) throws(SwiftWhisperError) {
        self.layerWidth = layerWidth
        self.capacity = capacity
        do {
            self.key = try MLMultiArray(
                shape: [1, NSNumber(value: layerWidth), 1, NSNumber(value: capacity)],
                dataType: .float16
            )
            self.value = try MLMultiArray(
                shape: [1, NSNumber(value: layerWidth), 1, NSNumber(value: capacity)],
                dataType: .float16
            )
        } catch {
            throw .modelLoadFailed("KV cache allocation: \(error.localizedDescription)")
        }
        self.length = 0
        zero(key)
        zero(value)
    }

    /// Discards the cached tokens and zeroes both buffers.
    mutating func reset() {
        length = 0
        zero(key)
        zero(value)
    }

    /// Copies a single-token update into the next free time slot of each cache.
    ///
    /// Advances ``length`` on success. Throws when the cache is full.
    mutating func splice(
        keyUpdate: MLMultiArray,
        valueUpdate: MLMultiArray
    ) throws(SwiftWhisperError) {
        guard length < capacity else {
            throw .decoderFailure("KV cache full at capacity \(capacity)")
        }
        let slot = length
        try writeSlice(keyUpdate, into: key, atTimeSlot: slot)
        try writeSlice(valueUpdate, into: value, atTimeSlot: slot)
        length += 1
    }

    private func writeSlice(
        _ source: MLMultiArray,
        into destination: MLMultiArray,
        atTimeSlot slot: Int
    ) throws(SwiftWhisperError) {
        // Source shape is [1, layerWidth, 1, 1]. Destination shape is
        // [1, layerWidth, 1, capacity]. Both are Float16.
        guard
            source.shape.count == 4,
            source.shape[1].intValue == layerWidth,
            source.shape[3].intValue == 1
        else {
            throw .decoderFailure("cache update shape mismatch: got \(source.shape)")
        }
        let srcPtr = source.dataPointer.bindMemory(to: UInt16.self, capacity: layerWidth)
        let dstPtr = destination.dataPointer.bindMemory(
            to: UInt16.self,
            capacity: layerWidth * capacity
        )
        let dstWidthStride = destination.strides[1].intValue
        let dstTimeStride = destination.strides[3].intValue
        for w in 0..<layerWidth {
            dstPtr[w * dstWidthStride + slot * dstTimeStride] = srcPtr[w]
        }
    }

    private func zero(_ array: MLMultiArray) {
        let count = array.count
        let ptr = array.dataPointer.bindMemory(to: UInt16.self, capacity: count)
        for i in 0..<count {
            ptr[i] = 0
        }
    }
}
