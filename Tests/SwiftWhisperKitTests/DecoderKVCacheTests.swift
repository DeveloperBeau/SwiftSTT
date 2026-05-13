@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Testing
@testable import SwiftWhisperKit

@Suite("DecoderKVCache")
struct DecoderKVCacheTests {

    @Test("init allocates correct shape")
    func initReadsShape() throws {
        let cache = try DecoderKVCache(layerWidth: 1536, capacity: 224)
        #expect(cache.layerWidth == 1536)
        #expect(cache.capacity == 224)
        #expect(cache.length == 0)
        #expect(cache.key.shape == [1, 1536, 1, 224])
        #expect(cache.value.shape == [1, 1536, 1, 224])
    }

    @Test("reset zeroes buffers and length")
    func resetZeroes() throws {
        var cache = try DecoderKVCache(layerWidth: 4, capacity: 6)
        // Write garbage then reset.
        let kPtr = cache.key.dataPointer.bindMemory(to: UInt16.self, capacity: 24)
        kPtr[0] = 0xFFFF
        cache.length = 7
        cache.reset()
        #expect(cache.length == 0)
        let zeroed = cache.key.dataPointer.bindMemory(to: UInt16.self, capacity: 24)
        #expect(zeroed[0] == 0)
    }

    @Test("splice writes update at current length offset")
    func spliceWrites() throws {
        var cache = try DecoderKVCache(layerWidth: 2, capacity: 3)
        let keyUpdate = try MLMultiArray(shape: [1, 2, 1, 1], dataType: .float16)
        let valueUpdate = try MLMultiArray(shape: [1, 2, 1, 1], dataType: .float16)
        let keyPtr = keyUpdate.dataPointer.bindMemory(to: UInt16.self, capacity: 2)
        let valuePtr = valueUpdate.dataPointer.bindMemory(to: UInt16.self, capacity: 2)
        keyPtr[0] = 0x3C00  // Float16 1.0
        keyPtr[1] = 0x4000  // Float16 2.0
        valuePtr[0] = 0x4200  // 3.0
        valuePtr[1] = 0x4400  // 4.0

        try cache.splice(keyUpdate: keyUpdate, valueUpdate: valueUpdate)
        #expect(cache.length == 1)

        // Verify cache[0, :, 0, 0] holds the update.
        let cachedKey = cache.key.dataPointer.bindMemory(to: UInt16.self, capacity: 6)
        let widthStride = cache.key.strides[1].intValue
        // First time slot is index 0; layer dim 0 -> address 0
        #expect(cachedKey[0] == 0x3C00)
        // Layer dim 1 sits at widthStride
        #expect(cachedKey[widthStride] == 0x4000)
    }

    @Test("splice throws when cache is full")
    func spliceOverflow() throws {
        var cache = try DecoderKVCache(layerWidth: 2, capacity: 1)
        let keyUpdate = try MLMultiArray(shape: [1, 2, 1, 1], dataType: .float16)
        let valueUpdate = try MLMultiArray(shape: [1, 2, 1, 1], dataType: .float16)
        try cache.splice(keyUpdate: keyUpdate, valueUpdate: valueUpdate)
        #expect(throws: SwiftWhisperError.self) {
            try cache.splice(keyUpdate: keyUpdate, valueUpdate: valueUpdate)
        }
    }
}
