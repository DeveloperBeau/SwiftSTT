@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Testing
@testable import SwiftWhisperKit

@Suite("StatelessDecoderRunner")
struct StatelessDecoderRunnerTests {

    private final class FakeModel: DecoderForwardModel, @unchecked Sendable {
        var lastFeatures: (any MLFeatureProvider)?
        var nextOutput: any MLFeatureProvider

        init() {
            self.nextOutput =
                (try? MLDictionaryFeatureProvider(dictionary: [:]))
                ?? MLDictionaryFeatureProvider()
        }

        func prediction(
            from features: any MLFeatureProvider
        ) async throws -> any MLFeatureProvider {
            lastFeatures = features
            return nextOutput
        }
    }

    private let allStatelessInputNames: Set<String> = [
        "input_ids", "cache_length", "key_cache", "value_cache",
        "kv_cache_update_mask", "encoder_output_embeds", "decoder_key_padding_mask",
    ]

    @Test("init succeeds when model has expected stateless inputs")
    func initAcceptsStatelessSchema() throws {
        let model = FakeModel()
        #expect(throws: Never.self) {
            _ = try StatelessDecoderRunner(
                model: model, inputNames: allStatelessInputNames, layerWidth: 1536, capacity: 224
            )
        }
    }

    @Test("init throws when key_cache input name is missing")
    func initRejectsMissingKeyCache() throws {
        let model = FakeModel()
        var incomplete = allStatelessInputNames
        incomplete.remove("key_cache")
        #expect(throws: SwiftWhisperError.self) {
            _ = try StatelessDecoderRunner(
                model: model, inputNames: incomplete, layerWidth: 1536, capacity: 224
            )
        }
    }

    /// Builds a feature provider that the runner receives from
    /// ``WhisperDecoder``, containing the three caller-supplied features.
    private func callerFeatures(
        tokenID: Int32,
        cacheLength: Int32
    ) throws -> any MLFeatureProvider {
        let tokenArray = try MLMultiArray(shape: [1], dataType: .int32)
        tokenArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = tokenID
        let lengthArray = try MLMultiArray(shape: [1], dataType: .int32)
        lengthArray.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = cacheLength
        let encoder = try MLMultiArray(shape: [1, 384, 1, 1500], dataType: .float16)
        return try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": tokenArray,
            "cache_length": lengthArray,
            "encoder_output_embeds": encoder,
        ])
    }

    /// Builds a model output containing logits + cache updates.
    private func modelOutput(
        logitsCount: Int,
        layerWidth: Int
    ) throws -> any MLFeatureProvider {
        let logits = try MLMultiArray(
            shape: [1, 1, NSNumber(value: logitsCount)],
            dataType: .float16
        )
        let keyUpdate = try MLMultiArray(
            shape: [1, NSNumber(value: layerWidth), 1, 1],
            dataType: .float16
        )
        let valueUpdate = try MLMultiArray(
            shape: [1, NSNumber(value: layerWidth), 1, 1],
            dataType: .float16
        )
        keyUpdate.dataPointer.bindMemory(to: UInt16.self, capacity: layerWidth)[0] = 0x3C00
        valueUpdate.dataPointer.bindMemory(to: UInt16.self, capacity: layerWidth)[0] = 0x4000
        return try MLDictionaryFeatureProvider(dictionary: [
            "logits": logits,
            "key_cache_updates": keyUpdate,
            "value_cache_updates": valueUpdate,
        ])
    }

    @Test("predict injects cache, masks, and reads logits from model output")
    func predictRoundtrip() async throws {
        let model = FakeModel()
        model.nextOutput = try modelOutput(logitsCount: 51_865, layerWidth: 4)
        let runner = try StatelessDecoderRunner(
            model: model, inputNames: allStatelessInputNames, layerWidth: 4, capacity: 8
        )
        await runner.resetState()

        let features = try callerFeatures(tokenID: 5, cacheLength: 0)
        let output = try await runner.predict(features: features)

        let logits = output.featureValue(for: "logits")?.multiArrayValue
        #expect(logits != nil)
        #expect(logits?.shape == [1, 1, 51_865])

        let received = try #require(model.lastFeatures)
        for name in StatelessDecoderInputs.required {
            #expect(received.featureValue(for: name) != nil, "missing \(name)")
        }
    }

    @Test("kv_cache_update_mask has 1.0 at current length and 0.0 elsewhere")
    func updateMaskShape() async throws {
        let model = FakeModel()
        model.nextOutput = try modelOutput(logitsCount: 100, layerWidth: 4)
        let runner = try StatelessDecoderRunner(
            model: model, inputNames: allStatelessInputNames, layerWidth: 4, capacity: 8
        )
        await runner.resetState()
        _ = try await runner.predict(features: try callerFeatures(tokenID: 1, cacheLength: 0))
        let received = try #require(model.lastFeatures)
        let mask = try #require(received.featureValue(for: "kv_cache_update_mask")?.multiArrayValue)
        let maskPtr = mask.dataPointer.bindMemory(to: UInt16.self, capacity: 8)
        #expect(maskPtr[0] == 0x3C00)  // Float16 1.0
        for i in 1..<8 {
            #expect(maskPtr[i] == 0x0000)
        }
    }

    @Test("decoder_key_padding_mask blocks positions beyond cache length")
    func paddingMaskShape() async throws {
        let model = FakeModel()
        model.nextOutput = try modelOutput(logitsCount: 100, layerWidth: 4)
        let runner = try StatelessDecoderRunner(
            model: model, inputNames: allStatelessInputNames, layerWidth: 4, capacity: 8
        )
        await runner.resetState()
        // Run two steps so internal cacheLength advances to 2.
        _ = try await runner.predict(features: try callerFeatures(tokenID: 1, cacheLength: 0))
        _ = try await runner.predict(features: try callerFeatures(tokenID: 2, cacheLength: 1))
        let received = try #require(model.lastFeatures)
        let padding = try #require(
            received.featureValue(for: "decoder_key_padding_mask")?.multiArrayValue
        )
        let ptr = padding.dataPointer.bindMemory(to: UInt16.self, capacity: 8)
        // Positions 0..<2 are 0.0 (attendable), 2..<8 are -inf (Float16 0xFC00).
        #expect(ptr[0] == 0x0000)
        #expect(ptr[1] == 0x0000)
        #expect(ptr[2] == 0xFC00)
        #expect(ptr[7] == 0xFC00)
    }

    @Test("predict repacks rank-2 input_ids to rank-1 for stateless decoder")
    func repackRank2TokenInput() async throws {
        let model = FakeModel()
        model.nextOutput = try modelOutput(logitsCount: 100, layerWidth: 4)
        let runner = try StatelessDecoderRunner(
            model: model, inputNames: allStatelessInputNames, layerWidth: 4, capacity: 8
        )
        await runner.resetState()

        // Simulate WhisperDecoder packing the token as rank-2 [1, 1] (legacy
        // stateful shape). The stateless runner must repack to rank-1 [1]
        // before feeding the model.
        let rank2Token = try MLMultiArray(shape: [1, 1], dataType: .int32)
        rank2Token.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = 42
        let lengthArr = try MLMultiArray(shape: [1], dataType: .int32)
        let encoder = try MLMultiArray(shape: [1, 384, 1, 1500], dataType: .float16)
        let features = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": rank2Token,
            "cache_length": lengthArr,
            "encoder_output_embeds": encoder,
        ])

        _ = try await runner.predict(features: features)

        let received = try #require(model.lastFeatures)
        let forwarded = try #require(
            received.featureValue(for: "input_ids")?.multiArrayValue
        )
        #expect(forwarded.shape == [1])
        #expect(forwarded.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] == 42)
    }
}
