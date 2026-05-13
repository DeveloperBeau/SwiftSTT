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
}
