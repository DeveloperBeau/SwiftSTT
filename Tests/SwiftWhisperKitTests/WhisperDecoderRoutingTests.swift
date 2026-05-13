@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

@Suite("WhisperDecoder routing")
struct WhisperDecoderRoutingTests {

    private let statelessInputNames: Set<String> = [
        "input_ids", "cache_length", "key_cache", "value_cache",
        "kv_cache_update_mask", "encoder_output_embeds", "decoder_key_padding_mask",
    ]

    private let statefulInputNames: Set<String> = [
        "decoder_input_ids", "cache_length", "encoder_output_embeds",
    ]

    @Test("stateless inputs select the stateless runner")
    func selectsStateless() throws {
        let kind = try WhisperDecoder.detectRunnerKind(inputNames: statelessInputNames)
        #expect(kind == .stateless)
    }

    @Test("stateful inputs select the stateful runner")
    func selectsStateful() throws {
        let kind = try WhisperDecoder.detectRunnerKind(inputNames: statefulInputNames)
        #expect(kind == .stateful)
    }

    @Test("ambiguous schema throws modelLoadFailed")
    func ambiguousThrows() {
        #expect(throws: SwiftWhisperError.self) {
            _ = try WhisperDecoder.detectRunnerKind(inputNames: ["cache_length"])
        }
    }
}
