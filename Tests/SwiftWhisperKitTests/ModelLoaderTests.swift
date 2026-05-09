@preconcurrency import CoreML
import Foundation
import Testing
@testable import SwiftWhisperKit
import SwiftWhisperCore

@Suite("ModelLoader")
struct ModelLoaderTests {

    @Test("loadEncoder throws on non-existent URL")
    func loadEncoderMissing() async {
        let loader = ModelLoader()
        let fake = URL(fileURLWithPath: "/tmp/does-not-exist-\(UUID()).mlmodelc")
        do {
            _ = try await loader.loadEncoder(at: fake)
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            if case .modelLoadFailed = error {} else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    @Test("loadDecoder throws on non-existent URL")
    func loadDecoderMissing() async {
        let loader = ModelLoader()
        let fake = URL(fileURLWithPath: "/tmp/does-not-exist-\(UUID()).mlmodelc")
        do {
            _ = try await loader.loadDecoder(at: fake)
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            if case .modelLoadFailed = error {} else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    @Test("loadBundle throws when encoder missing")
    func loadBundleMissing() async {
        let loader = ModelLoader()
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let bundle = ModelBundle(
            model: .tiny,
            directory: dir,
            encoderURL: dir.appendingPathComponent("enc.mlmodelc"),
            decoderURL: dir.appendingPathComponent("dec.mlmodelc"),
            tokenizerURL: dir.appendingPathComponent("tok.json")
        )
        do {
            _ = try await loader.loadBundle(bundle)
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            if case .modelLoadFailed = error {} else {
                Issue.record("wrong error: \(error)")
            }
        }
    }
}
