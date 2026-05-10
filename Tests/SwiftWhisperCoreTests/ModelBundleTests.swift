import Foundation
import Testing

@testable import SwiftWhisperCore

@Suite("ModelBundle")
struct ModelBundleTests {

    @Test("Memberwise init stores all fields")
    func memberwiseInit() {
        let dir = URL(fileURLWithPath: "/tmp/test")
        let bundle = ModelBundle(
            model: .tiny,
            directory: dir,
            encoderURL: dir.appendingPathComponent("enc"),
            decoderURL: dir.appendingPathComponent("dec"),
            tokenizerURL: dir.appendingPathComponent("tok.json")
        )
        #expect(bundle.model == .tiny)
        #expect(bundle.directory == dir)
    }

    @Test("Equality compares all fields")
    func equality() {
        let dir = URL(fileURLWithPath: "/tmp/a")
        let a = ModelBundle(
            model: .tiny, directory: dir,
            encoderURL: dir.appendingPathComponent("e"),
            decoderURL: dir.appendingPathComponent("d"),
            tokenizerURL: dir.appendingPathComponent("t")
        )
        let b = a
        #expect(a == b)
    }
}
