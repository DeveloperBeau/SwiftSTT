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
            ggmlModelURL: dir.appendingPathComponent("ggml-tiny.en.bin")
        )
        #expect(bundle.model == .tiny)
        #expect(bundle.directory == dir)
        #expect(bundle.ggmlModelURL.lastPathComponent == "ggml-tiny.en.bin")
        #expect(bundle.coreMLEncoderURL == nil)
    }

    @Test("Equality compares all fields")
    func equality() {
        let dir = URL(fileURLWithPath: "/tmp/a")
        let a = ModelBundle(
            model: .tiny,
            directory: dir,
            ggmlModelURL: dir.appendingPathComponent("g.bin")
        )
        let b = a
        #expect(a == b)
    }

    @Test("Inequality when model differs")
    func inequalityModel() {
        let dir = URL(fileURLWithPath: "/tmp/a")
        let a = ModelBundle(
            model: .tiny,
            directory: dir,
            ggmlModelURL: dir.appendingPathComponent("g.bin")
        )
        let b = ModelBundle(
            model: .base,
            directory: dir,
            ggmlModelURL: dir.appendingPathComponent("g.bin")
        )
        #expect(a != b)
    }

    @Test("coreMLEncoderURL stored and compared")
    func coreMLEncoder() {
        let dir = URL(fileURLWithPath: "/tmp/test")
        let encoder = dir.appendingPathComponent("encoder.mlmodelc")
        let bundle = ModelBundle(
            model: .tiny,
            directory: dir,
            ggmlModelURL: dir.appendingPathComponent("g.bin"),
            coreMLEncoderURL: encoder
        )
        #expect(bundle.coreMLEncoderURL == encoder)
        let bundleNoEncoder = ModelBundle(
            model: .tiny,
            directory: dir,
            ggmlModelURL: dir.appendingPathComponent("g.bin")
        )
        #expect(bundle != bundleNoEncoder)
    }
}
