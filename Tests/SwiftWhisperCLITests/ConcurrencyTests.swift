import ArgumentParser
import Foundation
import Testing
@testable import SwiftWhisperCLI

@Suite("--concurrency flag")
struct ConcurrencyTests {

    // MARK: - Parsing

    @Test("Default concurrency is 1")
    func defaultIsOne() throws {
        let url = try Self.tempAudio()
        defer { try? FileManager.default.removeItem(at: url) }
        let cmd = try TranscribeCommand.parse([url.path])
        #expect(cmd.concurrency == 1)
    }

    @Test("--concurrency 4 parses")
    func parsesFour() throws {
        let url = try Self.tempAudio()
        defer { try? FileManager.default.removeItem(at: url) }
        let cmd = try TranscribeCommand.parse([url.path, "--concurrency", "4"])
        #expect(cmd.concurrency == 4)
    }

    // MARK: - Validation

    @Test("--concurrency 0 rejected")
    func zeroRejected() {
        let url = try? Self.tempAudio()
        defer { if let url { try? FileManager.default.removeItem(at: url) } }
        do {
            let cmd = try TranscribeCommand.parse([url!.path, "--concurrency", "0"])
            try cmd.validate()
            Issue.record("expected validation failure")
        } catch {}
    }

    @Test("--concurrency -1 rejected")
    func negativeRejected() {
        let url = try? Self.tempAudio()
        defer { if let url { try? FileManager.default.removeItem(at: url) } }
        do {
            let cmd = try TranscribeCommand.parse([url!.path, "--concurrency", "-1"])
            try cmd.validate()
            Issue.record("expected validation failure")
        } catch {}
    }

    @Test("--concurrency 16 validates (warning only, not rejection)")
    func highValueValidates() throws {
        let url = try Self.tempAudio()
        defer { try? FileManager.default.removeItem(at: url) }
        let cmd = try TranscribeCommand.parse([url.path, "--concurrency", "16"])
        try cmd.validate()
        #expect(cmd.concurrency == 16)
    }

    @Test("--concurrency 1 validates")
    func oneValidates() throws {
        let url = try Self.tempAudio()
        defer { try? FileManager.default.removeItem(at: url) }
        let cmd = try TranscribeCommand.parse([url.path, "--concurrency", "1"])
        try cmd.validate()
    }

    // MARK: - Helpers

    static func tempAudio() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftwhisper-concurrency-\(UUID().uuidString).wav")
        FileManager.default.createFile(atPath: url.path, contents: Data([0]))
        return url
    }
}
