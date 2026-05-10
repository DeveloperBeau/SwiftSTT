import ArgumentParser
import Foundation
import Testing

@testable import SwiftWhisperCLI

@Suite("--cache-dir option")
struct CacheDirTests {

    @Test("download accepts --cache-dir")
    func downloadAccepts() throws {
        let cmd = try DownloadCommand.parse(["tiny", "--cache-dir", "/tmp/x"])
        #expect(cmd.cacheDir == "/tmp/x")
    }

    @Test("list-models accepts --cache-dir")
    func listModelsAccepts() throws {
        let cmd = try ListModelsCommand.parse(["--cache-dir", "/tmp/x"])
        #expect(cmd.cacheDir == "/tmp/x")
    }

    @Test("info accepts --cache-dir")
    func infoAccepts() throws {
        let cmd = try InfoCommand.parse(["base", "--cache-dir", "/tmp/x"])
        #expect(cmd.cacheDir == "/tmp/x")
    }

    @Test("transcribe accepts --cache-dir")
    func transcribeAccepts() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftwhisper-cachedir-\(UUID().uuidString).wav")
        FileManager.default.createFile(atPath: url.path, contents: Data([0]))
        defer { try? FileManager.default.removeItem(at: url) }
        let cmd = try TranscribeCommand.parse([url.path, "--cache-dir", "/tmp/x"])
        #expect(cmd.cacheDir == "/tmp/x")
    }

    @Test("transcribe-mic accepts --cache-dir")
    func transcribeMicAccepts() throws {
        let cmd = try TranscribeMicCommand.parse(["--cache-dir", "/tmp/x"])
        #expect(cmd.cacheDir == "/tmp/x")
    }

    // MARK: - Defaults

    @Test("Defaults applied when omitted")
    func defaults() throws {
        let dl = try DownloadCommand.parse(["tiny"])
        #expect(dl.cacheDir == nil)
        let list = try ListModelsCommand.parse([])
        #expect(list.cacheDir == nil)
        let info = try InfoCommand.parse(["base"])
        #expect(info.cacheDir == nil)
        let mic = try TranscribeMicCommand.parse([])
        #expect(mic.cacheDir == nil)
    }

    // MARK: - Path resolution

    @Test("nil and empty input returns nil")
    func resolveNil() {
        #expect(CacheDirectoryOption.resolve(nil) == nil)
        #expect(CacheDirectoryOption.resolve("") == nil)
    }

    @Test("Tilde expansion produces an absolute URL")
    func tildeExpansion() throws {
        let resolved = try #require(CacheDirectoryOption.resolve("~/swiftwhisper-cache"))
        let path = resolved.path
        #expect(path.hasPrefix("/"))
        #expect(!path.contains("~"))
        #expect(path.hasSuffix("/swiftwhisper-cache"))
    }

    @Test("Absolute path passes through unchanged")
    func absolutePath() throws {
        let resolved = try #require(CacheDirectoryOption.resolve("/tmp/whisper"))
        #expect(resolved.path == "/tmp/whisper")
    }

    @Test("Relative path is left untouched and made into a URL")
    func relativePath() throws {
        let resolved = try #require(CacheDirectoryOption.resolve("local/cache"))
        #expect(resolved.path.hasSuffix("/local/cache") || resolved.path.contains("local/cache"))
    }
}
