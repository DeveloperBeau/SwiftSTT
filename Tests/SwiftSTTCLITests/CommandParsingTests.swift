import ArgumentParser
import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTCLI

@Suite("CLI command parsing")
struct CommandParsingTests {

    // MARK: - DownloadCommand

    @Test("download parses model name")
    func downloadParsesModel() throws {
        let cmd = try DownloadCommand.parse(["tiny"])
        #expect(cmd.model == .tiny)
    }

    @Test("download accepts each known model")
    func downloadAcceptsEachModel() throws {
        for model in WhisperModel.allCases {
            let cmd = try DownloadCommand.parse([model.rawValue])
            #expect(cmd.model == model)
        }
    }

    @Test("download rejects unknown model")
    func downloadRejectsUnknown() {
        #expect(throws: (any Error).self) {
            try DownloadCommand.parse(["nonexistent"])
        }
    }

    @Test("download requires a model argument")
    func downloadRequiresModel() {
        #expect(throws: (any Error).self) {
            try DownloadCommand.parse([])
        }
    }

    // MARK: - ListModelsCommand

    @Test("list-models parses with no arguments")
    func listModelsParses() throws {
        let cmd = try ListModelsCommand.parse([])
        _ = cmd
    }

    // MARK: - InfoCommand

    @Test("info parses model name")
    func infoParsesModel() throws {
        let cmd = try InfoCommand.parse(["base"])
        #expect(cmd.model == .base)
    }

    @Test("info requires a model argument")
    func infoRequiresModel() {
        #expect(throws: (any Error).self) {
            try InfoCommand.parse([])
        }
    }

    // MARK: - TranscribeCommand

    @Test("transcribe defaults model to base and language to nil")
    func transcribeDefaults() throws {
        let url = try Self.makeTempFile()
        defer { try? FileManager.default.removeItem(at: url) }

        let cmd = try TranscribeCommand.parse([url.path])
        #expect(cmd.audioFiles == [url.path])
        #expect(cmd.model == .base)
        #expect(cmd.language == nil)
        #expect(cmd.format == .text)
    }

    @Test("transcribe parses --model and --language")
    func transcribeParsesFlags() throws {
        let url = try Self.makeTempFile()
        defer { try? FileManager.default.removeItem(at: url) }

        let cmd = try TranscribeCommand.parse([
            url.path,
            "--model", "tiny",
            "--language", "en",
        ])
        #expect(cmd.model == .tiny)
        #expect(cmd.language == "en")
    }

    @Test("transcribe parses short flags")
    func transcribeShortFlags() throws {
        let url = try Self.makeTempFile()
        defer { try? FileManager.default.removeItem(at: url) }

        let cmd = try TranscribeCommand.parse([
            url.path,
            "-m", "small",
            "-l", "de",
        ])
        #expect(cmd.model == .small)
        #expect(cmd.language == "de")
    }

    @Test("transcribe rejects non-existent file path during validate")
    func transcribeRejectsMissingFile() {
        let bogus = "/nonexistent/path/missing-\(UUID().uuidString).wav"
        do {
            let cmd = try TranscribeCommand.parse([bogus])
            try cmd.validate()
            Issue.record("expected validation to fail")
        } catch {}
    }

    @Test("transcribe accepts existing file path")
    func transcribeAcceptsExistingFile() throws {
        let url = try Self.makeTempFile()
        defer { try? FileManager.default.removeItem(at: url) }

        let cmd = try TranscribeCommand.parse([url.path])
        #expect(cmd.audioFiles == [url.path])
    }

    // MARK: - SwiftSTT root

    @Test("root command exposes version")
    func rootHasVersion() {
        #expect(SwiftSTT.configuration.version == "0.13.0")
    }

    @Test("root command lists subcommands")
    func rootSubcommands() {
        let names = SwiftSTT.configuration.subcommands.map { $0._commandName }
        #expect(names.contains("download"))
        #expect(names.contains("list-models"))
        #expect(names.contains("transcribe"))
        #expect(names.contains("transcribe-mic"))
        #expect(names.contains("info"))
    }

    @Test("help text generation does not crash")
    func helpText() {
        let downloadHelp = DownloadCommand.helpMessage()
        #expect(downloadHelp.contains("download"))

        let listHelp = ListModelsCommand.helpMessage()
        #expect(listHelp.contains("list-models"))

        let transcribeHelp = TranscribeCommand.helpMessage()
        #expect(transcribeHelp.contains("transcribe"))

        let infoHelp = InfoCommand.helpMessage()
        #expect(infoHelp.contains("info"))

        let rootHelp = SwiftSTT.helpMessage()
        #expect(rootHelp.contains("swiftstt"))
    }

    // MARK: - Helpers

    static func makeTempFile() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftstt-cli-\(UUID().uuidString).wav")
        FileManager.default.createFile(atPath: url.path, contents: Data([0]))
        return url
    }
}
