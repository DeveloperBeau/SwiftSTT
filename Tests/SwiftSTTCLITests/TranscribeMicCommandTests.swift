import ArgumentParser
import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTCLI

@Suite("transcribe-mic command and SignalHandler")
struct TranscribeMicCommandTests {

    // MARK: - Argument parsing

    @Test("Subcommand parses with no arguments")
    func parsesWithoutArgs() throws {
        let cmd = try TranscribeMicCommand.parse([])
        #expect(cmd.model == .base)
        #expect(cmd.language == nil)
        #expect(cmd.format == .text)
        #expect(cmd.maxDuration == nil)
        #expect(cmd.cacheDir == nil)
    }

    @Test("Subcommand parses every flag")
    func parsesAllFlags() throws {
        let cmd = try TranscribeMicCommand.parse([
            "--model", "small",
            "--language", "en",
            "--format", "srt",
            "--max-duration", "30",
            "--cache-dir", "/tmp/cache",
        ])
        #expect(cmd.model == .small)
        #expect(cmd.language == "en")
        #expect(cmd.format == .srt)
        #expect(cmd.maxDuration == 30)
        #expect(cmd.cacheDir == "/tmp/cache")
    }

    @Test("Default max-duration is nil")
    func defaultMaxDuration() throws {
        let cmd = try TranscribeMicCommand.parse([])
        #expect(cmd.maxDuration == nil)
    }

    @Test("Default format is text")
    func defaultFormat() throws {
        let cmd = try TranscribeMicCommand.parse([])
        #expect(cmd.format == .text)
    }

    @Test("Subcommand registered on root")
    func subcommandRegistered() {
        let names = SwiftSTT.configuration.subcommands.map { $0._commandName }
        #expect(names.contains("transcribe-mic"))
    }

    // MARK: - SignalHandler

    @Test("Stop flag toggles cleanly via the helper")
    func signalFlagToggle() {
        SignalHandler.reset()
        #expect(SignalHandler.isStopRequested() == false)
        SignalHandler.markStopRequested()
        #expect(SignalHandler.isStopRequested() == true)
        SignalHandler.reset()
        #expect(SignalHandler.isStopRequested() == false)
    }

    @Test("Reset is idempotent")
    func resetIdempotent() {
        SignalHandler.reset()
        SignalHandler.reset()
        #expect(SignalHandler.isStopRequested() == false)
    }

    @Test("Mark is idempotent")
    func markIdempotent() {
        SignalHandler.reset()
        SignalHandler.markStopRequested()
        SignalHandler.markStopRequested()
        #expect(SignalHandler.isStopRequested() == true)
        SignalHandler.reset()
    }

    @Test("Install and uninstall do not crash")
    func installUninstall() {
        SignalHandler.reset()
        SignalHandler.installSIGINT()
        SignalHandler.uninstallSIGINT()
        #expect(SignalHandler.isStopRequested() == false)
    }
}
