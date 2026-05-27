import ArgumentParser
import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTCLI

@Suite("download --background flag")
struct DownloadBackgroundFlagTests {

    @Test("--background flag parses")
    func backgroundFlagParses() throws {
        let cmd = try DownloadCommand.parse(["tiny", "--background"])
        #expect(cmd.background == true)
        #expect(cmd.model == .tiny)
    }

    @Test("default is foreground")
    func defaultIsForeground() throws {
        let cmd = try DownloadCommand.parse(["tiny"])
        #expect(cmd.background == false)
    }

    @Test("--background and --cache-dir together parse correctly")
    func backgroundAndCacheDir() throws {
        let cmd = try DownloadCommand.parse(["base", "--background", "--cache-dir", "/tmp/foo"])
        #expect(cmd.background == true)
        #expect(cmd.cacheDir == "/tmp/foo")
        #expect(cmd.model == .base)
    }

    @Test("--background flag appears in help text")
    func backgroundInHelp() throws {
        let help = DownloadCommand.helpMessage()
        #expect(help.contains("--background"))
    }
}
