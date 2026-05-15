import ArgumentParser
import Foundation
import Testing

@testable import SwiftSTTCLI

@Suite("--output flag and OutputDestination")
struct OutputFlagTests {

    // MARK: - Argument parsing

    @Test("transcribe parses --output and --no-clobber")
    func transcribeParsesOutput() throws {
        let url = try Self.tempAudio()
        defer { try? FileManager.default.removeItem(at: url) }
        let cmd = try TranscribeCommand.parse([
            url.path,
            "--output", "/tmp/out.txt",
            "--no-clobber",
        ])
        #expect(cmd.output == "/tmp/out.txt")
        #expect(cmd.noClobber == true)
    }

    @Test("transcribe -o short flag works")
    func transcribeShortOutput() throws {
        let url = try Self.tempAudio()
        defer { try? FileManager.default.removeItem(at: url) }
        let cmd = try TranscribeCommand.parse([url.path, "-o", "/tmp/out.txt"])
        #expect(cmd.output == "/tmp/out.txt")
        #expect(cmd.noClobber == false)
    }

    @Test("transcribe-mic parses --output")
    func micParsesOutput() throws {
        let cmd = try TranscribeMicCommand.parse([
            "--output", "/tmp/mic.txt",
            "--no-clobber",
        ])
        #expect(cmd.output == "/tmp/mic.txt")
        #expect(cmd.noClobber == true)
    }

    @Test("Default output is nil (stdout)")
    func defaultIsNil() throws {
        let url = try Self.tempAudio()
        defer { try? FileManager.default.removeItem(at: url) }
        let cmd = try TranscribeCommand.parse([url.path])
        #expect(cmd.output == nil)
        #expect(cmd.noClobber == false)
    }

    // MARK: - OutputDestination behaviour

    @Test("file destination writes content")
    func fileWrites() throws {
        let path = Self.tempPath()
        defer { try? FileManager.default.removeItem(atPath: path) }

        let dest = try OutputDestination.file(at: path, noClobber: false)
        dest.writeLine("hello")
        dest.writeLine("world")
        dest.close()

        let contents = try String(contentsOfFile: path, encoding: .utf8)
        #expect(contents == "hello\nworld\n")
    }

    @Test("--no-clobber refuses to overwrite an existing file")
    func noClobberRefuses() throws {
        let path = Self.tempPath()
        FileManager.default.createFile(atPath: path, contents: Data("existing".utf8))
        defer { try? FileManager.default.removeItem(atPath: path) }

        do {
            _ = try OutputDestination.file(at: path, noClobber: true)
            Issue.record("expected ValidationError")
        } catch let error as ValidationError {
            #expect(error.message.contains("already exists"))
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }

    @Test("Overwrite truncates the existing file")
    func overwriteTruncates() throws {
        let path = Self.tempPath()
        FileManager.default.createFile(atPath: path, contents: Data("longer existing content".utf8))
        defer { try? FileManager.default.removeItem(atPath: path) }

        let dest = try OutputDestination.file(at: path, noClobber: false)
        dest.write("hi")
        dest.close()

        let contents = try String(contentsOfFile: path, encoding: .utf8)
        #expect(contents == "hi")
    }

    @Test("Tilde paths expand")
    func tildeExpansion() throws {
        let unique = "swiftstt-output-\(UUID().uuidString).txt"
        let tildePath = "~/\(unique)"
        let expandedPath = (tildePath as NSString).expandingTildeInPath
        defer { try? FileManager.default.removeItem(atPath: expandedPath) }

        let dest = try OutputDestination.file(at: tildePath, noClobber: false)
        dest.writeLine("tilde works")
        dest.close()

        let contents = try String(contentsOfFile: expandedPath, encoding: .utf8)
        #expect(contents.contains("tilde works"))
    }

    @Test("Permission failure yields a clear error including the path")
    func permissionDenied() {
        let badPath = "/this/path/should/not/exist/output-\(UUID().uuidString).txt"
        do {
            _ = try OutputDestination.file(at: badPath, noClobber: false)
            Issue.record("expected ValidationError")
        } catch let error as ValidationError {
            #expect(error.message.contains(badPath))
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }

    @Test("stdout destination is a no-op on close()")
    func stdoutCloseSafe() {
        let dest = OutputDestination.stdout()
        dest.close()
        dest.close()
    }

    // MARK: - Helpers

    static func tempAudio() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftstt-output-flag-\(UUID().uuidString).wav")
        FileManager.default.createFile(atPath: url.path, contents: Data([0]))
        return url
    }

    static func tempPath() -> String {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftstt-output-\(UUID().uuidString).txt")
            .path
    }
}
