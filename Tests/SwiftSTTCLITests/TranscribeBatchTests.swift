import ArgumentParser
import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTCLI

@Suite("Transcribe batch and JSON wrappers")
struct TranscribeBatchTests {

    // MARK: - Argument parsing

    @Test("Single file argument parses")
    func singleFile() throws {
        let url = try Self.tempFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let cmd = try TranscribeCommand.parse([url.path])
        #expect(cmd.audioFiles == [url.path])
    }

    @Test("Multiple file arguments parse in order")
    func multipleFiles() throws {
        let urls = try (0..<3).map { _ in try Self.tempFile() }
        defer { for u in urls { try? FileManager.default.removeItem(at: u) } }
        let cmd = try TranscribeCommand.parse(urls.map(\.path))
        #expect(cmd.audioFiles == urls.map(\.path))
    }

    @Test("Empty file list rejected at validation")
    func emptyRejected() {
        do {
            let cmd = try TranscribeCommand.parse([])
            try cmd.validate()
            Issue.record("expected validation to fail")
        } catch {}
    }

    @Test("Missing file rejected at validation")
    func missingFileRejected() {
        let bogus = "/nonexistent/missing-\(UUID().uuidString).wav"
        do {
            let cmd = try TranscribeCommand.parse([bogus])
            try cmd.validate()
            Issue.record("expected validation to fail")
        } catch {}
    }

    @Test("Default format is text")
    func defaultFormat() throws {
        let url = try Self.tempFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let cmd = try TranscribeCommand.parse([url.path])
        #expect(cmd.format == .text)
    }

    @Test("Format flag parses each output type")
    func formatParsing() throws {
        let url = try Self.tempFile()
        defer { try? FileManager.default.removeItem(at: url) }
        for format in OutputFormat.allCases {
            let cmd = try TranscribeCommand.parse([url.path, "--format", format.rawValue])
            #expect(cmd.format == format)
        }
    }

    @Test("Short form -f for format")
    func formatShort() throws {
        let url = try Self.tempFile()
        defer { try? FileManager.default.removeItem(at: url) }
        let cmd = try TranscribeCommand.parse([url.path, "-f", "json"])
        #expect(cmd.format == .json)
    }

    // MARK: - JSON wrapper shape

    @Test("Single-file JSON uses {\"segments\": [...]} wrapper")
    func singleFileJSONWrapper() throws {
        let segments = [
            TranscriptionSegment(text: "hi", start: 0, end: 1)
        ]
        let payload = SegmentRendering.encodeJSON(
            perFile: [(path: "a.wav", segments: segments)],
            isBatch: false
        )
        let output = try #require(payload)
        let data = try #require(output.data(using: .utf8))
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(parsed?["segments"] != nil)
        #expect(parsed?["files"] == nil)
    }

    @Test("Multi-file JSON uses {\"files\": [{path, segments}, ...]} wrapper")
    func batchJSONWrapper() throws {
        let perFile: [(path: String, segments: [TranscriptionSegment])] = [
            (path: "a.wav", segments: [TranscriptionSegment(text: "hello", start: 0, end: 1)]),
            (path: "b.wav", segments: [TranscriptionSegment(text: "world", start: 2, end: 3)]),
        ]
        let payload = SegmentRendering.encodeJSON(perFile: perFile, isBatch: true)
        let output = try #require(payload)
        let data = try #require(output.data(using: .utf8))
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let files = try #require(parsed?["files"] as? [[String: Any]])
        #expect(files.count == 2)
        #expect(files[0]["path"] as? String == "a.wav")
        #expect(files[1]["path"] as? String == "b.wav")
        let bSegs = try #require(files[1]["segments"] as? [[String: Any]])
        #expect(bSegs[0]["text"] as? String == "world")
    }

    @Test("Batch JSON with empty file list emits empty files array")
    func batchJSONEmpty() throws {
        let output = try #require(SegmentRendering.encodeJSON(perFile: [], isBatch: true))
        let data = try #require(output.data(using: .utf8))
        let parsed = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        let files = try #require(parsed?["files"] as? [Any])
        #expect(files.isEmpty)
    }

    // MARK: - Helpers

    static func tempFile() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("swiftstt-batch-\(UUID().uuidString).wav")
        FileManager.default.createFile(atPath: url.path, contents: Data([0]))
        return url
    }
}
