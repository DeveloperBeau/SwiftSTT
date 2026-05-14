import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

@Suite("ModelDownloader")
struct ModelDownloaderTests {

    @Test("isDownloaded false on empty cache")
    func isDownloadedFalse() async {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString
        )
        let dl = ModelDownloader(cacheDirectory: tmp)
        let result = await dl.isDownloaded(.tiny)
        #expect(result == false)
    }

    @Test("Already-downloaded model: marker + ggml.bin present")
    func alreadyDownloaded() async throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString
        )
        let dir = tmp.appendingPathComponent("ggml-tiny.en")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: dir.appendingPathComponent("ggml-tiny.en.bin").path,
            contents: Data("stub".utf8)
        )
        FileManager.default.createFile(
            atPath: dir.appendingPathComponent(".complete").path,
            contents: nil
        )
        let dl = ModelDownloader(cacheDirectory: tmp)
        #expect(await dl.isDownloaded(.tiny) == true)
        try FileManager.default.removeItem(at: tmp)
    }
}
