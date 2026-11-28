import Foundation
import SwiftSTTCore
import SwiftSTTKit
import Testing

/// Checks the model preparation shared by the integration suites.
///
/// Runs in the default gate. Each test uses its own temporary cache and makes no
/// network request.
@Suite("Prepared model", .timeLimit(.minutes(1)))
struct PreparedModelTests {

    @Test("A cached model is returned without a download")
    func cachedModelIsReturned() async throws {
        let cache = FileManager.default.temporaryDirectory
            .appendingPathComponent("prepared-model-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: cache) }
        let modelDirectory = cache.appendingPathComponent("tiny", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)
        try Data([1]).write(to: modelDirectory.appendingPathComponent("tiny.bin"))
        try Data().write(to: modelDirectory.appendingPathComponent(".complete"))

        let prepared = try await PreparedModel.prepare(
            .tiny, using: ModelDownloader(cacheDirectory: cache))

        #expect(!prepared.wasDownloaded)
        #expect(prepared.label == "tiny model: cached")
        #expect(
            prepared.bundle.ggmlModelURL.standardizedFileURL.path
                == cache.standardizedFileURL.path + "/tiny/tiny.bin")
    }

    @Test("A missing model whose download fails throws the download error")
    func failedDownloadThrows() async throws {
        // A regular file where the cache directory's parent should be makes the download
        // fail when it creates the model directory, before any request is made.
        let blocker = FileManager.default.temporaryDirectory
            .appendingPathComponent("prepared-model-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: blocker) }
        try Data([1]).write(to: blocker)
        let downloader = ModelDownloader(
            cacheDirectory: blocker.appendingPathComponent("Models", isDirectory: true))

        await #expect {
            _ = try await PreparedModel.prepare(.tiny, using: downloader)
        } throws: { error in
            guard case .modelDownloadFailed = error as? SwiftSTTError else { return false }
            return true
        }
    }
}
