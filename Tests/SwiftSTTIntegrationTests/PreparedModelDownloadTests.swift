import Foundation
import SwiftSTTCore
import SwiftSTTKit
import Testing

/// Drives the download path of model preparation against ``StubURLProtocol``.
///
/// Runs in the default gate. Every request goes to the stub, so no request leaves the
/// machine. The tests share the stub's request counts, so they run one at a time and
/// each starts from zero.
@Suite("Prepared model download", .serialized, .timeLimit(.minutes(1)))
struct PreparedModelDownloadTests {

    private let cache: URL

    init() {
        StubURLProtocol.reset()
        cache = FileManager.default.temporaryDirectory
            .appendingPathComponent("prepared-model-\(UUID().uuidString)", isDirectory: true)
    }

    private func modelFile() -> URL {
        cache.appendingPathComponent("tiny/tiny.bin")
    }

    @Test("A missing model is downloaded and reported as downloaded")
    func missingModelIsDownloaded() async throws {
        defer { try? FileManager.default.removeItem(at: cache) }
        let downloader = ModelDownloader(
            cacheDirectory: cache, urlSession: StubURLProtocol.makeSession())

        let prepared = try await PreparedModel.prepare(.tiny, using: downloader)

        #expect(prepared.wasDownloaded == true)
        #expect(prepared.label == "tiny model: downloaded")
        #expect(
            prepared.bundle.ggmlModelURL.standardizedFileURL.path
                == modelFile().standardizedFileURL.path)
        #expect(try Data(contentsOf: modelFile()) == Data("stub tiny model bytes".utf8))
        #expect(
            FileManager.default.fileExists(
                atPath: cache.appendingPathComponent("tiny/.complete").path))
        #expect(StubURLProtocol.requestCount(for: "ggml-tiny.bin") == 1)
        // The encoder is best effort: one request, answered 404, and the model is kept.
        #expect(StubURLProtocol.requestCount(for: "ggml-tiny-encoder.mlmodelc.zip") == 1)
    }

    @Test("Two concurrent prepares on one empty cache make one request")
    func concurrentPreparesMakeOneRequest() async throws {
        defer { try? FileManager.default.removeItem(at: cache) }
        let cache = self.cache

        // Each result is kept, thrown or not, so the request count is checked either way.
        let results = await withTaskGroup(of: Result<PreparedModel, any Error>.self) { group in
            for _ in 0..<2 {
                group.addTask {
                    let downloader = ModelDownloader(
                        cacheDirectory: cache, urlSession: StubURLProtocol.makeSession())
                    do {
                        return .success(try await PreparedModel.prepare(.tiny, using: downloader))
                    } catch {
                        return .failure(error)
                    }
                }
            }
            return await group.reduce(into: [Result<PreparedModel, any Error>]()) { $0.append($1) }
        }

        let modelRequests = StubURLProtocol.requestCount(for: "ggml-tiny.bin")
        print("concurrent prepares: requests \(StubURLProtocol.allRequestCounts)")
        #expect(modelRequests == 1)
        let prepared = try results.map { try $0.get() }
        #expect(prepared.count == 2)
        #expect(prepared.filter(\.wasDownloaded).count == 1)
        #expect(try Data(contentsOf: modelFile()) == Data("stub tiny model bytes".utf8))
    }
}
