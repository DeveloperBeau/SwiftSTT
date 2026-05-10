import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

// MARK: - Mock URLProtocol

/// Intercepts URL requests and returns canned responses registered per URL path.
private final class MockURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var handlers: [String: (Data, Int)] = [:]

    static func register(path: String, data: Data, statusCode: Int = 200) {
        handlers[path] = (data, statusCode)
    }

    static func reset() { handlers = [:] }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard
            let url = request.url,
            let (data, code) = Self.handlers[url.path]
        else {
            let error = NSError(
                domain: "MockURLProtocol", code: -1,
                userInfo: [
                    NSLocalizedDescriptionKey: "no handler for \(request.url?.path ?? "nil")"
                ])
            client?.urlProtocol(self, didFailWithError: error)
            return
        }
        let response = HTTPURLResponse(
            url: url, statusCode: code, httpVersion: nil, headerFields: nil)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: data)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

// MARK: - Helpers

private func makeMockSession() -> URLSession {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [MockURLProtocol.self]
    return URLSession(configuration: config)
}

// MARK: - Tests

@Suite("ModelDownloader")
struct ModelDownloaderTests {

    @Test("isDownloaded returns false on empty cache")
    func isDownloadedFalse() async {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let dl = ModelDownloader(cacheDirectory: tmp, urlSession: makeMockSession())
        let result = await dl.isDownloaded(.tiny)
        #expect(result == false)
    }

    @Test("Custom cache directory is respected")
    func customCacheDir() async {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let dl = ModelDownloader(cacheDirectory: tmp, urlSession: makeMockSession())
        let dir = await dl.cacheDirectory(for: .tiny)
        #expect(dir.path.contains(tmp.path))
    }

    @Test("bundle(for:) throws when model not downloaded")
    func bundleThrowsWhenMissing() async {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let dl = ModelDownloader(cacheDirectory: tmp, urlSession: makeMockSession())
        do {
            _ = try await dl.bundle(for: .tiny)
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            if case .modelFileMissing = error {
            } else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    @Test("delete on non-existent directory does not throw")
    func deleteNonExistent() async throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let dl = ModelDownloader(cacheDirectory: tmp, urlSession: makeMockSession())
        try await dl.delete(.tiny)
    }

    @Test("Concurrent download guard throws")
    func concurrentDownloadGuard() async throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        MockURLProtocol.reset()

        let treeJSON = """
            [{"type":"file","rfilename":"openai_whisper-tiny/test.bin","size":10}]
            """
        MockURLProtocol.register(
            path: "/api/models/argmaxinc/whisperkit-coreml/tree/main/openai_whisper-tiny",
            data: Data(treeJSON.utf8)
        )
        MockURLProtocol.register(
            path: "/argmaxinc/whisperkit-coreml/resolve/main/openai_whisper-tiny/test.bin",
            data: Data(repeating: 0, count: 10)
        )

        let dl = ModelDownloader(cacheDirectory: tmp, urlSession: makeMockSession())
        _ = try await dl.download(.tiny)

        do {
            _ = try await dl.download(.tiny)
            Issue.record("expected throw on concurrent download")
        } catch let error as SwiftWhisperError {
            if case .modelDownloadFailed = error {
            } else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    @Test("Already-downloaded model returns immediate .complete")
    func alreadyDownloaded() async throws {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let modelDir = tmp.appendingPathComponent("openai_whisper-tiny")
        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        FileManager.default.createFile(
            atPath: modelDir.appendingPathComponent(".complete").path,
            contents: nil
        )

        let dl = ModelDownloader(cacheDirectory: tmp, urlSession: makeMockSession())
        let stream = try await dl.download(.tiny)
        var phases: [DownloadProgress.Phase] = []
        for try await progress in stream {
            phases.append(progress.phase)
        }
        #expect(phases == [.complete])

        // Cleanup
        try FileManager.default.removeItem(at: tmp)
    }
}
