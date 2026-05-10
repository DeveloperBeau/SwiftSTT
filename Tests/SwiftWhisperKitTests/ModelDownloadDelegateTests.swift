import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

// MARK: - URLProtocol that streams data in chunks so we can observe progress callbacks

private final class ChunkedMockProtocol: URLProtocol, @unchecked Sendable {

    nonisolated(unsafe) static var nextResponse: ResponseSpec?

    struct ResponseSpec {
        var data: Data
        var statusCode: Int
        var chunkSize: Int
        var error: (any Error)?
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        guard let url = request.url, let spec = Self.nextResponse else {
            client?.urlProtocol(self, didFailWithError: NSError(domain: "Mock", code: -1))
            return
        }
        if let error = spec.error {
            client?.urlProtocol(self, didFailWithError: error)
            return
        }
        let response = HTTPURLResponse(
            url: url,
            statusCode: spec.statusCode,
            httpVersion: nil,
            headerFields: ["Content-Length": "\(spec.data.count)"]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        var offset = 0
        while offset < spec.data.count {
            let end = min(offset + spec.chunkSize, spec.data.count)
            client?.urlProtocol(self, didLoad: spec.data.subdata(in: offset..<end))
            offset = end
        }
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

private func makeBackgroundLikeSession(delegate: any URLSessionDownloadDelegate) -> URLSession {
    let config = URLSessionConfiguration.ephemeral
    config.protocolClasses = [ChunkedMockProtocol.self]
    return URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
}

@Suite("ModelDownloadDelegate")
struct ModelDownloadDelegateTests {

    @Test("didWriteData emits .downloading progress with totals")
    func didWriteDataEmitsProgress() async throws {
        let delegate = ModelDownloadDelegate()
        let session = makeBackgroundLikeSession(delegate: delegate)
        defer { session.invalidateAndCancel() }
        let task = session.downloadTask(with: URL(string: "https://example.com/file.bin")!)

        let (stream, continuation) = AsyncThrowingStream<DownloadProgress, any Error>.makeStream()
        delegate.register(
            taskIdentifier: task.taskIdentifier,
            handle: .init(
                continuation: continuation,
                destination: FileManager.default.temporaryDirectory.appendingPathComponent(
                    UUID().uuidString)
            ))

        delegate.urlSession(
            session,
            downloadTask: task,
            didWriteData: 50,
            totalBytesWritten: 50,
            totalBytesExpectedToWrite: 100
        )
        continuation.finish()

        var phases: [DownloadProgress.Phase] = []
        var seenFraction: Double = 0
        for try await progress in stream {
            phases.append(progress.phase)
            seenFraction = max(seenFraction, progress.fractionComplete)
        }
        #expect(phases.contains(.downloading))
        #expect(seenFraction >= 0.4)
    }

    @Test("didFinishDownloadingTo moves file and emits .complete")
    func didFinishMovesFile() async throws {
        let delegate = ModelDownloadDelegate()
        let session = makeBackgroundLikeSession(delegate: delegate)
        defer { session.invalidateAndCancel() }
        let task = session.downloadTask(with: URL(string: "https://example.com/file.bin")!)

        let temp = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString + ".bin")
        try Data("payload".utf8).write(to: temp)

        let dest = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
            .appendingPathComponent("out.bin")

        let (stream, continuation) = AsyncThrowingStream<DownloadProgress, any Error>.makeStream()
        let finishedExpectation = ExpectationBox()
        delegate.register(
            taskIdentifier: task.taskIdentifier,
            handle: .init(
                continuation: continuation,
                destination: dest,
                finished: { result in
                    finishedExpectation.set(result)
                }
            ))

        delegate.urlSession(session, downloadTask: task, didFinishDownloadingTo: temp)
        delegate.urlSession(session, task: task, didCompleteWithError: nil)

        var phases: [DownloadProgress.Phase] = []
        for try await progress in stream {
            phases.append(progress.phase)
        }
        #expect(phases.contains(.complete))
        #expect(FileManager.default.fileExists(atPath: dest.path))
        #expect(finishedExpectation.value != nil)
        if case .success(let url) = finishedExpectation.value {
            #expect(url == dest)
        } else {
            Issue.record("expected success result")
        }
    }

    @Test("didCompleteWithError finishes stream with thrown error")
    func didCompleteWithErrorFinishes() async {
        let delegate = ModelDownloadDelegate()
        let session = makeBackgroundLikeSession(delegate: delegate)
        defer { session.invalidateAndCancel() }
        let task = session.downloadTask(with: URL(string: "https://example.com/file.bin")!)

        let (stream, continuation) = AsyncThrowingStream<DownloadProgress, any Error>.makeStream()
        let finishedBox = ExpectationBox()
        delegate.register(
            taskIdentifier: task.taskIdentifier,
            handle: .init(
                continuation: continuation,
                destination: FileManager.default.temporaryDirectory.appendingPathComponent(
                    UUID().uuidString),
                finished: { finishedBox.set($0) }
            ))

        let error = NSError(domain: "Test", code: 42)
        delegate.urlSession(session, task: task, didCompleteWithError: error)

        do {
            for try await _ in stream {}
            Issue.record("expected error")
        } catch let nserror as NSError {
            #expect(nserror.code == 42)
        }
        if case .failure = finishedBox.value {
        } else {
            Issue.record("expected failure result")
        }
    }

    @Test("didResumeAtOffset emits .downloading progress")
    func didResumeAtOffsetEmits() async throws {
        let delegate = ModelDownloadDelegate()
        let session = makeBackgroundLikeSession(delegate: delegate)
        defer { session.invalidateAndCancel() }
        let task = session.downloadTask(with: URL(string: "https://example.com/file.bin")!)

        let (stream, continuation) = AsyncThrowingStream<DownloadProgress, any Error>.makeStream()
        delegate.register(
            taskIdentifier: task.taskIdentifier,
            handle: .init(
                continuation: continuation,
                destination: FileManager.default.temporaryDirectory.appendingPathComponent(
                    UUID().uuidString)
            ))

        delegate.urlSession(
            session,
            downloadTask: task,
            didResumeAtOffset: 25,
            expectedTotalBytes: 100
        )
        continuation.finish()

        var fractions: [Double] = []
        for try await progress in stream {
            fractions.append(progress.fractionComplete)
        }
        #expect(fractions.contains { $0 >= 0.2 })
    }

    @Test("Multiple concurrent tasks routed to correct continuations")
    func routesToCorrectContinuations() async throws {
        let delegate = ModelDownloadDelegate()
        let session = makeBackgroundLikeSession(delegate: delegate)
        defer { session.invalidateAndCancel() }
        let taskA = session.downloadTask(with: URL(string: "https://example.com/a")!)
        let taskB = session.downloadTask(with: URL(string: "https://example.com/b")!)

        let (streamA, contA) = AsyncThrowingStream<DownloadProgress, any Error>.makeStream()
        let (streamB, contB) = AsyncThrowingStream<DownloadProgress, any Error>.makeStream()
        delegate.register(
            taskIdentifier: taskA.taskIdentifier,
            handle: .init(
                continuation: contA,
                destination: FileManager.default.temporaryDirectory.appendingPathComponent(
                    UUID().uuidString)
            ))
        delegate.register(
            taskIdentifier: taskB.taskIdentifier,
            handle: .init(
                continuation: contB,
                destination: FileManager.default.temporaryDirectory.appendingPathComponent(
                    UUID().uuidString)
            ))

        delegate.urlSession(
            session, downloadTask: taskA, didWriteData: 10, totalBytesWritten: 10,
            totalBytesExpectedToWrite: 100)
        delegate.urlSession(
            session, downloadTask: taskB, didWriteData: 50, totalBytesWritten: 50,
            totalBytesExpectedToWrite: 100)
        contA.finish()
        contB.finish()

        var aMax: Double = 0
        var bMax: Double = 0
        for try await p in streamA { aMax = max(aMax, p.fractionComplete) }
        for try await p in streamB { bMax = max(bMax, p.fractionComplete) }
        #expect(aMax > 0)
        #expect(bMax > aMax)
    }

    @Test("urlSessionDidFinishEvents fires registered completions")
    func didFinishEventsFiresCompletions() async {
        let delegate = ModelDownloadDelegate()
        let session = makeBackgroundLikeSession(delegate: delegate)
        defer { session.invalidateAndCancel() }

        let counter = CounterBox()
        delegate.stash { counter.increment() }
        delegate.stash { counter.increment() }

        delegate.urlSessionDidFinishEvents(forBackgroundURLSession: session)

        #expect(counter.value == 2)
        #expect(delegate.drainCompletions().isEmpty)
    }

    @Test("Static registry roundtrip and unregister")
    func registryRoundtrip() {
        let id = "test.\(UUID().uuidString)"
        let delegate = ModelDownloadDelegate()
        ModelDownloadDelegate.register(delegate, for: id)
        #expect(ModelDownloadDelegate.delegate(for: id) === delegate)
        ModelDownloadDelegate.unregister(identifier: id)
        #expect(ModelDownloadDelegate.delegate(for: id) == nil)
    }
}

// MARK: - Tiny boxed helpers (the test harness avoids actor-isolated state)

private final class ExpectationBox: @unchecked Sendable {
    private var _value: Result<URL, any Error>?
    private let lock = NSLock()
    var value: Result<URL, any Error>? {
        lock.withLock { _value }
    }
    func set(_ v: Result<URL, any Error>) {
        lock.withLock { _value = v }
    }
}

private final class CounterBox: @unchecked Sendable {
    private var _value: Int = 0
    private let lock = NSLock()
    var value: Int { lock.withLock { _value } }
    func increment() { lock.withLock { _value += 1 } }
}
