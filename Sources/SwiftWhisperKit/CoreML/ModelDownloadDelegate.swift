import Foundation
import SwiftWhisperCore
import Synchronization

/// Bridges `URLSessionDownloadDelegate` callbacks into per-task
/// `AsyncThrowingStream` continuations.
///
/// URLSession invokes delegate methods from a private serial queue, so the delegate keeps its mutable state
/// inside a `Mutex<DelegateState>`. The class is `@unchecked Sendable`
/// because URLSession's delegate API is not Sendable-aware.
final class ModelDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {

    /// Per-task hooks the delegate needs to drive a single file download to
    /// completion or failure on behalf of the actor-owned downloader.
    struct TaskHandle {
        var continuation: AsyncThrowingStream<DownloadProgress, any Error>.Continuation
        var destination: URL
        /// Optional terminal hook fired exactly once on success or failure.
        ///
        /// Lets the actor await per-file completion without listening to the stream.
        var finished: (@Sendable (Result<URL, any Error>) -> Void)?
    }

    struct DelegateState {
        var handles: [Int: TaskHandle] = [:]
        var pendingCompletions: [@Sendable () -> Void] = []
    }

    let state = Mutex(DelegateState())

    // MARK: - Static registry

    /// Maps background-session identifiers to the delegate that owns them so
    /// `application(_:handleEventsForBackgroundURLSession:completionHandler:)`
    /// can route the system completion handler back to the right session.
    private static let registry = Mutex<[String: ModelDownloadDelegate]>([:])

    static func register(_ delegate: ModelDownloadDelegate, for identifier: String) {
        registry.withLock { $0[identifier] = delegate }
    }

    static func unregister(identifier: String) {
        registry.withLock { _ = $0.removeValue(forKey: identifier) }
    }

    static func delegate(for identifier: String) -> ModelDownloadDelegate? {
        registry.withLock { $0[identifier] }
    }

    // MARK: - Task tracking

    func register(taskIdentifier: Int, handle: TaskHandle) {
        state.withLock { $0.handles[taskIdentifier] = handle }
    }

    func handle(for taskIdentifier: Int) -> TaskHandle? {
        state.withLock { $0.handles[taskIdentifier] }
    }

    @discardableResult
    func clear(taskIdentifier: Int) -> TaskHandle? {
        state.withLock { $0.handles.removeValue(forKey: taskIdentifier) }
    }

    func stash(completion: @escaping @Sendable () -> Void) {
        state.withLock { $0.pendingCompletions.append(completion) }
    }

    func drainCompletions() -> [@Sendable () -> Void] {
        state.withLock {
            let drained = $0.pendingCompletions
            $0.pendingCompletions = []
            return drained
        }
    }

    // MARK: - URLSessionDownloadDelegate

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        emitFraction(
            taskId: downloadTask.taskIdentifier,
            written: totalBytesWritten,
            expected: totalBytesExpectedToWrite
        )
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didResumeAtOffset fileOffset: Int64,
        expectedTotalBytes: Int64
    ) {
        emitFraction(
            taskId: downloadTask.taskIdentifier,
            written: fileOffset,
            expected: expectedTotalBytes
        )
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        guard let handle = handle(for: downloadTask.taskIdentifier) else { return }

        // Move the file synchronously here: URLSession deletes the temp file
        // when this callback returns. Errors propagate via the finished hook.
        let fileManager = FileManager.default
        do {
            if fileManager.fileExists(atPath: handle.destination.path) {
                try fileManager.removeItem(at: handle.destination)
            }
            try fileManager.createDirectory(
                at: handle.destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try fileManager.moveItem(at: location, to: handle.destination)
            handle.continuation.yield(
                DownloadProgress(
                    totalFiles: 1, completedFiles: 1,
                    totalBytes: 0, totalBytesDownloaded: 0,
                    phase: .complete
                ))
            handle.finished?(.success(handle.destination))
        } catch {
            handle.continuation.finish(throwing: error)
            handle.finished?(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: (any Error)?
    ) {
        guard let handle = clear(taskIdentifier: task.taskIdentifier) else { return }
        if let error {
            handle.continuation.finish(throwing: error)
            handle.finished?(.failure(error))
        } else {
            handle.continuation.finish()
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        for completion in drainCompletions() {
            completion()
        }
    }

    // MARK: - Helpers

    private func emitFraction(taskId: Int, written: Int64, expected: Int64) {
        guard let handle = handle(for: taskId) else { return }
        let totalBytes = max(expected, 0)
        let downloaded = min(written, totalBytes == 0 ? written : totalBytes)
        handle.continuation.yield(
            DownloadProgress(
                totalFiles: 1, completedFiles: 0,
                totalBytes: totalBytes, totalBytesDownloaded: downloaded,
                phase: .downloading
            ))
    }
}
