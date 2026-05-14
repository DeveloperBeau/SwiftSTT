import Foundation
import SwiftWhisperCore

/// Downloads a single ggml Whisper model file from HuggingFace and caches it locally.
///
/// The downloader fetches `ggml-<stem>.bin` directly from the ggerganov/whisper.cpp
/// HuggingFace repo and writes a `.complete` marker when the file has fully
/// landed. The marker prevents `isDownloaded` from returning `true` for a
/// partially-downloaded model (e.g. after a crash or cancellation).
///
/// Guards against concurrent downloads of the same model using an in-flight
/// task dictionary. A second call to `download(_:)` for a model that is already
/// being downloaded throws `modelDownloadFailed`.
///
/// ## Background download mode
///
/// Background download mode is explicitly deferred to a follow-up. The
/// `ModelDownloadDelegate` file stays on disk but is unused for now.
public actor ModelDownloader {

    private let baseCacheDirectory: URL
    private let mode: ModelDownloadMode
    private let urlSession: URLSession
    private var inFlightDownloads: [WhisperModel: Task<Void, Never>] = [:]
    private var inFlightStreams: [WhisperModel: AsyncThrowingStream<DownloadProgress, any Error>] =
        [:]

    /// Default Application Support location used when no cache directory is supplied.
    public static var defaultCacheDirectory: URL {
        let appSupport =
            FileManager.default.urls(
                for: .applicationSupportDirectory,
                in: .userDomainMask
            ).first ?? FileManager.default.temporaryDirectory
        return
            appSupport
            .appendingPathComponent("SwiftWhisper", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
    }

    /// Foreground-mode convenience that preserves the existing API.
    ///
    /// - Parameters:
    ///   - cacheDirectory: where to store downloaded models. Pass `nil` to use
    ///     the platform default (`~/Library/Application Support/SwiftWhisper/Models/`).
    ///   - urlSession: session for all network requests. Pass a custom session
    ///     with a mock `URLProtocol` for testing.
    public init(cacheDirectory: URL? = nil, urlSession: URLSession = .shared) {
        self.init(
            cacheDirectory: cacheDirectory,
            mode: .foreground,
            urlSession: urlSession
        )
    }

    /// Mode-aware initializer.
    ///
    /// Background mode support is deferred; passing `.background` currently
    /// behaves identically to `.foreground`.
    public init(
        cacheDirectory: URL? = nil,
        mode: ModelDownloadMode,
        urlSession: URLSession? = nil
    ) {
        if let cacheDirectory {
            self.baseCacheDirectory = cacheDirectory
        } else {
            self.baseCacheDirectory = ModelDownloader.defaultCacheDirectory
        }
        self.mode = mode
        self.urlSession = urlSession ?? .shared
    }

    /// On-disk directory for a specific model variant.
    public func cacheDirectory(for model: WhisperModel) -> URL {
        baseCacheDirectory.appendingPathComponent(model.huggingFacePath, isDirectory: true)
    }

    /// Whether the model has been fully downloaded.
    ///
    /// Verifies the `.complete` marker and that `ggml-<stem>.bin` exists. A
    /// marker without the binary (e.g. from an older incomplete download) is
    /// treated as not-downloaded so the next `download(_:)` call starts fresh.
    public func isDownloaded(_ model: WhisperModel) -> Bool {
        let fm = FileManager.default
        let dir = cacheDirectory(for: model)
        let ggmlURL = dir.appendingPathComponent("\(model.huggingFacePath).bin")
        let marker = markerPath(for: model)
        guard fm.fileExists(atPath: marker.path) else { return false }
        guard fm.fileExists(atPath: ggmlURL.path) else {
            try? fm.removeItem(at: marker)
            return false
        }
        return true
    }

    /// Returns a `ModelBundle` for an already-downloaded model.
    ///
    /// Throws `modelFileMissing` if the model has not been downloaded.
    public func bundle(for model: WhisperModel) throws(SwiftWhisperError) -> ModelBundle {
        let dir = cacheDirectory(for: model)
        guard isDownloaded(model) else {
            throw .modelFileMissing("model \(model.rawValue) not downloaded")
        }
        let ggmlURL = dir.appendingPathComponent("\(model.huggingFacePath).bin")
        return ModelBundle(
            model: model,
            directory: dir,
            ggmlModelURL: ggmlURL,
            coreMLEncoderURL: nil
        )
    }

    /// Starts downloading a model.
    ///
    /// Returns a stream of progress updates.
    ///
    /// Throws `modelDownloadFailed` if a download for this model is already
    /// in flight. Completed models are skipped (the stream yields `.complete`
    /// immediately).
    public func download(
        _ model: WhisperModel
    ) throws(SwiftWhisperError) -> AsyncThrowingStream<DownloadProgress, any Error> {
        if inFlightDownloads[model] != nil {
            throw .modelDownloadFailed("download already in progress for \(model.rawValue)")
        }

        if isDownloaded(model) {
            return AsyncThrowingStream { continuation in
                continuation.yield(
                    DownloadProgress(
                        totalFiles: 0, completedFiles: 0,
                        totalBytes: 0, totalBytesDownloaded: 0,
                        phase: .complete
                    ))
                continuation.finish()
            }
        }

        let (stream, continuation) = AsyncThrowingStream<DownloadProgress, any Error>.makeStream()

        let task = Task { [weak self] in
            do {
                try await self?.performDownload(model: model, continuation: continuation)
            } catch let error as SwiftWhisperError {
                continuation.finish(throwing: error)
            } catch {
                continuation.finish(
                    throwing: SwiftWhisperError.modelDownloadFailed(error.localizedDescription))
            }
            await self?.clearInFlight(model)
        }

        inFlightDownloads[model] = task
        inFlightStreams[model] = stream
        return stream
    }

    /// Deletes the cached model directory and its marker file.
    public func delete(_ model: WhisperModel) throws {
        let dir = cacheDirectory(for: model)
        if FileManager.default.fileExists(atPath: dir.path) {
            try FileManager.default.removeItem(at: dir)
        }
    }

    // MARK: - Internals

    private func clearInFlight(_ model: WhisperModel) {
        inFlightDownloads[model] = nil
        inFlightStreams[model] = nil
    }

    private func markerPath(for model: WhisperModel) -> URL {
        cacheDirectory(for: model).appendingPathComponent(".complete")
    }

    private func performDownload(
        model: WhisperModel,
        continuation: AsyncThrowingStream<DownloadProgress, any Error>.Continuation
    ) async throws {
        let dir = cacheDirectory(for: model)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let stem = model.huggingFacePath
        let urlString =
            "https://huggingface.co/\(WhisperModel.huggingFaceRepo)/resolve/main/\(stem).bin"
        guard let url = URL(string: urlString) else {
            throw SwiftWhisperError.modelDownloadFailed("invalid URL: \(urlString)")
        }
        let destURL = dir.appendingPathComponent("\(stem).bin")

        continuation.yield(
            DownloadProgress(
                totalFiles: 1, completedFiles: 0,
                totalBytes: model.approximateSizeBytes, totalBytesDownloaded: 0,
                currentFile: "\(stem).bin",
                phase: .downloading
            )
        )

        let progressDelegate = ProgressDelegate { totalBytesWritten, totalBytesExpectedToWrite in
            let total =
                totalBytesExpectedToWrite > 0
                ? totalBytesExpectedToWrite
                : max(model.approximateSizeBytes, totalBytesWritten)
            continuation.yield(
                DownloadProgress(
                    totalFiles: 1, completedFiles: 0,
                    totalBytes: total, totalBytesDownloaded: totalBytesWritten,
                    currentFile: "\(stem).bin",
                    phase: .downloading
                )
            )
        }

        let (tempURL, response) = try await urlSession.download(
            from: url, delegate: progressDelegate
        )

        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            let code = (response as? HTTPURLResponse)?.statusCode ?? -1
            throw SwiftWhisperError.modelDownloadFailed("\(stem).bin returned HTTP \(code)")
        }

        try moveDownloaded(from: tempURL, to: destURL)

        let size: Int64 =
            (try? FileManager.default.attributesOfItem(atPath: destURL.path)[.size] as? Int64)
            ?? 0
        continuation.yield(
            DownloadProgress(
                totalFiles: 1, completedFiles: 1,
                totalBytes: size, totalBytesDownloaded: size,
                phase: .verifying
            )
        )
        FileManager.default.createFile(atPath: markerPath(for: model).path, contents: nil)
        continuation.yield(
            DownloadProgress(
                totalFiles: 1, completedFiles: 1,
                totalBytes: size, totalBytesDownloaded: size,
                phase: .complete
            )
        )
        continuation.finish()
    }

    private func moveDownloaded(from tempURL: URL, to destURL: URL) throws {
        try FileManager.default.createDirectory(
            at: destURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if FileManager.default.fileExists(atPath: destURL.path) {
            try FileManager.default.removeItem(at: destURL)
        }
        try FileManager.default.moveItem(at: tempURL, to: destURL)
    }
}

// MARK: - Background re-entry

extension ModelDownloader {

    /// No-op entry point retained for API compatibility.
    ///
    /// Background download mode is deferred to a future task.
    public static func handleBackgroundEvents(
        identifier: String,
        completion: @escaping @Sendable () -> Void
    ) async {
        completion()
    }

    /// Returns the currently in-flight background downloads keyed by model.
    ///
    /// Background mode is deferred; always returns an empty dictionary.
    public func currentBackgroundDownloads() async -> [WhisperModel: Float] {
        return [:]
    }
}

// MARK: - Progress delegate

private final class ProgressDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    private let onProgress: @Sendable (Int64, Int64) -> Void

    init(onProgress: @escaping @Sendable (Int64, Int64) -> Void) {
        self.onProgress = onProgress
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Handled by the async download(from:delegate:) call's tempURL return.
    }
}
