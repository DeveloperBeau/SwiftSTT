import CryptoKit
import Foundation
import SwiftWhisperCore

/// Downloads Whisper Core ML models from HuggingFace and caches them locally.
///
/// Uses the HuggingFace tree API to discover files inside a model directory,
/// downloads each one into Application Support, and writes a `.complete` marker
/// when all files have landed. The marker prevents `isDownloaded` from returning
/// `true` for a partially-downloaded model (e.g. after a crash or cancellation).
///
/// Guards against concurrent downloads of the same model using an in-flight
/// task dictionary. A second call to `download(_:)` for a model that is already
/// being downloaded throws `modelDownloadFailed`.
///
/// ## Foreground vs background
///
/// The default ``init(cacheDirectory:urlSession:)`` uses URLSession.shared
/// (or the injected session) and only progresses while the host process is in
/// memory. For iOS apps that need transfers to survive suspension, use
/// ``init(cacheDirectory:mode:urlSession:)`` with
/// ``ModelDownloadMode/background(identifier:)`` and forward
/// `application(_:handleEventsForBackgroundURLSession:completionHandler:)`
/// to ``handleBackgroundEvents(identifier:completion:)``.
public actor ModelDownloader {

    private let baseCacheDirectory: URL
    private let mode: ModelDownloadMode
    private let urlSession: URLSession
    /// Foreground session reused for HuggingFace tree API calls.
    ///
    /// Background `URLSessionConfiguration` instances reject `data(from:)`,
    /// so listing always goes through this session.
    private let listingSession: URLSession
    private let delegate: ModelDownloadDelegate?
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

    /// Foreground-mode convenience that preserves the M4 API.
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
    /// When `mode == .background(let id)`, the downloader builds a
    /// `URLSessionConfiguration.background(withIdentifier: id)` with
    /// `sessionSendsLaunchEvents = true` and registers a delegate that bridges
    /// system callbacks back into the async stream API.
    ///
    /// When `mode == .foreground`, the supplied `urlSession` (or
    /// `URLSession.shared`) handles file transfers directly.
    ///
    /// Pass a non-nil `urlSession` to override the mode default (test injection
    /// of a `URLProtocol`-mocked session works for both modes).
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

        switch mode {
        case .foreground:
            self.delegate = nil
            self.urlSession = urlSession ?? .shared
            self.listingSession = urlSession ?? .shared
        case .background(let identifier):
            let bridge = ModelDownloadDelegate()
            self.delegate = bridge
            ModelDownloadDelegate.register(bridge, for: identifier)
            if let urlSession {
                self.urlSession = urlSession
            } else {
                let config = URLSessionConfiguration.background(withIdentifier: identifier)
                config.sessionSendsLaunchEvents = true
                config.isDiscretionary = false
                let session = URLSession(
                    configuration: config,
                    delegate: bridge,
                    delegateQueue: nil
                )
                session.sessionDescription = "swiftwhisper.\(identifier)"
                self.urlSession = session
            }
            self.listingSession = URLSession(configuration: .ephemeral)
        }
    }

    /// On-disk directory for a specific model variant.
    public func cacheDirectory(for model: WhisperModel) -> URL {
        baseCacheDirectory.appendingPathComponent(model.huggingFacePath, isDirectory: true)
    }

    /// Whether the model has been fully downloaded.
    ///
    /// Verifies both the `.complete` marker and the presence of the encoder,
    /// decoder, and tokenizer files. A marker without the expected files
    /// (e.g. from an older buggy download) is treated as not-downloaded so
    /// the next call to `download(_:)` redownloads cleanly.
    public func isDownloaded(_ model: WhisperModel) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: markerPath(for: model).path) else { return false }
        let dir = cacheDirectory(for: model)
        let required = [
            dir.appendingPathComponent("AudioEncoder.mlmodelc", isDirectory: true),
            dir.appendingPathComponent("TextDecoder.mlmodelc", isDirectory: true),
            dir.appendingPathComponent("tokenizer.json"),
        ]
        for url in required where !fm.fileExists(atPath: url.path) {
            try? fm.removeItem(at: markerPath(for: model))
            return false
        }
        return true
    }

    /// Returns a `ModelBundle` for an already-downloaded model.
    ///
    /// Throws `modelFileMissing` if the model has not been downloaded or
    /// expected files are absent.
    public func bundle(for model: WhisperModel) throws(SwiftWhisperError) -> ModelBundle {
        let dir = cacheDirectory(for: model)
        guard isDownloaded(model) else {
            throw .modelFileMissing("model \(model.rawValue) not downloaded")
        }
        let encoderURL = dir.appendingPathComponent("AudioEncoder.mlmodelc", isDirectory: true)
        let decoderURL = dir.appendingPathComponent("TextDecoder.mlmodelc", isDirectory: true)
        let tokenizerURL = dir.appendingPathComponent("tokenizer.json")

        for (name, url) in [
            ("AudioEncoder.mlmodelc", encoderURL), ("TextDecoder.mlmodelc", decoderURL),
            ("tokenizer.json", tokenizerURL),
        ] {
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw .modelFileMissing(name)
            }
        }
        return ModelBundle(
            model: model,
            directory: dir,
            encoderURL: encoderURL,
            decoderURL: decoderURL,
            tokenizerURL: tokenizerURL
        )
    }

    /// Starts downloading a model.
    ///
    /// Returns a stream of progress updates.
    ///
    /// Throws `modelDownloadFailed` if a download for this model is already
    /// in flight. Completed models are skipped (the stream yields `.complete`
    /// immediately).
    ///
    /// In background mode, if a system-managed task for the same URL is
    /// already running (e.g. left over from a previous app launch), the
    /// existing task is adopted rather than restarted.
    public func download(
        _ model: WhisperModel
    ) throws(SwiftWhisperError) -> AsyncThrowingStream<DownloadProgress, any Error> {
        if inFlightDownloads[model] != nil {
            if case .background = mode, let existing = inFlightStreams[model] {
                return existing
            }
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

        continuation.yield(
            DownloadProgress(
                totalFiles: 0, completedFiles: 0,
                totalBytes: 0, totalBytesDownloaded: 0,
                phase: .listing
            ))

        let files = try await listFiles(model: model)
        let totalBytes = files.reduce(Int64(0)) { $0 + $1.size }
        var bytesDownloaded: Int64 = 0

        for (index, file) in files.enumerated() {
            try Task.checkCancellation()
            continuation.yield(
                DownloadProgress(
                    totalFiles: files.count, completedFiles: index,
                    totalBytes: totalBytes, totalBytesDownloaded: bytesDownloaded,
                    currentFile: file.name,
                    phase: .downloading
                ))

            try await downloadFile(file, to: dir)
            bytesDownloaded += file.size
        }

        try await downloadTokenizer(for: model, to: dir)

        continuation.yield(
            DownloadProgress(
                totalFiles: files.count, completedFiles: files.count,
                totalBytes: totalBytes, totalBytesDownloaded: bytesDownloaded,
                phase: .verifying
            ))
        FileManager.default.createFile(atPath: markerPath(for: model).path, contents: nil)

        continuation.yield(
            DownloadProgress(
                totalFiles: files.count, completedFiles: files.count,
                totalBytes: totalBytes, totalBytesDownloaded: bytesDownloaded,
                phase: .complete
            ))
        continuation.finish()
    }

    // MARK: - HuggingFace API

    struct HFFile {
        let name: String
        let relativePath: String
        let size: Int64
        let sha256: String?
    }

    private func listFiles(model: WhisperModel) async throws -> [HFFile] {
        let repo = WhisperModel.huggingFaceRepo
        let path = model.huggingFacePath
        let urlString =
            "https://huggingface.co/api/models/\(repo)/tree/main/\(path)?recursive=true"
        guard let url = URL(string: urlString) else {
            throw SwiftWhisperError.modelDownloadFailed(
                "invalid HuggingFace URL for \(model.rawValue)")
        }

        let (data, response) = try await listingSession.data(from: url)
        guard let http = response as? HTTPURLResponse, http.statusCode == 200 else {
            throw SwiftWhisperError.modelDownloadFailed(
                "HuggingFace tree API returned \((response as? HTTPURLResponse)?.statusCode ?? -1)"
            )
        }

        guard let items = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else {
            throw SwiftWhisperError.modelDownloadFailed("unexpected tree API response format")
        }

        return items.compactMap { item -> HFFile? in
            guard
                let type = item["type"] as? String, type == "file",
                let rfilename = item["rfilename"] as? String
            else { return nil }
            let size = (item["size"] as? Int64) ?? 0
            let sha: String? = (item["lfs"] as? [String: Any])?["oid"] as? String
            let name = (rfilename as NSString).lastPathComponent
            return HFFile(name: name, relativePath: rfilename, size: size, sha256: sha)
        }
    }

    /// Fetches the upstream OpenAI `tokenizer.json` for the model and places it
    /// in the model's cache directory root. argmaxinc's Core ML repo doesn't
    /// ship tokenizer files, so they have to come from the original OpenAI repo.
    private func downloadTokenizer(for model: WhisperModel, to directory: URL) async throws {
        let destURL = directory.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: destURL.path) { return }
        let tokenizerURL = model.tokenizerDownloadURL
        let (tempURL, response) = try await listingSession.download(from: tokenizerURL)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw SwiftWhisperError.modelDownloadFailed(
                "tokenizer.json download returned \((response as? HTTPURLResponse)?.statusCode ?? -1)"
            )
        }
        try moveDownloaded(from: tempURL, to: destURL)
    }

    private func downloadFile(_ file: HFFile, to directory: URL) async throws {
        let repo = WhisperModel.huggingFaceRepo
        guard
            let downloadURL = URL(
                string: "https://huggingface.co/\(repo)/resolve/main/\(file.relativePath)"
            )
        else {
            throw SwiftWhisperError.modelDownloadFailed("invalid download URL for \(file.name)")
        }

        let destURL = destinationURL(for: file, in: directory)

        switch mode {
        case .foreground:
            try await downloadFileForeground(file, from: downloadURL, to: destURL)
        case .background:
            try await downloadFileBackground(file, from: downloadURL, to: destURL)
        }
    }

    private func downloadFileForeground(_ file: HFFile, from downloadURL: URL, to destURL: URL)
        async throws
    {
        let (tempURL, response) = try await urlSession.download(from: downloadURL)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw SwiftWhisperError.modelDownloadFailed(
                "download \(file.name) returned \((response as? HTTPURLResponse)?.statusCode ?? -1)"
            )
        }

        if let expected = file.sha256 {
            let fileData = try Data(contentsOf: tempURL)
            let actual = SHA256.hash(data: fileData)
                .map { String(format: "%02x", $0) }
                .joined()
            if actual != expected {
                throw SwiftWhisperError.modelChecksumMismatch(file: file.name)
            }
        }

        try moveDownloaded(from: tempURL, to: destURL)
    }

    private func downloadFileBackground(_ file: HFFile, from downloadURL: URL, to destURL: URL)
        async throws
    {
        guard let delegate else {
            throw SwiftWhisperError.modelDownloadFailed(
                "background mode without delegate (illegal state)")
        }

        let existing = await findExistingTask(for: downloadURL)
        let task = existing ?? urlSession.downloadTask(with: downloadURL)

        // Per-file progress stream is held only to satisfy the delegate's
        // TaskHandle contract; the outer download loop emits its own
        // file-granularity progress so this continuation is intentionally drained.
        let (sinkStream, sinkContinuation) = AsyncThrowingStream<DownloadProgress, any Error>
            .makeStream()
        let drain = Task { for try await _ in sinkStream {} }

        let finishedURL: URL = try await withCheckedThrowingContinuation {
            (cont: CheckedContinuation<URL, any Error>) in
            let handle = ModelDownloadDelegate.TaskHandle(
                continuation: sinkContinuation,
                destination: destURL,
                finished: { result in
                    switch result {
                    case .success(let url): cont.resume(returning: url)
                    case .failure(let error): cont.resume(throwing: error)
                    }
                }
            )
            delegate.register(taskIdentifier: task.taskIdentifier, handle: handle)
            if existing == nil {
                task.resume()
            }
        }

        try? await drain.value

        if let expected = file.sha256 {
            let fileData = try Data(contentsOf: finishedURL)
            let actual = SHA256.hash(data: fileData)
                .map { String(format: "%02x", $0) }
                .joined()
            if actual != expected {
                throw SwiftWhisperError.modelChecksumMismatch(file: file.name)
            }
        }
    }

    private func destinationURL(for file: HFFile, in directory: URL) -> URL {
        let destPath = file.relativePath
            .split(separator: "/")
            .dropFirst()
        if destPath.count > 1 {
            let subdir = destPath.dropLast().reduce(directory) {
                $0.appendingPathComponent(String($1), isDirectory: true)
            }
            return subdir.appendingPathComponent(String(destPath.last!))
        }
        return directory.appendingPathComponent(file.name)
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

    private func findExistingTask(for url: URL) async -> URLSessionDownloadTask? {
        await withCheckedContinuation { cont in
            urlSession.getAllTasks { tasks in
                let match = tasks.first { task in
                    guard task is URLSessionDownloadTask else { return false }
                    return task.originalRequest?.url == url && task.state == .running
                }
                cont.resume(returning: match as? URLSessionDownloadTask)
            }
        }
    }
}

// MARK: - Background re-entry

extension ModelDownloader {

    /// Stashes the system completion handler delivered by
    /// `application(_:handleEventsForBackgroundURLSession:completionHandler:)`.
    ///
    /// The handler fires once the URLSession reports
    /// `urlSessionDidFinishEvents(forBackgroundURLSession:)` for the matching
    /// identifier. Calling with an unknown identifier is a no-op so apps can
    /// safely route every relaunch event without checking session names first.
    public static func handleBackgroundEvents(
        identifier: String,
        completion: @escaping @Sendable () -> Void
    ) async {
        guard let delegate = ModelDownloadDelegate.delegate(for: identifier) else {
            completion()
            return
        }
        delegate.stash(completion: completion)
    }

    /// Returns the currently in-flight background downloads keyed by model.
    ///
    /// Useful on app relaunch to surface progress for downloads that started
    /// before suspension. Returns an empty dictionary in foreground mode.
    public func currentBackgroundDownloads() async -> [WhisperModel: Float] {
        guard case .background = mode else { return [:] }
        let tasks = await allTasks()
        var result: [WhisperModel: Float] = [:]
        for task in tasks {
            guard
                let url = task.originalRequest?.url?.absoluteString,
                let model = WhisperModel.allCases.first(where: { url.contains($0.huggingFacePath) })
            else { continue }
            let expected = task.countOfBytesExpectedToReceive
            let received = task.countOfBytesReceived
            let fraction: Float
            if expected > 0 {
                fraction = Float(received) / Float(expected)
            } else {
                fraction = 0
            }
            // Keep the highest fraction seen for the model (a model spans many files).
            result[model] = max(result[model] ?? 0, fraction)
        }
        return result
    }

    private func allTasks() async -> [URLSessionTask] {
        await withCheckedContinuation { cont in
            urlSession.getAllTasks { tasks in
                cont.resume(returning: tasks)
            }
        }
    }
}
