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
public actor ModelDownloader {

    private let baseCacheDirectory: URL
    private let urlSession: URLSession
    private var inFlightDownloads: [WhisperModel: Task<Void, Never>] = [:]

    /// Creates a downloader.
    ///
    /// - Parameters:
    ///   - cacheDirectory: where to store downloaded models. Pass `nil` to use
    ///     the platform default (`~/Library/Application Support/SwiftWhisper/Models/`).
    ///   - urlSession: session for all network requests. Pass a custom session
    ///     with a mock `URLProtocol` for testing.
    public init(cacheDirectory: URL? = nil, urlSession: URLSession = .shared) {
        if let cacheDirectory {
            self.baseCacheDirectory = cacheDirectory
        } else {
            let appSupport = FileManager.default.urls(
                for: .applicationSupportDirectory,
                in: .userDomainMask
            ).first!
            self.baseCacheDirectory = appSupport
                .appendingPathComponent("SwiftWhisper", isDirectory: true)
                .appendingPathComponent("Models", isDirectory: true)
        }
        self.urlSession = urlSession
    }

    /// On-disk directory for a specific model variant.
    public func cacheDirectory(for model: WhisperModel) -> URL {
        baseCacheDirectory.appendingPathComponent(model.huggingFacePath, isDirectory: true)
    }

    /// Whether the model has been fully downloaded (marker file exists).
    public func isDownloaded(_ model: WhisperModel) -> Bool {
        FileManager.default.fileExists(atPath: markerPath(for: model).path)
    }

    /// Returns a `ModelBundle` for an already-downloaded model.
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

        for (name, url) in [("AudioEncoder.mlmodelc", encoderURL), ("TextDecoder.mlmodelc", decoderURL), ("tokenizer.json", tokenizerURL)] {
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

    /// Starts downloading a model. Returns a stream of progress updates.
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
                continuation.yield(DownloadProgress(
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
                continuation.finish(throwing: SwiftWhisperError.modelDownloadFailed(error.localizedDescription))
            }
            await self?.clearInFlight(model)
        }

        inFlightDownloads[model] = task
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

        // 1. List files via HuggingFace tree API.
        continuation.yield(DownloadProgress(
            totalFiles: 0, completedFiles: 0,
            totalBytes: 0, totalBytesDownloaded: 0,
            phase: .listing
        ))

        let files = try await listFiles(model: model)
        let totalBytes = files.reduce(Int64(0)) { $0 + $1.size }
        var bytesDownloaded: Int64 = 0

        // 2. Download each file.
        for (index, file) in files.enumerated() {
            try Task.checkCancellation()
            continuation.yield(DownloadProgress(
                totalFiles: files.count, completedFiles: index,
                totalBytes: totalBytes, totalBytesDownloaded: bytesDownloaded,
                currentFile: file.name,
                phase: .downloading
            ))

            try await downloadFile(file, to: dir)
            bytesDownloaded += file.size
        }

        // 3. Write completion marker.
        continuation.yield(DownloadProgress(
            totalFiles: files.count, completedFiles: files.count,
            totalBytes: totalBytes, totalBytesDownloaded: bytesDownloaded,
            phase: .verifying
        ))
        FileManager.default.createFile(atPath: markerPath(for: model).path, contents: nil)

        continuation.yield(DownloadProgress(
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
        let url = URL(string: "https://huggingface.co/api/models/\(repo)/tree/main/\(path)")!

        let (data, response) = try await urlSession.data(from: url)
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

    private func downloadFile(_ file: HFFile, to directory: URL) async throws {
        let repo = WhisperModel.huggingFaceRepo
        let downloadURL = URL(
            string: "https://huggingface.co/\(repo)/resolve/main/\(file.relativePath)"
        )!

        let (tempURL, response) = try await urlSession.download(from: downloadURL)
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw SwiftWhisperError.modelDownloadFailed(
                "download \(file.name) returned \((response as? HTTPURLResponse)?.statusCode ?? -1)"
            )
        }

        // Verify checksum if available.
        if let expected = file.sha256 {
            let fileData = try Data(contentsOf: tempURL)
            let actual = SHA256.hash(data: fileData)
                .map { String(format: "%02x", $0) }
                .joined()
            if actual != expected {
                throw SwiftWhisperError.modelChecksumMismatch(file: file.name)
            }
        }

        // Create subdirectory structure if needed (e.g. `AudioEncoder.mlmodelc/weights/`).
        let destPath = file.relativePath
            .split(separator: "/")
            .dropFirst() // drop model directory prefix repeated in relativePath
        let destURL: URL
        if destPath.count > 1 {
            let subdir = destPath.dropLast().reduce(directory) { $0.appendingPathComponent(String($1), isDirectory: true) }
            try FileManager.default.createDirectory(at: subdir, withIntermediateDirectories: true)
            destURL = subdir.appendingPathComponent(String(destPath.last!))
        } else {
            destURL = directory.appendingPathComponent(file.name)
        }

        if FileManager.default.fileExists(atPath: destURL.path) {
            try FileManager.default.removeItem(at: destURL)
        }
        try FileManager.default.moveItem(at: tempURL, to: destURL)
    }
}
