import Foundation

/// Snapshot of a model download's progress, emitted by the download stream.
public struct DownloadProgress: Sendable, Equatable {

    public enum Phase: String, Sendable, Equatable {
        case listing
        case downloading
        case verifying
        case complete
    }

    /// Total number of files to download for the model.
    public let totalFiles: Int

    /// How many files have finished downloading so far.
    public let completedFiles: Int

    /// Total expected download size in bytes (from the file listing).
    public let totalBytes: Int64

    /// Cumulative bytes downloaded so far across all files.
    public let totalBytesDownloaded: Int64

    /// Name of the file currently being downloaded, if any.
    public let currentFile: String?

    /// Which stage of the download pipeline we are in.
    public let phase: Phase

    public init(
        totalFiles: Int,
        completedFiles: Int,
        totalBytes: Int64,
        totalBytesDownloaded: Int64,
        currentFile: String? = nil,
        phase: Phase
    ) {
        self.totalFiles = totalFiles
        self.completedFiles = completedFiles
        self.totalBytes = totalBytes
        self.totalBytesDownloaded = totalBytesDownloaded
        self.currentFile = currentFile
        self.phase = phase
    }

    /// Fraction complete as a value between 0.0 and 1.0. Returns 0 when
    /// `totalBytes` is zero to avoid division by zero on empty listings.
    public var fractionComplete: Double {
        guard totalBytes > 0 else { return 0 }
        return Double(totalBytesDownloaded) / Double(totalBytes)
    }
}
