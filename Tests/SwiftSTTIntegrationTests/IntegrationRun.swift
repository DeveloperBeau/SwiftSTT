import Foundation
import SwiftSTTCore
import SwiftSTTKit
import Testing

/// Parent of the suites that use the real model cache.
///
/// Its suites run one after another, never at the same time, so they never download
/// into the same cache at once.
@Suite("Integration run", .serialized)
enum IntegrationRun {

    /// Whether the model file and its `.complete` marker are in the default cache.
    ///
    /// Read with a plain file check, not through the downloader, before a suite
    /// prepares the model.
    static func modelIsInDefaultCache(_ model: WhisperModel) -> Bool {
        let directory = ModelDownloader.defaultCacheDirectory
            .appendingPathComponent(model.fileStem, isDirectory: true)
        let fileManager = FileManager.default
        return fileManager.fileExists(
            atPath: directory.appendingPathComponent("\(model.fileStem).bin").path)
            && fileManager.fileExists(atPath: directory.appendingPathComponent(".complete").path)
    }
}
