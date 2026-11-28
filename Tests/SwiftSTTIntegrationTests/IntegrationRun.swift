import Foundation
import SwiftSTTCore
import SwiftSTTKit
import Testing

/// Parent of the suites that use the real model cache.
///
/// Its suites run one after another, never at the same time, so they never download
/// into the same cache at once. The time limit covers each suite's `init` as well as
/// its test, so a stalled model download fails by name.
@Suite("Integration run", .serialized, .timeLimit(.minutes(10)))
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
