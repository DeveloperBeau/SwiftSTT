import ArgumentParser
import Foundation
import SwiftSTTCore
import SwiftSTTKit

/// Lists all known Whisper models and shows whether each one has been
/// downloaded into the configured cache.
///
/// `--cache-dir` overrides the default Application Support location.
struct ListModelsCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "list-models",
        abstract: "List known Whisper models and their cache status."
    )

    @Option(
        name: .long,
        help:
            "Override the model cache directory. Defaults to ~/Library/Application Support/SwiftSTT/Models."
    )
    var cacheDir: String?

    func run() async throws {
        let downloader = ModelDownloader(cacheDirectory: CacheDirectoryOption.resolve(cacheDir))
        for model in WhisperModel.allCases {
            let downloaded = await downloader.isDownloaded(model)
            let status = downloaded ? "[downloaded]" : "[not downloaded]"
            print("\(model.rawValue) \(status)")
        }
    }
}
