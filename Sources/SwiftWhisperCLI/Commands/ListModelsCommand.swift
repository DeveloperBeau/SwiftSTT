import ArgumentParser
import Foundation
import SwiftWhisperCore
import SwiftWhisperKit

/// Lists all known Whisper models and shows whether each one has been
/// downloaded into the default Application Support cache.
///
/// Both this command and `download` consult the same default cache directory.
/// The `--cache-dir` flag is intentionally deferred; tests use a custom
/// downloader instance directly.
struct ListModelsCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "list-models",
        abstract: "List known Whisper models and their cache status."
    )

    func run() async throws {
        let downloader = ModelDownloader()
        for model in WhisperModel.allCases {
            let downloaded = await downloader.isDownloaded(model)
            let status = downloaded ? "[downloaded]" : "[not downloaded]"
            print("\(model.rawValue) \(status)")
        }
    }
}
