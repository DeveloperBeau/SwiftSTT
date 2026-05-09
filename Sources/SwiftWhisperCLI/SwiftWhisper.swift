import ArgumentParser
import Foundation

/// Root command for the `swiftwhisper` binary.
///
/// Subcommands:
/// - ``DownloadCommand`` fetches a model from HuggingFace.
/// - ``ListModelsCommand`` shows known models and their cache status.
/// - ``TranscribeCommand`` runs the streaming transcription pipeline against a file.
/// - ``InfoCommand`` shows the on-disk paths and download status of a single model.
///
/// All subcommands share the same Application Support cache directory; the
/// `--cache-dir` override is intentionally deferred to a future milestone.
@main
struct SwiftWhisper: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "swiftwhisper",
        abstract: "On-device speech-to-text powered by Whisper Core ML models.",
        version: "0.7.0",
        subcommands: [
            DownloadCommand.self,
            ListModelsCommand.self,
            TranscribeCommand.self,
            InfoCommand.self,
        ]
    )
}
