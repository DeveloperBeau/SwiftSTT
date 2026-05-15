import ArgumentParser
import Foundation

/// Root command for the `swiftstt` binary.
///
/// Subcommands:
/// - ``DownloadCommand`` fetches a model from HuggingFace.
/// - ``ListModelsCommand`` shows known models and their cache status.
/// - ``TranscribeCommand`` runs the streaming transcription pipeline against
///   one or more files. Supports text/srt/vtt/json output formats.
/// - ``TranscribeMicCommand`` runs the pipeline against live microphone audio
///   until Ctrl+C or `--max-duration`.
/// - ``InfoCommand`` shows the on-disk paths and download status of a single
///   model.
///
/// Subcommands that touch the model cache accept `--cache-dir` to override
/// the default Application Support location.
@main
struct SwiftSTT: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "swiftstt",
        abstract: "On-device speech-to-text powered by Whisper Core ML models.",
        version: "0.13.0",
        subcommands: [
            DownloadCommand.self,
            ListModelsCommand.self,
            TranscribeCommand.self,
            TranscribeMicCommand.self,
            InfoCommand.self,
        ]
    )
}
