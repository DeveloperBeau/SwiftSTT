import ArgumentParser
import Foundation
import SwiftWhisperCore
import SwiftWhisperKit

/// Live-mic transcription.
///
/// Uses ``AVMicrophoneInput`` instead of the file reader and stops on Ctrl+C (SIGINT) or `--max-duration` timeout.
///
/// > Note: Microphone access requires `NSMicrophoneUsageDescription` in the
/// > host binary's `Info.plist`. The bare `swift run` binary does not have
/// > one; the first run will be denied. The command surfaces a friendly
/// > guidance message and exits with code 77 (`EX_NOPERM`) when permission
/// > is denied. To grant access, package the binary inside a `.app` bundle
/// > with the key set, or grant access via `tccutil` for the parent terminal.
///
/// Output formats match `transcribe` (text, srt, vtt, json, ndjson, ttml,
/// sbv). Buffered formats (json, ttml) flush only at shutdown.
///
/// `-o, --output <path>` writes the transcript to a file instead of stdout.
///
/// TODO(Task 8): Rewrite to use `WhisperCppEngine` directly.
struct TranscribeMicCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "transcribe-mic",
        abstract: "Transcribe live microphone audio until Ctrl+C or --max-duration."
    )

    @Option(name: .shortAndLong, help: "Model to use (default: base).")
    var model: WhisperModel = .base

    @Option(
        name: .shortAndLong, help: "ISO-639-1 language code, e.g. 'en'. Auto-detect when omitted.")
    var language: String?

    @Option(
        name: .shortAndLong,
        help: "Output format: text, srt, vtt, json, ndjson, ttml, sbv (default: text).")
    var format: OutputFormat = .text

    @Option(name: .long, help: "Stop after this many seconds. Unbounded when omitted.")
    var maxDuration: Double?

    @Option(
        name: [.short, .long],
        help: "Write transcript to <path> instead of stdout. Tilde paths are expanded.")
    var output: String?

    @Flag(name: .long, help: "Refuse to overwrite an existing --output file.")
    var noClobber: Bool = false

    @Option(
        name: .long,
        help:
            "Override the model cache directory. Honors SWIFTWHISPER_CACHE_DIR; default ~/Library/Application Support/SwiftWhisper/Models."
    )
    var cacheDir: String?

    func run() async throws {
        // TODO(Task 8): Rewrite this command to use WhisperCppEngine directly.
        // The Core ML pipeline (ModelLoader, WhisperEncoder, WhisperDecoder) has been
        // removed. Task 8 will replace this body with a whisper.cpp-based implementation.
        throw ValidationError(
            "transcribe-mic requires the Task 8 rewrite. Use 'swiftwhisper transcribe' for now."
        )
    }

    /// `EX_NOPERM` from `sysexits.h`.
    ///
    /// Distinct from the generic `1` so callers (e.g. shell scripts wrapping the CLI) can detect a permission failure.
    static let permissionDeniedExitCode: Int32 = 77

    /// Prints a multi-line guidance message to stderr describing how to grant
    /// microphone access on each platform.
    static func printPermissionGuidance() {
        writeStderr(
            """
            Microphone access denied.

            SwiftWhisper needs microphone access to transcribe live audio.

            On macOS:
              System Settings -> Privacy & Security -> Microphone
              Add the terminal app you ran swiftwhisper from.

            On iOS:
              Settings -> Privacy & Security -> Microphone -> [App Name]

            Re-run after granting access.

            """
        )
    }
}
