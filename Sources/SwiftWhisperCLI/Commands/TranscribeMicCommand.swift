import ArgumentParser
import Foundation
import SwiftWhisperCore
import SwiftWhisperKit

/// Live-mic transcription.
///
/// Uses `WhisperCppEngine` for record-then-transcribe. Recording stops on
/// Ctrl+C (SIGINT) or `--max-duration`, then transcription runs once on stop.
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
        let downloader = ModelDownloader(cacheDirectory: CacheDirectoryOption.resolve(cacheDir))
        guard await downloader.isDownloaded(model) else {
            throw ValidationError(
                "model '\(model.rawValue)' is not downloaded."
                    + " Run 'swiftwhisper download \(model.rawValue)' first."
            )
        }

        // Point the engine's storage at the requested model for this session,
        // then restore it on exit so we don't clobber the user's persisted
        // default.
        let storage = DefaultModelStorage()
        let savedModel = storage.model
        storage.model = model
        defer { storage.model = savedModel }

        let engine = WhisperCppEngine(storage: storage)
        await engine.prepare()

        let formatter = SegmentFormatters.make(format)
        let buffering = formatter.bufferingRequired
        let micCollector = MicSegmentCollector()

        let destination: OutputDestination
        if let output {
            destination = try OutputDestination.file(at: output, noClobber: noClobber)
        } else {
            destination = .stdout()
        }
        defer { destination.close() }

        // Subscribe to the segment stream BEFORE starting so we don't miss any
        // segments emitted during stop().
        let segStream = engine.segmentStream()

        // Consume segments in a background Task while recording is active.
        // For buffered formats we accumulate; for streaming formats we emit live.
        var segmentIndex = 0
        let consumer = Task { [micCollector] in
            for await segment in segStream {
                if buffering {
                    await micCollector.append(segment)
                } else {
                    let index = segmentIndex
                    let line = formatter.format(segment: segment, index: index)
                    if !line.isEmpty {
                        destination.writeLine(line)
                    }
                    segmentIndex += 1
                }
            }
        }

        SignalHandler.reset()
        SignalHandler.installSIGINT()
        defer { SignalHandler.uninstallSIGINT() }

        if let header = formatter.header() {
            destination.writeLine(header)
        }

        writeStderr("Listening. Press Ctrl+C to stop.\n")

        do {
            try await engine.start()
        } catch SwiftWhisperError.micPermissionDenied {
            consumer.cancel()
            Self.printPermissionGuidance()
            throw ExitCode(Self.permissionDeniedExitCode)
        } catch {
            consumer.cancel()
            throw ValidationError("microphone start failed: \(error)")
        }

        // Poll until Ctrl+C or --max-duration fires.
        await Self.watchUntilStop(maxDuration: maxDuration)

        // Stop the engine; this triggers whisper_full internally and emits
        // all segments before returning.
        await engine.stop()

        // engine.stop() finishes the segment stream so consumer's
        // for-await loop terminates naturally; wait for it.
        await consumer.value

        if buffering {
            let segments = await micCollector.snapshot()
            if formatter is JSONFormatter {
                if let payload = SegmentRendering.encodeJSON(
                    perFile: [(path: "<microphone>", segments: segments)],
                    isBatch: false
                ) {
                    destination.writeLine(payload)
                }
            } else if let body = formatter.footer(segments: segments) {
                destination.writeLine(body)
            }
        }
    }

    /// `EX_NOPERM` from `sysexits.h`.
    ///
    /// Distinct from the generic `1` so callers (e.g. shell scripts wrapping the CLI) can detect a permission failure.
    static let permissionDeniedExitCode: Int32 = 77

    /// Polls `SignalHandler.isStopRequested()` every 100 ms and races it
    /// against an optional `--max-duration` deadline.
    private static func watchUntilStop(maxDuration: Double?) async {
        let pollNs: UInt64 = 100_000_000
        let deadline: Date? = maxDuration.map { Date().addingTimeInterval($0) }
        while !SignalHandler.isStopRequested() {
            if let deadline, Date() >= deadline { return }
            try? await Task.sleep(nanoseconds: pollNs)
        }
    }

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

/// Collects transcription segments from the live-mic pipeline without
/// violating `Sendable` capture rules across concurrent closures.
actor MicSegmentCollector {
    private var segments: [TranscriptionSegment] = []

    func append(_ segment: TranscriptionSegment) {
        segments.append(segment)
    }

    func snapshot() -> [TranscriptionSegment] {
        segments
    }
}
