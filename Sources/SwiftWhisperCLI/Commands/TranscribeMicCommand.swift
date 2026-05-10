import ArgumentParser
@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import SwiftWhisperKit

/// Live-mic transcription. Uses ``AVMicrophoneInput`` instead of the file
/// reader and stops on Ctrl+C (SIGINT) or `--max-duration` timeout.
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
                "model '\(model.rawValue)' is not downloaded. Run 'swiftwhisper download \(model.rawValue)' first."
            )
        }
        let bundle = try await downloader.bundle(for: model)

        let loader = ModelLoader()
        let loaded = try await loader.loadBundle(bundle)
        let tokenizer = try WhisperTokenizer(contentsOf: loaded.tokenizerURL)
        let encoder = WhisperEncoder(runner: MLModelRunner(model: loaded.encoder))
        let decoder = WhisperDecoder(
            runner: MLStateModelRunner(model: loaded.decoder),
            tokenizer: tokenizer
        )
        let melSpectrogram = try MelSpectrogram()
        let vad = EnergyVAD()
        let policy = LocalAgreementPolicy()

        var options = DecodingOptions.default
        options.language = language

        let audioInput = AVMicrophoneInput()
        let pipeline = TranscriptionPipeline(
            audioInput: audioInput,
            vad: vad,
            melSpectrogram: melSpectrogram,
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: policy,
            options: options
        )

        let formatter = SegmentFormatters.make(format)
        let buffering = formatter.bufferingRequired
        let collector = SegmentCollector()

        let destination: OutputDestination
        if let output {
            destination = try OutputDestination.file(at: output, noClobber: noClobber)
        } else {
            destination = .stdout()
        }
        defer { destination.close() }

        SignalHandler.reset()
        SignalHandler.installSIGINT()
        defer { SignalHandler.uninstallSIGINT() }

        if let header = formatter.header() {
            destination.writeLine(header)
        }

        writeStderr("Listening. Press Ctrl+C to stop.\n")

        let stream: AsyncStream<TranscriptionSegment>
        do {
            stream = try await pipeline.start()
        } catch SwiftWhisperError.micPermissionDenied {
            Self.printPermissionGuidance()
            throw ExitCode(Self.permissionDeniedExitCode)
        } catch {
            throw ValidationError("microphone start failed: \(error)")
        }

        let consumer = Task { [collector, formatter] in
            var index = 0
            for await segment in stream {
                if buffering {
                    await collector.append(segment)
                } else {
                    let line = formatter.format(segment: segment, index: index)
                    if !line.isEmpty {
                        destination.writeLine(line)
                    }
                    index += 1
                }
            }
        }

        let watchdog = Task {
            await Self.watchUntilStop(maxDuration: maxDuration)
        }

        await watchdog.value
        await pipeline.stop()
        await consumer.value

        if buffering {
            let segments = await collector.snapshot()
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

    /// `EX_NOPERM` from `sysexits.h`. Distinct from the generic `1` so callers
    /// (e.g. shell scripts wrapping the CLI) can detect a permission failure.
    static let permissionDeniedExitCode: Int32 = 77

    /// Polls ``SignalHandler/isStopRequested()`` every 100 ms and races it
    /// against an optional `--max-duration` deadline. Returns when either
    /// fires.
    private static func watchUntilStop(maxDuration: Double?) async {
        let pollNs: UInt64 = 100_000_000
        let deadline: Date? = maxDuration.map { Date().addingTimeInterval($0) }
        while !SignalHandler.isStopRequested() {
            if let deadline, Date() >= deadline { return }
            try? await Task.sleep(nanoseconds: pollNs)
        }
    }

    /// Prints a multi-line guidance message to stderr describing how to grant
    /// microphone access on each platform. Called only on a confirmed
    /// `micPermissionDenied`.
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
