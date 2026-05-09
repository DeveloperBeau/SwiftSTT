import ArgumentParser
@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import SwiftWhisperKit

/// Runs the streaming transcription pipeline against a single audio file and
/// prints each confirmed segment to stdout.
///
/// Format: `[HH:MM:SS -> HH:MM:SS] text` per segment, ASCII only.
///
/// The model bundle must already be on disk; this command does not auto-download.
/// On a missing bundle the user gets a `swiftwhisper download <model>` hint.
struct TranscribeCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe a local audio file using a downloaded model."
    )

    @Argument(help: "Path to the audio file to transcribe (WAV recommended).")
    var audioFile: String

    @Option(name: .shortAndLong, help: "Model to use (default: base).")
    var model: WhisperModel = .base

    @Option(name: .shortAndLong, help: "ISO-639-1 language code, e.g. 'en'. Auto-detect when omitted.")
    var language: String?

    func validate() throws {
        let url = URL(fileURLWithPath: audioFile)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw ValidationError("audio file not found: \(audioFile)")
        }
    }

    func run() async throws {
        let url = URL(fileURLWithPath: audioFile)

        let downloader = ModelDownloader()
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

        let audioInput = AudioFileInput(fileURL: url)
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

        let stream = try await pipeline.start()
        let consumer = Task {
            for await segment in stream {
                let (start, end) = TimeFormatter.format(start: segment.start, end: segment.end)
                print("[\(start) -> \(end)] \(segment.text)")
            }
        }

        await audioInput.waitUntilComplete()
        await pipeline.stop()
        await consumer.value
    }
}
