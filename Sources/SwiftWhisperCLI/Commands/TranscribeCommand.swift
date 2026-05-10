import ArgumentParser
@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import SwiftWhisperKit

/// Runs the streaming transcription pipeline against one or more audio files
/// and prints the resulting segments to stdout in the requested format.
///
/// Supported formats (`--format`):
/// - `text` (default): `[HH:MM:SS -> HH:MM:SS] text` per segment
/// - `srt`: SubRip subtitle file with comma millisecond separators
/// - `vtt`: WebVTT with `WEBVTT` header and period millisecond separators
/// - `json`: structured payload buffered until the end
///
/// Multi-file batch is supported: pass any number of file paths. For
/// `text`, `srt`, and `vtt` each file's output is preceded by `# <basename>`
/// and a blank line. For `json` a `{"files": [...]}` wrapper is used; for a
/// single file the JSON wrapper is `{"segments": [...]}`.
///
/// The model bundle must already be on disk; this command does not auto-
/// download. On a missing bundle the user gets a `swiftwhisper download
/// <model>` hint.
struct TranscribeCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe one or more local audio files using a downloaded model."
    )

    @Argument(help: "Paths to audio files to transcribe (WAV recommended).")
    var audioFiles: [String]

    @Option(name: .shortAndLong, help: "Model to use (default: base).")
    var model: WhisperModel = .base

    @Option(name: .shortAndLong, help: "ISO-639-1 language code, e.g. 'en'. Auto-detect when omitted.")
    var language: String?

    @Option(name: .shortAndLong, help: "Output format: text, srt, vtt, json (default: text).")
    var format: OutputFormat = .text

    @Option(name: .long, help: "Override the model cache directory. Defaults to ~/Library/Application Support/SwiftWhisper/Models.")
    var cacheDir: String?

    func validate() throws {
        guard !audioFiles.isEmpty else {
            throw ValidationError("at least one audio file is required")
        }
        for path in audioFiles {
            let url = URL(fileURLWithPath: path)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw ValidationError("audio file not found: \(path)")
            }
        }
    }

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

        var options = DecodingOptions.default
        options.language = language

        let formatter = SegmentFormatters.make(format)
        let isBatch = audioFiles.count > 1

        if format == .json {
            try await runJSONBatch(
                files: audioFiles,
                tokenizer: tokenizer,
                encoder: encoder,
                decoder: decoder,
                options: options,
                isBatch: isBatch
            )
        } else {
            try await runStreaming(
                files: audioFiles,
                tokenizer: tokenizer,
                encoder: encoder,
                decoder: decoder,
                options: options,
                formatter: formatter,
                isBatch: isBatch
            )
        }
    }

    // MARK: - Streaming formats (text, srt, vtt)

    private func runStreaming(
        files: [String],
        tokenizer: WhisperTokenizer,
        encoder: WhisperEncoder,
        decoder: WhisperDecoder,
        options: DecodingOptions,
        formatter: any SegmentFormatter,
        isBatch: Bool
    ) async throws {
        if let header = formatter.header() {
            print(header)
        }

        for (fileIndex, path) in files.enumerated() {
            if isBatch {
                if fileIndex > 0 { print("") }
                let name = (path as NSString).lastPathComponent
                print("# \(name)")
            }

            let segments = try await collectSegments(
                path: path,
                tokenizer: tokenizer,
                encoder: encoder,
                decoder: decoder,
                options: options
            )
            for (index, segment) in segments.enumerated() {
                let line = formatter.format(segment: segment, index: index)
                if !line.isEmpty {
                    print(line)
                }
            }
        }
    }

    // MARK: - JSON (buffered)

    private func runJSONBatch(
        files: [String],
        tokenizer: WhisperTokenizer,
        encoder: WhisperEncoder,
        decoder: WhisperDecoder,
        options: DecodingOptions,
        isBatch: Bool
    ) async throws {
        var perFile: [(path: String, segments: [TranscriptionSegment])] = []
        for path in files {
            let segments = try await collectSegments(
                path: path,
                tokenizer: tokenizer,
                encoder: encoder,
                decoder: decoder,
                options: options
            )
            perFile.append((path: path, segments: segments))
        }

        if let output = SegmentRendering.encodeJSON(perFile: perFile, isBatch: isBatch) {
            print(output)
        }
    }

    // MARK: - Per-file pipeline

    private func collectSegments(
        path: String,
        tokenizer: WhisperTokenizer,
        encoder: WhisperEncoder,
        decoder: WhisperDecoder,
        options: DecodingOptions
    ) async throws -> [TranscriptionSegment] {
        let url = URL(fileURLWithPath: path)
        let melSpectrogram = try MelSpectrogram()
        let vad = EnergyVAD()
        let policy = LocalAgreementPolicy()
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
        let collector = SegmentCollector()
        let consumer = Task { [collector] in
            for await segment in stream {
                await collector.append(segment)
            }
        }

        await audioInput.waitUntilComplete()
        await pipeline.stop()
        await consumer.value
        return await collector.snapshot()
    }
}

/// Helpers shared between `transcribe` and `transcribe-mic` for JSON encoding.
enum SegmentRendering {

    nonisolated static func encodeJSON(
        perFile: [(path: String, segments: [TranscriptionSegment])],
        isBatch: Bool
    ) -> String? {
        if isBatch {
            let payload = BatchFilePayload(
                files: perFile.map { entry in
                    BatchFilePayload.File(
                        path: entry.path,
                        segments: entry.segments.map(JSONSegmentPayload.init(segment:))
                    )
                }
            )
            return JSONFormatter.encode(payload)
        } else {
            let segments = perFile.first?.segments ?? []
            let payload = SingleFilePayload(segments: segments.map(JSONSegmentPayload.init(segment:)))
            return JSONFormatter.encode(payload)
        }
    }
}

/// Actor used to collect segments from a streaming pipeline without violating
/// `Sendable` capture rules in concurrent closures.
actor SegmentCollector {

    private var segments: [TranscriptionSegment] = []

    func append(_ segment: TranscriptionSegment) {
        segments.append(segment)
    }

    func snapshot() -> [TranscriptionSegment] {
        segments
    }
}
