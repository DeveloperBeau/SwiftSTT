import ArgumentParser
import Foundation
import SwiftSTTCore
import SwiftSTTKit

/// Runs whisper.cpp transcription against one or more audio files and writes
/// the resulting segments in the requested format.
///
/// Supported formats (`--format`):
/// - `text` (default): `[HH:MM:SS -> HH:MM:SS] text` per segment
/// - `srt`: SubRip with comma millisecond separators
/// - `vtt`: WebVTT with `WEBVTT` header and period millisecond separators
/// - `json`: structured payload buffered until the end
/// - `ndjson`: one JSON object per line (streamable)
/// - `ttml`: Timed Text Markup Language (XML, buffered)
/// - `sbv`: YouTube SubViewer (streamable)
///
/// Multi-file batch is supported. By default files are processed sequentially
/// with `--concurrency 1`; raise it to run several files in parallel. Output
/// ordering always matches the order of file arguments regardless of
/// completion order.
///
/// > Note: `--concurrency > 1` uses a shared `WhisperCppContext`, which
/// > serialises `whisper_full` calls through the actor. Effective parallelism
/// > is therefore limited by the time spent in I/O and resampling. For
/// > CPU-bound throughput, running multiple `swiftstt transcribe`
/// > processes is more effective.
///
/// Output destination defaults to standard output. Pass `-o, --output <path>`
/// to write to a file. Combine with `--no-clobber` to refuse overwrite.
///
/// Audio format support follows what `AVAudioConverter` can resample to 16
/// kHz mono Float32. WAV, AIFF, and CAF are well-tested. M4A typically works
/// when the platform codec is available.
struct TranscribeCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "transcribe",
        abstract: "Transcribe one or more local audio files using a downloaded model."
    )

    @Argument(
        help: "Paths to audio files to transcribe (WAV, AIFF, CAF; M4A when codec available).")
    var audioFiles: [String]

    @Option(name: .shortAndLong, help: "Model to use (default: base).")
    var model: WhisperModel = .base

    @Option(
        name: .shortAndLong, help: "ISO-639-1 language code, e.g. 'en'. Auto-detect when omitted.")
    var language: String?

    @Option(
        name: .shortAndLong,
        help: "Output format: text, srt, vtt, json, ndjson, ttml, sbv (default: text).")
    var format: OutputFormat = .text

    @Option(
        name: [.short, .long],
        help: "Write transcript to <path> instead of stdout. Tilde paths are expanded.")
    var output: String?

    @Flag(name: .long, help: "Refuse to overwrite an existing --output file.")
    var noClobber: Bool = false

    @Option(
        name: .long,
        help:
            "Process N files in parallel (default 1). I/O and resampling overlap, but whisper.cpp inference is serialized through a shared context."
    )
    var concurrency: Int = 1

    @Option(
        name: .long,
        help:
            "Override the model cache directory. Honors SWIFTSTT_CACHE_DIR; default ~/Library/Application Support/SwiftSTT/Models."
    )
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
        guard concurrency >= 1 else {
            throw ValidationError("--concurrency must be >= 1 (got \(concurrency))")
        }
    }

    func run() async throws {
        if concurrency > 8 {
            writeStderr(
                "warning: --concurrency \(concurrency) is high; expect heavy memory and CPU pressure.\n"
            )
        }

        let downloader = ModelDownloader(cacheDirectory: CacheDirectoryOption.resolve(cacheDir))
        guard await downloader.isDownloaded(model) else {
            throw ValidationError(
                "model '\(model.rawValue)' is not downloaded."
                    + " Run 'swiftstt download \(model.rawValue)' first."
            )
        }
        let bundle = try await downloader.bundle(for: model)
        let context = try WhisperCppContext(ggmlModelURL: bundle.ggmlModelURL)

        var options = DecodingOptions.default
        options.language = language

        let formatter = SegmentFormatters.make(format)
        let isBatch = audioFiles.count > 1

        let destination: OutputDestination
        if let output {
            destination = try OutputDestination.file(at: output, noClobber: noClobber)
        } else {
            destination = .stdout()
        }
        defer { destination.close() }

        let perFile: [(path: String, segments: [TranscriptionSegment])]
        if concurrency > 1 && audioFiles.count > 1 {
            perFile = try await collectAllParallel(
                files: audioFiles,
                concurrency: concurrency,
                context: context,
                options: options
            )
        } else {
            perFile = try await collectAllSequential(
                files: audioFiles,
                context: context,
                options: options
            )
        }

        if formatter.bufferingRequired {
            try emitBuffered(
                perFile: perFile,
                formatter: formatter,
                isBatch: isBatch,
                destination: destination
            )
        } else {
            emitStreaming(
                perFile: perFile,
                formatter: formatter,
                isBatch: isBatch,
                destination: destination
            )
        }
    }

    // MARK: - Sequential collection

    private func collectAllSequential(
        files: [String],
        context: WhisperCppContext,
        options: DecodingOptions
    ) async throws -> [(path: String, segments: [TranscriptionSegment])] {
        var collected: [(path: String, segments: [TranscriptionSegment])] = []
        collected.reserveCapacity(files.count)
        for (index, path) in files.enumerated() {
            if files.count > 1 {
                writeStderr("[\(index)/\(files.count) done] processing \(path)...\n")
            }
            let segments = try await Self.collectSegmentsStatic(
                path: path,
                context: context,
                options: options
            )
            collected.append((path: path, segments: segments))
        }
        if files.count > 1 {
            writeStderr("[\(files.count)/\(files.count) done]\n")
        }
        return collected
    }

    // MARK: - Parallel collection

    private func collectAllParallel(
        files: [String],
        concurrency: Int,
        context: WhisperCppContext,
        options: DecodingOptions
    ) async throws -> [(path: String, segments: [TranscriptionSegment])] {
        let tasksLimit = max(1, min(concurrency, files.count))
        var indexed: [Int: [TranscriptionSegment]] = [:]
        let counter = ProgressCounter(total: files.count)

        try await withThrowingTaskGroup(of: (Int, [TranscriptionSegment]).self) { group in
            var nextIndex = 0
            var inFlight = 0

            while nextIndex < files.count && inFlight < tasksLimit {
                let captured = nextIndex
                let path = files[captured]
                group.addTask {
                    let segments = try await Self.collectSegmentsStatic(
                        path: path,
                        context: context,
                        options: options
                    )
                    let done = await counter.completed(path: path)
                    writeStderr("[\(done)/\(files.count) done] finished \(path)\n")
                    return (captured, segments)
                }
                nextIndex += 1
                inFlight += 1
            }

            while let (idx, segs) = try await group.next() {
                indexed[idx] = segs
                inFlight -= 1
                if nextIndex < files.count {
                    let captured = nextIndex
                    let path = files[captured]
                    group.addTask {
                        let segments = try await Self.collectSegmentsStatic(
                            path: path,
                            context: context,
                            options: options
                        )
                        let done = await counter.completed(path: path)
                        writeStderr("[\(done)/\(files.count) done] finished \(path)\n")
                        return (captured, segments)
                    }
                    nextIndex += 1
                    inFlight += 1
                }
            }
        }

        var results: [(path: String, segments: [TranscriptionSegment])] = []
        results.reserveCapacity(files.count)
        for (i, path) in files.enumerated() {
            results.append((path: path, segments: indexed[i] ?? []))
        }
        return results
    }

    // MARK: - Emission

    private func emitStreaming(
        perFile: [(path: String, segments: [TranscriptionSegment])],
        formatter: any SegmentFormatter,
        isBatch: Bool,
        destination: OutputDestination
    ) {
        if let header = formatter.header() {
            destination.writeLine(header)
        }
        for (fileIndex, entry) in perFile.enumerated() {
            if isBatch,
                let separator = formatter.fileSeparator(path: entry.path, fileIndex: fileIndex)
            {
                destination.writeLine(separator)
            }
            for (index, segment) in entry.segments.enumerated() {
                let line = formatter.format(segment: segment, index: index)
                if !line.isEmpty {
                    destination.writeLine(line)
                }
            }
        }
    }

    private func emitBuffered(
        perFile: [(path: String, segments: [TranscriptionSegment])],
        formatter: any SegmentFormatter,
        isBatch: Bool,
        destination: OutputDestination
    ) throws {
        if formatter is JSONFormatter {
            if let output = SegmentRendering.encodeJSON(perFile: perFile, isBatch: isBatch) {
                destination.writeLine(output)
            }
            return
        }
        let allSegments = perFile.flatMap(\.segments)
        if let body = formatter.footer(segments: allSegments) {
            destination.writeLine(body)
        }
    }

    // MARK: - Per-file pipeline

    /// Reads an audio file, transcribes it via `WhisperCppContext`, and returns
    /// the resulting segments.
    nonisolated static func collectSegmentsStatic(
        path: String,
        context: WhisperCppContext,
        options: DecodingOptions
    ) async throws -> [TranscriptionSegment] {
        let url = URL(fileURLWithPath: path)
        let samples = try await readAllSamples(at: url)
        return try await context.transcribe(samples: samples, options: options)
    }

    /// Drains an audio file into an in-memory `[Float]` at 16 kHz mono.
    nonisolated static func readAllSamples(at url: URL) async throws -> [Float] {
        let (chunkStream, chunkContinuation) = AsyncStream<[Float]>.makeStream()
        let input = AudioFileInput(fileURL: url)

        // Start emitting chunks into the stream as soon as start() yields them.
        // onChunk is @Sendable @escaping but synchronous, so we can yield directly.
        try await input.start(
            targetSampleRate: 16_000,
            bufferDurationSeconds: 1.0
        ) { @Sendable samples in
            chunkContinuation.yield(samples)
        }

        // Wait for EOF on a separate Task so we can finish the stream
        // and let the for-await loop terminate.
        Task {
            await input.waitUntilComplete()
            await input.stop()
            chunkContinuation.finish()
        }

        var samples: [Float] = []
        for await chunk in chunkStream {
            samples.append(contentsOf: chunk)
        }
        return samples
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
            let payload = SingleFilePayload(
                segments: segments.map(JSONSegmentPayload.init(segment:)))
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

/// Tracks how many files in a parallel batch have completed. Used solely to
/// produce the `[i/N done]` stderr progress line; never affects ordering.
actor ProgressCounter {

    private let total: Int
    private var done: Int = 0

    init(total: Int) {
        self.total = total
    }

    func completed(path: String) -> Int {
        done += 1
        return done
    }
}
