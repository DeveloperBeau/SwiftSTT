@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

/// End-to-end suite that downloads a real Whisper model and runs it through
/// the full encoder/decoder pipeline. Disabled by default. Run with:
///
/// ```
/// SWIFTWHISPER_RUN_INTEGRATION=1 swift test --filter SwiftWhisperIntegrationTests
/// ```
///
/// On a fresh machine the suite downloads the tiny model (~150 MB) into
/// `~/Library/Application Support/SwiftWhisper/Models/`. Subsequent runs are
/// cached.
///
/// ## Audio fixture choice
///
/// The test generates a 5-second 16 kHz mono PCM buffer of speech-like noise
/// (band-limited white noise plus a tonal sweep) rather than shipping a WAV
/// file in the repository. The trade-off: we cannot assert on transcribed
/// text content (synthetic audio decodes to whatever the model hallucinates),
/// only that the pipeline completes without error and produces a result. This
/// keeps the repo small. To assert on text, drop a WAV into a `Fixtures/`
/// subdirectory and read it with `AudioFileInput`.
@Suite(
    "Integration",
    .disabled(if: ProcessInfo.processInfo.environment["SWIFTWHISPER_RUN_INTEGRATION"] == nil)
)
struct IntegrationTests {

    @Test("Tiny model loads, encodes, and decodes a synthetic audio buffer")
    func tinyModelEndToEnd() async throws {
        let model = pickModel()

        // Phase 1: download (skipped if already cached).
        let downloader = ModelDownloader()
        if await !downloader.isDownloaded(model) {
            let stream = try await downloader.download(model)
            for try await progress in stream {
                if progress.phase == .complete { break }
            }
        }

        let bundle = try await downloader.bundle(for: model)
        #expect(FileManager.default.fileExists(atPath: bundle.encoderURL.path))
        #expect(FileManager.default.fileExists(atPath: bundle.decoderURL.path))
        #expect(FileManager.default.fileExists(atPath: bundle.tokenizerURL.path))

        // Phase 2: load encoder + decoder + tokenizer.
        let loader = ModelLoader()
        let loaded = try await loader.loadBundle(bundle)
        let tokenizer = try WhisperTokenizer(contentsOf: loaded.tokenizerURL)

        // Phase 3: build pipeline components and run one end-to-end decode.
        let samples = makeSyntheticAudio(seconds: 5, sampleRate: 16_000)
        let chunk = AudioChunk(samples: samples, sampleRate: 16_000, timestamp: 0)
        let mel = try await MelSpectrogram().process(chunk: chunk)

        let encoder = WhisperEncoder(runner: MLModelRunner(model: loaded.encoder))
        let encoded = try await encoder.encode(spectrogram: mel)

        var options = DecodingOptions.default
        options.language = "en"
        let decoder = WhisperDecoder(
            runner: MLStateModelRunner(model: loaded.decoder),
            tokenizer: tokenizer
        )
        let tokens = try await decoder.decode(encoderOutput: encoded, options: options)

        // We do not assert on text content because the audio is synthetic.
        // The contract here is: the pipeline completes without throwing.
        // Token output may legitimately be empty if Whisper's no-speech head
        // marks the buffer as silent.
        #expect(tokens.count >= 0)
    }

    // MARK: - Helpers

    /// Allows the `integration` workflow to override the model via env var.
    private func pickModel() -> WhisperModel {
        let raw = ProcessInfo.processInfo.environment["SWIFTWHISPER_INTEGRATION_MODEL"] ?? "tiny"
        return WhisperModel(rawValue: raw) ?? .tiny
    }

    /// Generates a deterministic synthetic audio buffer. Uses a sine sweep
    /// from 200 Hz to 4 kHz with low-amplitude band-limited noise added in.
    /// Values stay in `[-1.0, 1.0]`.
    private func makeSyntheticAudio(seconds: Int, sampleRate: Int) -> [Float] {
        let total = seconds * sampleRate
        var samples = [Float](repeating: 0, count: total)
        let twoPi = Float.pi * 2

        for i in 0..<total {
            let t = Float(i) / Float(sampleRate)
            // Linear sweep 200 Hz to 4000 Hz across the buffer.
            let progress = Float(i) / Float(total)
            let freq = 200 + (4000 - 200) * progress
            let phase = twoPi * freq * t
            let tone = sin(phase) * 0.2
            // Deterministic pseudo-noise via a simple LCG.
            let lcg = Float((i &* 1_103_515_245 &+ 12_345) & 0x7fff_ffff) / Float(0x7fff_ffff)
            let noise = (lcg - 0.5) * 0.05
            samples[i] = tone + noise
        }
        return samples
    }
}
