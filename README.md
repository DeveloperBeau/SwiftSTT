# SwiftWhisper

Pure Swift 6, Core ML-native real-time speech-to-text. No C++ or Objective-C bridges.

[![CI](https://github.com/DeveloperBeau/SwiftWhisper/actions/workflows/ci.yml/badge.svg)](https://github.com/DeveloperBeau/SwiftWhisper/actions/workflows/ci.yml)
[![Lint](https://github.com/DeveloperBeau/SwiftWhisper/actions/workflows/lint.yml/badge.svg)](https://github.com/DeveloperBeau/SwiftWhisper/actions/workflows/lint.yml)
[![Swift](https://img.shields.io/badge/swift-6.3-orange.svg)](https://www.swift.org/)
[![Platforms](https://img.shields.io/badge/platforms-iOS%2018%20%7C%20macOS%2015-blue.svg)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What it does

SwiftWhisper transcribes audio to text on-device using OpenAI's Whisper model running on Core ML. It supports live microphone capture and pre-recorded files, streams confirmed segments as the model produces them, and runs the entire pipeline in pure Swift. Output formats cover plain text plus six subtitle and structured formats. The package ships 437 unit tests, all mock-driven so CI runs in under 12 seconds without downloading model weights.

## Requirements

- Swift 6.3
- Xcode 16.x or newer
- iOS 18+ or macOS 15+

The iOS 18 / macOS 15 floor is set by `MLState`, which the decoder uses to hold the KV cache between forward passes. Earlier OS versions cannot run a stateful Core ML model.

## Installation

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/DeveloperBeau/SwiftWhisper.git", from: "0.13.0"),
```

Then add `SwiftWhisperKit` to your target:

```swift
.target(
    name: "MyApp",
    dependencies: [
        .product(name: "SwiftWhisperKit", package: "SwiftWhisper"),
    ]
)
```

`SwiftWhisperCore` (the protocol layer with no Apple framework dependencies) is pulled in as a transitive dependency.

In Xcode: **File > Add Package Dependencies** and paste the repo URL.

## Quick start (CLI)

```
swift run swiftwhisper download tiny
swift run swiftwhisper transcribe audio.wav
swift run swiftwhisper transcribe audio.wav --format srt -o subs.srt
swift run swiftwhisper transcribe-mic --max-duration 30
swift run swiftwhisper list-models
```

Available subcommands: `download`, `list-models`, `transcribe`, `transcribe-mic`, `info`.

## Quick start (library)

Pipeline-based live transcription:

```swift
import SwiftWhisperCore
import SwiftWhisperKit

@MainActor
final class Dictation {
    private var pipeline: TranscriptionPipeline?

    func start() async throws {
        let downloader = ModelDownloader()
        let model = WhisperModel.recommendedForCurrentDevice()

        if await !downloader.isDownloaded(model) {
            for try await _ in try await downloader.download(model) {}
        }

        let bundle = try await downloader.bundle(for: model)
        let loaded = try await ModelLoader().loadBundle(bundle)
        let tokenizer = try WhisperTokenizer(contentsOf: loaded.tokenizerURL)

        let pipeline = TranscriptionPipeline(
            audioInput: AVMicrophoneInput(),
            vad: EnergyVAD(),
            melSpectrogram: try MelSpectrogram(),
            encoder: WhisperEncoder(runner: MLModelRunner(model: loaded.encoder)),
            decoder: WhisperDecoder(
                runner: MLStateModelRunner(model: loaded.decoder),
                tokenizer: tokenizer
            ),
            tokenizer: tokenizer,
            policy: LocalAgreementPolicy()
        )
        self.pipeline = pipeline

        for await segment in try await pipeline.start() {
            print("[\(segment.start)s] \(segment.text)")
        }
    }

    func stop() async { await pipeline?.stop() }
}
```

For one-shot file transcription, lower-level token access, or meeting-length stitching, see the DocC articles under `Sources/SwiftWhisperKit/SwiftWhisperKit.docc/Articles/`.

## Models

| Model           | Disk    | Peak runtime | Recommended for                  |
| :-------------- | :------ | :----------- | :------------------------------- |
| `.tiny`         | ~150 MB | ~350 MB      | Apple Watch class, <2 GB devices |
| `.base`         | ~290 MB | ~600 MB      | iPhone SE, 2 to 4 GB devices     |
| `.small`        | ~580 MB | ~1.2 GB      | Recent iPhones, 4 to 6 GB        |
| `.largeV3Turbo` | ~800 MB | ~1.7 GB      | iPad Pro, Mac, 6+ GB             |

`WhisperModel.recommendedForCurrentDevice()` picks the largest model that fits the host's `ProcessInfo.physicalMemory` with headroom for the OS and host app. Use it when shipping a single binary that targets multiple device classes.

Models are pulled from `argmaxinc/whisperkit-coreml` on HuggingFace and cached in `~/Library/Application Support/SwiftWhisper/Models/`. Override the cache directory with `SWIFTWHISPER_CACHE_DIR` or pass `--cache-dir` to any CLI subcommand.

## Output formats

The CLI's `--format` flag accepts:

- `text` (default): `[HH:MM:SS -> HH:MM:SS] text` per segment
- `srt`: SubRip with comma millisecond separators
- `vtt`: WebVTT with period millisecond separators
- `json`: structured payload, buffered until end
- `ndjson`: one JSON object per line, streams as segments confirm
- `ttml`: Timed Text Markup Language XML, buffered
- `sbv`: YouTube SubViewer, streams

`ndjson` and `sbv` emit incrementally, so they work well for piping live mic output to another process. `json` and `ttml` buffer because they need a closing wrapper element.

## Architecture

```
mic / file -> AudioInputProvider -> VAD -> MelSpectrogram -> WhisperEncoder
                                                                |
                                                                v
                       TranscriptionSegment <- LocalAgreementPolicy <- WhisperDecoder
                                  |
                                  v
                         AsyncStream<TranscriptionSegment>
```

Components:

- **`AudioInputProvider`** (protocol): pulls 16 kHz mono Float audio from a source. Implementations: `AVMicrophoneInput`, `AudioFileInput`, `AVAudioCapture`.
- **`VoiceActivityDetector`** (protocol): gates downstream work to speech-only chunks. Implementations: `EnergyVAD`, `SileroVAD`, `VADBoundaryRefiner`.
- **`MelSpectrogramProcessor`** (actor): rolling 30-second log-mel buffer; matches Whisper's preprocessing.
- **`WhisperEncoder`** (actor): runs the audio encoder Core ML model and returns the encoder embeddings.
- **`WhisperDecoder`** (actor): autoregressive token decoder with greedy, temperature, and beam search modes; supports timestamp tokens, anti-hallucination thresholds, and per-beam KV cache via `BranchableStatefulRunner`.
- **`LocalAgreementPolicy`**: suppresses unstable partial hypotheses across decode windows.
- **`WordAligner`**: optional DTW-based word timing refinement.
- **`TranscriptionPipeline`** (actor): wires the above into a single `AsyncStream<TranscriptionSegment>` source.

Three SPM targets:

- `SwiftWhisperCore`: protocols, value types, and policy logic. Zero Apple framework deps.
- `SwiftWhisperKit`: AVFoundation, CoreML, and Accelerate implementations.
- `SwiftWhisperCLI`: `swiftwhisper` binary built on `SwiftWhisperKit`.

## Testing

```
swift test
```

437 unit tests in 49 suites finish in roughly 11 seconds on an M-series Mac. The full suite is mock-driven: no model download is required and no Core ML weights are loaded. Tests cover the encoder/decoder feature plumbing, beam search, sampling, timestamp parsing, VAD, mel spectrogram numerics, audio decoding, CLI argument parsing, and all seven output formatters.

A separate integration test target runs the real `.mlmodelc` end-to-end. It is gated by an environment variable so default `swift test` skips it:

```
SWIFTWHISPER_RUN_INTEGRATION=1 swift test
```

The integration suite downloads the tiny model (~150 MB on first run) and exercises the full pipeline against a synthetic audio buffer. It runs on demand via the `integration` GitHub Actions workflow (`workflow_dispatch`).

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Issues and pull requests are welcome. Please:

1. Run `swift test` before opening a PR.
2. Keep commit messages plain and conventional (`feat(scope): ...`, `fix(scope): ...`).
3. The pre-commit hook rejects em-dashes in source and markdown. Use `--` instead.
