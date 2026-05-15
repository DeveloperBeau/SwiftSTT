# SwiftWhisper

On-device speech-to-text for Apple platforms, built on whisper.cpp.

[![CI](https://github.com/DeveloperBeau/SwiftSTT/actions/workflows/ci.yml/badge.svg)](https://github.com/DeveloperBeau/SwiftSTT/actions/workflows/ci.yml)
[![Lint](https://github.com/DeveloperBeau/SwiftSTT/actions/workflows/lint.yml/badge.svg)](https://github.com/DeveloperBeau/SwiftSTT/actions/workflows/lint.yml)
[![Swift](https://img.shields.io/badge/swift-6.3-orange.svg)](https://www.swift.org/)
[![Platforms](https://img.shields.io/badge/platforms-iOS%2018%20%7C%20macOS%2015-blue.svg)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## What it does

SwiftWhisper transcribes audio to text on-device using OpenAI's Whisper models, running on [whisper.cpp](https://github.com/ggml-org/whisper.cpp) through a prebuilt xcframework. It captures from the microphone or reads pre-recorded files, then runs `whisper_full` once over the buffered audio and returns timestamped segments. The models are multilingual with automatic language detection. A `swiftwhisper` CLI covers transcription and model management across seven output formats; the library exposes the same engine for host apps. An optional Core ML encoder is downloaded alongside each model for extra on-device acceleration. The unit test suite is mock-driven, so CI runs without downloading model weights.

## Requirements

- Swift 6.3 toolchain
- iOS 18+ or macOS 15+

The iOS Simulator is supported, but whisper.cpp runs on its CPU backend there: the Metal backend crashes inside the Simulator's Metal driver, so the package forces `use_gpu = false` on the Simulator. Transcription works, just slower. Real iOS devices and macOS use the Metal GPU backend.

## Installation

Add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/DeveloperBeau/SwiftSTT.git", from: "0.0.1"),
```

Then add `SwiftWhisperKit` to your target:

```swift
.target(
    name: "MyApp",
    dependencies: [
        .product(name: "SwiftWhisperKit", package: "SwiftSTT"),
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
swift run swiftwhisper info small
```

Subcommands: `download`, `list-models`, `transcribe`, `transcribe-mic`, `info`.

`download --background` opts into a background `URLSession` so model pulls keep making progress while the host app is suspended. `transcribe` accepts multiple files and a `--concurrency` flag for parallel processing.

## Quick start (library)

One-shot transcription with `WhisperCppContext`. It loads a ggml model and runs `whisper_full` over 16 kHz mono `Float` samples:

```swift
import SwiftWhisperCore
import SwiftWhisperKit

let downloader = ModelDownloader()
let model = WhisperModel.recommendedForCurrentDevice()

if await !downloader.isDownloaded(model) {
    for try await _ in try await downloader.download(model) {}
}

let bundle = try await downloader.bundle(for: model)
let context = try WhisperCppContext(ggmlModelURL: bundle.ggmlModelURL)

// `samples` is 16 kHz mono Float audio (see AudioFileInput / AVMicrophoneInput).
let segments = try await context.transcribe(samples: samples)
for segment in segments {
    print("[\(segment.start)s] \(segment.text)")
}
```

For live microphone capture, `WhisperCppEngine` wraps the record-then-transcribe lifecycle and publishes async streams of status and segments:

```swift
let engine = WhisperCppEngine()

Task {
    for await segment in engine.segmentStream() {
        print(segment.text)
    }
}

await engine.prepare()   // loads the model recorded in WhisperModelStorage
try await engine.start() // begins mic capture
// ...
await engine.stop()      // runs whisper.cpp on the buffered audio
```

`prepare()` loads whichever model is recorded in `WhisperModelStorage`, so download one and set it as the default first.

## Models

| Model           | ggml weights | Peak runtime | Display name | Suited to                |
| :-------------- | :----------- | :----------- | :----------- | :----------------------- |
| `.tiny`         | ~75 MB       | ~200 MB      | Tiny         | <2 GB devices            |
| `.base`         | ~145 MB      | ~400 MB      | Small        | 2 to 4 GB devices        |
| `.small`        | ~465 MB      | ~1.0 GB      | Default      | 4 to 6 GB devices        |
| `.largeV3Turbo` | ~1.62 GB     | ~3.2 GB      | Best         | 6+ GB devices, Mac       |

`displayName` is a quality-ladder label, deliberately not matching the underlying Whisper size name: a user picks by how good they want results, not by parameter count.

`WhisperModel.recommendedForCurrentDevice()` picks the largest model that fits the host's `ProcessInfo.physicalMemory` with headroom for the OS and host app. Use it when shipping a single binary that targets multiple device classes.

Models are pulled from [`ggerganov/whisper.cpp`](https://huggingface.co/ggerganov/whisper.cpp) on HuggingFace and cached in `~/Library/Application Support/SwiftWhisper/Models/<model>/`. Each download also fetches an optional Core ML encoder (`<stem>-encoder.mlmodelc`). Override the cache directory with `SWIFTWHISPER_CACHE_DIR` or pass `--cache-dir` to any CLI subcommand.

## Output formats

The CLI's `--format` flag accepts:

- `text` (default): `[HH:MM:SS -> HH:MM:SS] text` per segment
- `srt`: SubRip with comma millisecond separators
- `vtt`: WebVTT with period millisecond separators
- `json`: structured payload, buffered until end
- `ndjson`: one JSON object per line, streams as segments confirm
- `ttml`: Timed Text Markup Language XML, buffered
- `sbv`: YouTube SubViewer, streams

`ndjson` and `sbv` emit incrementally, so they work well for piping output to another process. `json` and `ttml` buffer because they need a closing wrapper element.

## Architecture

```
mic / file -> AudioInputProvider -> [Float] buffer -> WhisperCppContext (whisper.cpp)
                                                              |
                                                              v
                                                    [TranscriptionSegment]
```

This is a record-then-transcribe engine: audio accumulates in a buffer while recording, then `whisper_full` runs once on `stop()`.

Components:

- **`AudioInputProvider`** (protocol): pulls 16 kHz mono `Float` audio from a source. Implementations: `AVMicrophoneInput`, `AudioFileInput`, `AVAudioCapture`.
- **`WhisperCppContext`** (actor): wraps a `whisper_context`; loads a ggml model plus an optional Core ML encoder and runs `whisper_full`, reporting progress through a callback.
- **`WhisperCppEngine`** (actor): mic capture and the `prepare` / `start` / `stop` lifecycle, publishing `WhisperEngineStatus` and `TranscriptionSegment` async streams. Conforms to `WhisperTranscriptionEngine`.
- **`ModelDownloader`** (actor): fetches ggml weights and the Core ML encoder with progress reporting, and manages the on-disk cache.
- **`EnergyVAD`** / **`SileroVAD`**: optional voice-activity detectors.

Three SPM library/executable targets, plus a binary target:

- `WhisperCpp`: the whisper.cpp v1.8.4 xcframework (binary target).
- `SwiftWhisperCore`: protocols, value types, and model metadata. Zero Apple framework deps.
- `SwiftWhisperKit`: the engine, audio capture, and downloader. Depends on `WhisperCpp` and ZIPFoundation.
- `SwiftWhisperCLI`: the `swiftwhisper` executable, built on `SwiftWhisperKit` and ArgumentParser.

## Testing

```
swift test
```

241 tests across 34 files in three unit test targets. The unit suite is mock-driven: no model download is required and no model weights are loaded.

A separate integration test target runs the real model end-to-end. It is gated by an environment variable so default `swift test` skips it:

```
SWIFTWHISPER_RUN_INTEGRATION=1 swift test --filter SwiftWhisperIntegrationTests
```

The integration suite downloads the tiny model (~75 MB on first run) and exercises the full pipeline against a synthetic audio buffer. `SWIFTWHISPER_INTEGRATION_MODEL` overrides which model it uses. It runs on demand via the `integration` GitHub Actions workflow (`workflow_dispatch`).

## License

MIT. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. In short:

1. Run `swift test` and `swift format lint -r Sources Tests Package.swift` before
   opening a PR.
2. Keep commit messages conventional (`feat(scope): ...`, `fix(scope): ...`).
3. The CI lint workflow rejects em-dash characters (Unicode `U+2014`) and a
   small set of marketing words in production sources. Use `--` or rewrite.

Run `scripts/install-hooks.sh` once after cloning to install a pre-commit hook
that runs `swift format lint` against staged Swift files.

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Security
reports go through [private vulnerability reporting](https://github.com/DeveloperBeau/SwiftSTT/security/advisories/new)
rather than public issues -- see [SECURITY.md](SECURITY.md).
