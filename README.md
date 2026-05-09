# SwiftWhisper

Pure Swift 6, Core ML-native, real-time speech-to-text engine. No C++. No Objective-C bridges. No paywalls.

**Status:** M1 through M8 implemented. Streaming pipeline, beam search, temperature sampling, and timestamp parsing all live.

## Quick start

```swift
import SwiftWhisperCore
import SwiftWhisperKit

let downloader = ModelDownloader()
let model = WhisperModel.recommendedForCurrentDevice()
let bundle = try await downloader.bundle(for: model)
let loaded = try await ModelLoader().loadBundle(bundle)

let pipeline = TranscriptionPipeline(
    audioInput: AVMicrophoneInput(),
    vad: EnergyVAD(),
    melSpectrogram: MelSpectrogram(),
    encoder: WhisperEncoder(runner: loaded.encoderRunner),
    decoder: WhisperDecoder(runner: loaded.decoderRunner, tokenizer: loaded.tokenizer),
    tokenizer: loaded.tokenizer,
    policy: LocalAgreementPolicy()
)

for await segment in try await pipeline.start() {
    print(segment.text)
}
```

## Documentation

Build the docs with `swift package generate-documentation --target SwiftWhisperKit`. The articles cover:

- Integrating into an iOS or macOS app, including entitlements and device-aware model selection.
- Providing pre-downloaded models (bundled, side-loaded, or downloaded on first launch).
- Recommended implementations for live mic dictation, file transcription, meeting capture, and lower-level token access.

## Targets

- `SwiftWhisperCore` - Pure logic, no Apple framework deps. Protocols and models.
- `SwiftWhisperKit` - Apple framework implementations (AVFoundation, CoreML, Accelerate).
- `SwiftWhisperCLI` - Command-line demo binary.

## Requirements

- iOS 18+ / macOS 15+
- Swift 6.0+
- Xcode 16+

## License

MIT - see LICENSE file.
