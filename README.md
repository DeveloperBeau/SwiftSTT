# SwiftWhisper

Pure Swift 6, Core ML-native, real-time speech-to-text engine. No C++. No Objective-C bridges. No paywalls.

**Status:** M1 scaffold (protocols + models). Implementation in progress.

## Targets

- `SwiftWhisperCore` - Pure logic, no Apple framework deps. Protocols and models.
- `SwiftWhisperKit` - Apple framework implementations (AVFoundation, CoreML, Accelerate).
- `SwiftWhisperCLI` - Command-line demo binary.

## Requirements

- iOS 18+ / macOS 15+
- Swift 6.0+
- Xcode 16+

## Roadmap

| Milestone | Status | Scope |
|-----------|--------|-------|
| M1 - Scaffold | done | Protocols, models, package structure |
| M2 - Audio | next | AVAudioCapture, EnergyVAD |
| M3 - DSP | next | FFT, mel spectrogram |
| M4 - Models | planned | Core ML model loading |
| M5 - Text | planned | Tokenizer, decoder |
| M6 - Streaming | planned | LocalAgreement, sliding window |
| M7 - Polish | planned | CLI, model downloader |

## License

MIT - see LICENSE file.
