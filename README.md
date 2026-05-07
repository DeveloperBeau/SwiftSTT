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

## License

MIT - see LICENSE file.
