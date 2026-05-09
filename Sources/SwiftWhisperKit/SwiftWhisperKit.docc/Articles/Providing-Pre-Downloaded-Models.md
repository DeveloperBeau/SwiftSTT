# Providing pre-downloaded models

Ship models inside your app, side-load them from a custom directory, or let `ModelDownloader` fetch them on first launch.

## When to ship a model in the app bundle

Bundling a model removes the first-launch download (and the network requirement) at the cost of a bigger `.ipa` / `.app`. The trade-offs:

| Strategy            | App size impact | First-launch UX | Offline-first |
| :------------------ | :-------------- | :-------------- | :------------ |
| Bundled in app      | +150 MB to +800 MB | Instant transcription | Yes |
| Downloaded on demand | None            | Wait for download | Cached after first run |
| Side-loaded         | None            | Instant if pre-staged | Yes, if pre-staged |

For a focused dictation utility, bundling `.tiny` or `.base` is reasonable. For a feature inside a larger app, lazy download is usually the better choice.

## Layout the loader expects

A model directory must contain three things:

```
<cache>/openai_whisper-base/
  AudioEncoder.mlmodelc/        compiled Core ML encoder
  TextDecoder.mlmodelc/         compiled Core ML decoder  
  tokenizer.json                HuggingFace tokenizer file
  .complete                     marker file written by ModelDownloader
```

The directory name **must** match `WhisperModel.huggingFacePath` (e.g. `openai_whisper-base`). The `.complete` marker is what `ModelDownloader.isDownloaded(_:)` checks before handing out a `ModelBundle`.

## Option 1: bundle in the app

1. Download the model once with `swiftwhisper download base` (or via `ModelDownloader.download(_:)` in a one-off script).
2. Locate the cache directory the CLI used: `~/Library/Caches/SwiftWhisper/openai_whisper-base` on macOS, or the iOS Simulator equivalent.
3. Drag the entire `openai_whisper-base/` folder into your Xcode project. **Choose "Create folder reference"** (the blue folder icon), not "Create groups". Folder references preserve the `.mlmodelc` directory structure.
4. Make sure the folder is added to your app target's *Copy Bundle Resources* phase.

At runtime, build a `ModelBundle` directly:

```swift
import SwiftWhisperCore
import SwiftWhisperKit

func bundledModel() throws(SwiftWhisperError) -> ModelBundle {
    guard let directory = Bundle.main.url(
        forResource: "openai_whisper-base",
        withExtension: nil
    ) else {
        throw .modelFileMissing("openai_whisper-base in app bundle")
    }
    return ModelBundle(
        model: .base,
        directory: directory,
        encoderURL: directory.appendingPathComponent("AudioEncoder.mlmodelc"),
        decoderURL: directory.appendingPathComponent("TextDecoder.mlmodelc"),
        tokenizerURL: directory.appendingPathComponent("tokenizer.json")
    )
}
```

You can skip `ModelDownloader` entirely and pass this bundle straight to `ModelLoader.loadBundle(_:)`.

## Option 2: side-load into Application Support

If you want shared models across multiple apps (or you need to update them out-of-band), copy the model directory into a known location at install time:

```swift
let appSupport = try FileManager.default.url(
    for: .applicationSupportDirectory,
    in: .userDomainMask,
    appropriateFor: nil,
    create: true
).appendingPathComponent("MyApp/Whisper", isDirectory: true)

let downloader = ModelDownloader(cacheDirectory: appSupport)
let bundle = try downloader.bundle(for: .base)
```

If the `.complete` marker is missing from your pre-staged copy, write one:

```swift
let marker = appSupport
    .appendingPathComponent(WhisperModel.base.huggingFacePath)
    .appendingPathComponent(".complete")
try Data().write(to: marker)
```

## Option 3: download on first launch

The default. Hand the user a progress UI while `ModelDownloader.download(_:)` streams progress events:

```swift
let downloader = ModelDownloader()
let stream = try await downloader.download(.base)
for try await progress in stream {
    // progress.phase: .downloading | .verifying | .complete
    // progress.fraction: 0.0...1.0
    updateProgressBar(progress.fraction)
}
let bundle = try await downloader.bundle(for: .base)
```

The first download takes several seconds on a fast connection. Cache the resulting `ModelBundle` in your view model so subsequent launches go straight to `ModelLoader.loadBundle(_:)`.

## Where the cache lives by default

`ModelDownloader.init(cacheDirectory:)` defaults to `Library/Caches/SwiftWhisper` inside the app sandbox. The OS may evict files in `Caches` under storage pressure, which is fine because `ModelDownloader.isDownloaded(_:)` notices and re-downloads on demand. If you want guaranteed persistence, pass a directory under *Application Support* explicitly.

## Verification checklist

- `bundle.encoderURL` ends in `AudioEncoder.mlmodelc` and is a directory (not a `.mlmodel` file).
- `bundle.decoderURL` ends in `TextDecoder.mlmodelc` and is a directory.
- `bundle.tokenizerURL` ends in `tokenizer.json` and is readable.
- The `.complete` marker exists when going through `ModelDownloader.bundle(for:)`.

If the loader throws ``SwiftWhisperError/modelFileMissing(_:)``, the associated string names which file is missing.
