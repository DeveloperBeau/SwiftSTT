# Integrating SwiftSTT into an iOS or macOS app

Add the package to your Xcode project, request the right permissions, and pick the model that fits your device.

## Add the package

1. In Xcode, choose **File > Add Package Dependencies**.
2. Enter the SwiftSTT repository URL.
3. Pin to the latest minor version.
4. Add `SwiftSTTKit` to your app target. `SwiftSTTCore` is added automatically as a transitive dependency.

If you depend on the package directly in another Swift package:

```swift
.package(url: "https://github.com/<owner>/SwiftSTT.git", from: "0.1.0"),

.target(
    name: "MyApp",
    dependencies: [
        .product(name: "SwiftSTTKit", package: "SwiftSTT"),
    ]
)
```

The package requires iOS 18 / macOS 15 minimum. Earlier OS versions are not supported because the decoder relies on `MLState` for its KV cache, which only ships on iOS 18 / macOS 15.

## Required entitlements

### Microphone capture (live transcription)

Add `NSMicrophoneUsageDescription` to your app's `Info.plist` (or to the matching `Info` tab in the target editor). The string is shown to the user the first time you start the audio engine.

```
<key>NSMicrophoneUsageDescription</key>
<string>SwiftSTT transcribes your voice on-device. Audio never leaves your device.</string>
```

On macOS, also enable the **Audio Input** capability under *Signing & Capabilities*. Without it, the sandboxed app cannot read from the system microphone.

### File transcription

No entitlement is required if the user picks the file via `UIDocumentPickerViewController` / `NSOpenPanel`. If your app reads files from a custom location, enable **User Selected File** (read-only) on macOS.

## Pick a model for the device

Different model sizes consume very different amounts of RAM:

| Model         | Disk    | Peak runtime | Recommended for       |
| :------------ | :------ | :----------- | :-------------------- |
| `.tiny`       | ~150 MB | ~350 MB      | Apple Watch class, <2 GB devices |
| `.base`       | ~290 MB | ~600 MB      | iPhone SE, 2-4 GB devices |
| `.small`      | ~580 MB | ~1.2 GB      | Recent iPhones, 4-6 GB devices |
| `.largeV3Turbo` | ~800 MB | ~1.7 GB    | iPad Pro, Mac, 6+ GB devices |

`WhisperModel.recommendedForCurrentDevice()` reads `ProcessInfo.physicalMemory` and picks the largest model that fits with a comfortable headroom. Use it when shipping a single binary that should adapt to any device:

```swift
import SwiftSTTCore

let model = WhisperModel.recommendedForCurrentDevice()
```

If you target a fixed device class (only iPad, only Mac, etc), you can pin the model directly:

```swift
let model: WhisperModel = .largeV3Turbo
```

For finer control, call `WhisperModel.recommended(forPhysicalMemoryBytes:)` with your own budget.

## Plan for memory pressure

iOS terminates background apps holding large models without warning. If your app runs the model in the foreground only, you do not need extra handling. Otherwise:

- Drop your model loader instance when entering the background and reload on `.willEnterForeground`.
- Listen for `UIApplication.didReceiveMemoryWarningNotification` (or the `NSProcessInfoMemoryPressure` notification on macOS) and tear down the decoder.
- Avoid keeping more than one ``WhisperEncoder`` and ``WhisperDecoder`` alive at the same time.

## Where to next

- <doc:Providing-Pre-Downloaded-Models> for shipping models inside your app bundle, or pre-staging them in `Application Support`.
- <doc:Recommended-Implementations> for use-case tutorials covering live mic transcription, file transcription, and meeting capture.
