# Recommended implementations by use case

End-to-end snippets for the four most common transcription patterns, with the inputs each one expects and the outputs you should plumb into your UI.

## Inputs and outputs at a glance

| Input                             | Output                                       | Pattern               |
| :-------------------------------- | :------------------------------------------- | :-------------------- |
| Live microphone                   | `AsyncStream<TranscriptionSegment>`          | <doc:Recommended-Implementations#Live-mic-dictation> |
| Audio file on disk                | `[TranscriptionSegment]` once at the end     | <doc:Recommended-Implementations#One-shot-file-transcription> |
| Long meeting / multi-window audio | Stream of segments with absolute timestamps  | <doc:Recommended-Implementations#Meeting-capture-with-segment-stitching> |
| Pre-recorded `[Float]` PCM buffer | `[WhisperToken]` (raw, no segmenting)        | <doc:Recommended-Implementations#Lower-level-token-output> |

`TranscriptionSegment` carries `text`, `start`, and `end` (seconds since the start of capture). `WhisperToken` carries `id`, `text`, `probability`, and an optional `timestamp`.

## Live mic dictation

Best for: voice memos, hands-free notes, accessibility input.

The pipeline owns the mic, the VAD, the encoder, the decoder, and a local-agreement policy that suppresses partial hypotheses. You consume one `TranscriptionSegment` per stable utterance.

```swift
import SwiftWhisperCore
import SwiftWhisperKit

@MainActor
final class DictationViewModel: ObservableObject {
    @Published var transcript: String = ""
    private var pipeline: TranscriptionPipeline?

    func start() async throws {
        let downloader = ModelDownloader()
        let bundle = try await downloader.bundle(for: .base)
        let loader = ModelLoader()
        let loaded = try await loader.loadBundle(bundle)

        let pipeline = TranscriptionPipeline(
            audioInput: AVMicrophoneInput(),
            vad: EnergyVAD(),
            melSpectrogram: MelSpectrogram(),
            encoder: WhisperEncoder(runner: loaded.encoderRunner),
            decoder: WhisperDecoder(runner: loaded.decoderRunner, tokenizer: loaded.tokenizer),
            tokenizer: loaded.tokenizer,
            policy: LocalAgreementPolicy()
        )
        self.pipeline = pipeline

        let segments = try await pipeline.start()
        for await segment in segments {
            transcript += segment.text
        }
    }

    func stop() async {
        await pipeline?.stop()
    }
}
```

Tips:

- Call `start()` from a Task tied to the view's lifecycle and `stop()` when the view disappears.
- The pipeline already filters silence with VAD, so an idle mic does not produce empty segments.
- For best quality on noisy audio, set `options.temperature = 0.4`. Greedy is fine for clean studio capture.

## One-shot file transcription

Best for: voice notes, podcast import, video subtitling.

You skip the `TranscriptionPipeline` and call the encoder + decoder directly on a precomputed mel spectrogram.

```swift
import SwiftWhisperCore
import SwiftWhisperKit

func transcribe(file url: URL) async throws -> [TranscriptionSegment] {
    let downloader = ModelDownloader()
    let bundle = try await downloader.bundle(for: .base)
    let loader = ModelLoader()
    let loaded = try await loader.loadBundle(bundle)

    let input = AudioFileInput(url: url)
    let samples = try await input.readAllSamples(targetSampleRate: 16_000)

    let chunk = AudioChunk(samples: samples, sampleRate: 16_000, timestamp: 0)
    let mel = try await MelSpectrogram().process(chunk: chunk)
    let encoded = try await WhisperEncoder(runner: loaded.encoderRunner)
        .encode(spectrogram: mel)

    var options = DecodingOptions.default
    options.withoutTimestamps = false
    let decoder = WhisperDecoder(runner: loaded.decoderRunner, tokenizer: loaded.tokenizer)
    let tokens = try await decoder.decode(encoderOutput: encoded, options: options)

    return WhisperDecoder.parseSegments(
        tokens: tokens,
        tokenizer: loaded.tokenizer,
        windowOffsetSeconds: 0
    )
}
```

Whisper expects 16 kHz mono `Float` audio in the range `[-1.0, 1.0]`. `AudioFileInput` handles the resampling.

For files longer than 30 seconds, slide a 30-second window with a 5-second overlap:

```swift
let windowSeconds: TimeInterval = 30
let strideSeconds: TimeInterval = 25
var windowStart: TimeInterval = 0
var allSegments: [TranscriptionSegment] = []

while Int(windowStart * 16_000) < samples.count {
    let startIndex = Int(windowStart * 16_000)
    let endIndex = min(samples.count, startIndex + Int(windowSeconds * 16_000))
    let window = Array(samples[startIndex..<endIndex])

    let chunk = AudioChunk(samples: window, sampleRate: 16_000, timestamp: windowStart)
    let mel = try await MelSpectrogram().process(chunk: chunk)
    let encoded = try await encoder.encode(spectrogram: mel)
    let tokens = try await decoder.decode(encoderOutput: encoded, options: options)
    allSegments.append(contentsOf: WhisperDecoder.parseSegments(
        tokens: tokens,
        tokenizer: loaded.tokenizer,
        windowOffsetSeconds: windowStart
    ))
    windowStart += strideSeconds
}
```

`parseSegments` adds the window offset to every segment's `start` and `end`, so all timestamps come out in absolute file time.

## Meeting capture with segment stitching

Best for: long meetings, lectures, podcasts captured live.

This combines the live mic pipeline with the windowing approach above. The pipeline already runs on a sliding window internally, but for meetings you typically want timestamp-bearing segments that survive a one-hour conversation.

```swift
var options = DecodingOptions.default
options.withoutTimestamps = false
options.temperature = 0.4   // helps with cross-talk and reverberation

let pipeline = TranscriptionPipeline(
    audioInput: AVMicrophoneInput(),
    vad: EnergyVAD(),
    melSpectrogram: MelSpectrogram(),
    encoder: encoder,
    decoder: decoder,
    tokenizer: tokenizer,
    policy: LocalAgreementPolicy(),
    options: options,
    decodeIntervalSeconds: 2.0,         // less frequent decodes for less CPU
    maxBufferedFrames: 30 * 100         // cap at 30 s of audio per decode
)
```

Persist segments incrementally so a process termination during a long meeting does not lose work:

```swift
for await segment in try await pipeline.start() {
    Task.detached {
        try? await meetingStore.append(segment)
    }
}
```

## Lower level token output

Best for: custom segmenters, alignment tools, research.

Skip `TranscriptionPipeline` and `parseSegments` and read the raw `[WhisperToken]` from the decoder. Each token has its `id`, the decoded text fragment, and a `probability` that is useful for confidence-based filtering.

```swift
let tokens = try await decoder.decode(encoderOutput: encoded, options: .default)

let confident = tokens.filter { $0.probability > 0.5 }
let lowConfidenceCount = tokens.count - confident.count
```

Use beam search for the highest-quality output when latency does not matter:

```swift
var options = DecodingOptions.default
options.beamSize = 5
let tokens = try await decoder.decode(encoderOutput: encoded, options: options)
```

The current beam-search implementation re-prefills the KV cache for every beam at every step. It is correct but slow; expect roughly `beamSize` times the runtime of greedy decoding. Per-beam state lands in a future milestone.

## Picking decoding options

| Option              | Default | When to change                                 |
| :------------------ | :------ | :--------------------------------------------- |
| `language`          | `nil`   | Set explicitly to skip auto-detection. ISO 639-1 codes. |
| `task`              | `.transcribe` | `.translate` to translate non-English audio to English. |
| `temperature`       | `0.0`   | Raise to `0.2-0.6` if greedy gets stuck on repetitions. |
| `beamSize`          | `1`     | Raise to `5` for highest quality, batch jobs only. |
| `withoutTimestamps` | `true`  | Set to `false` whenever you want to call `parseSegments`. |
| `suppressBlank`     | `true`  | Leave on. Disabling lets silence emit text. |
| `suppressTokens`    | `[]`    | Add token IDs you want banned (e.g. profanity). |

`temperature > 0` and `beamSize > 1` are mutually exclusive. The decoder throws ``SwiftWhisperError/invalidDecodingOption(_:)`` if you set both.
