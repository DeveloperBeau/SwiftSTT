# ``SwiftWhisperCore``

Pure-logic foundation for SwiftWhisper. Models, protocols, and the streaming agreement policy with no Apple framework dependencies.

## Overview

`SwiftWhisperCore` is the part of SwiftWhisper that does not need AVFoundation, Core ML, or Accelerate. It defines the data types that flow through the pipeline (``AudioChunk``, ``MelSpectrogramResult``, ``WhisperToken``, ``TranscriptionResult``) and the protocols that pipeline stages conform to (``AudioCapturer``, ``VoiceActivityDetector``, ``MelSpectrogramProcessor``, ``Transcriber``).

Keeping this module framework-free has two payoffs. First, the test target compiles and runs in seconds because it does not need to link Apple frameworks. Second, the same types and protocols can be reused by anything that wants to plug into the pipeline, including non-Whisper transcribers or offline batch tools.

## Pipeline shape

A standard transcription run looks like this:

```
AudioCapturer ── AsyncStream<AudioChunk> ── VoiceActivityDetector
                                                │
                          (chunks marked as speech)
                                                ▼
                       MelSpectrogramProcessor ── MelSpectrogramResult ──▶ encoder
```

The ``Transcriber`` protocol wraps the whole thing and exposes its updates as `AsyncStream<TranscriptionResult>`. ``LocalAgreementPolicy`` runs inside the transcriber to stabilise the stream so the UI does not flicker when the model rewrites its own hypothesis.

## Topics

### Audio data

- ``AudioChunk``

### Spectrogram data

- ``MelSpectrogramResult``

### Decoder output

- ``WhisperToken``
- ``TranscriptionSegment``
- ``TranscriptionResult``

### Decoder configuration

- ``DecodingOptions``
- ``TaskKind``

### Pipeline protocols

- ``AudioCapturer``
- ``VoiceActivityDetector``
- ``MelSpectrogramProcessor``
- ``Transcriber``

### Streaming policy

- ``LocalAgreementPolicy``

### Errors

- ``SwiftWhisperError``
