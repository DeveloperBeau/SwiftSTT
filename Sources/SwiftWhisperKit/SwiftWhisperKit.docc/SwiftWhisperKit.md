# ``SwiftWhisperKit``

Apple-framework implementations of the SwiftWhisper pipeline. AVFoundation for capture, Accelerate for DSP, Core ML for the encoder and decoder.

## Overview

`SwiftWhisperKit` is where the abstract protocols in `SwiftWhisperCore` get plugged into Apple frameworks. Everything that needs AVFoundation, Core ML, or Accelerate lives here so that the Core target stays portable and quick to test.

The intended usage is straight composition: pick one capturer, one VAD, one mel processor, one encoder, one decoder, and one tokenizer; hand them to the (forthcoming) `TranscriptionPipeline` actor; iterate the result stream.

## Status

Milestones M1 through M3 are implemented:

- **M1 scaffold.** All public types and protocols compile with Swift 6 strict concurrency.
- **M2 audio.** ``AVAudioCapture`` and ``EnergyVAD`` are real implementations covered by tests.
- **M3 DSP.** ``FFTProcessor`` and ``MelSpectrogram`` produce Whisper-compatible mel features with carry-over between chunks.

The Core ML side (``ModelLoader``, ``WhisperEncoder``, ``WhisperDecoder``, ``WhisperTokenizer``) is stubbed and lands in M4 and M5. The neural ``SileroVAD`` is stubbed pending a Core ML conversion of the upstream ONNX model.

## Topics

### Audio capture

- ``AVAudioCapture``
- ``AudioConverter``

### Voice activity detection

- ``EnergyVAD``
- ``SileroVAD``

### Digital signal processing

- ``FFTProcessor``
- ``MelSpectrogram``
- <doc:Understanding-Mel-Spectrograms>

### Core ML pipeline

- ``ModelLoader``
- ``WhisperEncoder``
- ``WhisperDecoder``

### Tokenizer

- ``WhisperTokenizer``
- ``BPETokenizer``

### Core ML protocols

- ``AudioEncoding``
- ``TokenDecoding``
