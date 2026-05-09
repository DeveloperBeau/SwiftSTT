# ``SwiftWhisperKit``

Apple-framework implementations of the SwiftWhisper pipeline. AVFoundation for capture, Accelerate for DSP, Core ML for the encoder and decoder.

## Overview

`SwiftWhisperKit` is where the abstract protocols in `SwiftWhisperCore` get plugged into Apple frameworks. Everything that needs AVFoundation, Core ML, or Accelerate lives here so that the Core target stays portable and quick to test.

The intended usage is straight composition: pick one capturer, one VAD, one mel processor, one encoder, one decoder, and one tokenizer; hand them to the (forthcoming) `TranscriptionPipeline` actor; iterate the result stream.

## Status

Milestones M1 through M8 are implemented:

- **M1 scaffold.** All public types and protocols compile with Swift 6 strict concurrency.
- **M2 audio.** ``AVAudioCapture`` and ``EnergyVAD`` are real implementations covered by tests.
- **M3 DSP.** ``FFTProcessor`` and ``MelSpectrogram`` produce Whisper-compatible mel features with carry-over between chunks.
- **M4 model loading.** ``ModelDownloader`` and ``ModelLoader`` resolve and load Core ML bundles.
- **M5 inference.** ``WhisperEncoder`` and ``WhisperDecoder`` run real Core ML inference.
- **M6 pipeline.** ``TranscriptionPipeline`` wires capture, VAD, mel, encoder, decoder, and local agreement into a streaming actor.
- **M7 CLI.** The `swiftwhisper` executable exposes `transcribe`, `download`, `list-models`, and `info`.
- **M8 decoder catch-up.** Temperature sampling, beam search, and timestamp parsing land on the decoder.

The neural ``SileroVAD`` remains stubbed pending a Core ML conversion of the upstream ONNX model.

## Topics

### Getting started

- <doc:Integrating-Into-an-App>
- <doc:Providing-Pre-Downloaded-Models>
- <doc:Recommended-Implementations>

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
- ``ModelDownloader``
- ``WhisperEncoder``
- ``WhisperDecoder``
- ``StatefulCoreMLModelRunner``
- ``MLStateModelRunner``

### Sampling

- ``RandomSource``
- ``SystemRandom``
- ``SeededRandom``

### Tokenizer

- ``WhisperTokenizer``
- ``BPETokenizer``

### Core ML protocols

- ``AudioEncoding``
- ``TokenDecoding``
