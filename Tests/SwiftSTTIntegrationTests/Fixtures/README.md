# Accuracy fixtures

## jfk.wav

Source: `samples/jfk.wav` from whisper.cpp, https://github.com/ggml-org/whisper.cpp, tag `v1.9.6`. The tag points at commit `96ec4d9823e9d0211944172da35f9f29134af990`. The file is committed unchanged. Its sha256 is `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`.

Format: WAV, 1 channel, 16000 Hz, 16-bit integer PCM, 176000 frames (11.0 s). This is the rate `WhisperCppContext` takes. `AudioFileInput` reads it and converts it to 16 kHz mono Float32 samples. `AccuracyFixtureFormatTests` checks this format in the default gate.

## jfk.txt

The expected transcript, written by hand from the known speech. It was not produced by any transcription run. The word error rate check normalises case and punctuation only.

## Use

`AccuracyFixtureTests` transcribes `jfk.wav` with the tiny model and checks the word error rate against `jfk.txt`. It runs only when `SWIFTSTT_RUN_INTEGRATION=1` is set. The audio is read from the test bundle and decoded on this machine. It is never uploaded.

## Licence

whisper.cpp is distributed under the MIT License. Its licence text, which covers `jfk.wav`, follows.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
