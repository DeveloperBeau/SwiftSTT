# Understanding Mel Spectrograms

What a spectrogram actually is, why we take its logarithm, and what the "mel" part adds on top.

## Overview

Speech recognition models like Whisper do not look at audio waveforms directly. The waveform, with its 16 000 samples per second of wiggling amplitude, is a brutal representation to learn from. Instead, models receive a feature called the **log-mel spectrogram**: a compact picture of how loud each frequency band is over time, with the loudness on a log scale and the frequency axis warped to match human hearing.

This article unpacks what each piece of that means.

## From waveform to spectrogram

Sound is a pressure wave. When you record it, you sample that pressure thousands of times a second. The result is a one-dimensional array of floating-point amplitudes, one per sample. At 16 kHz, one second of audio is 16 000 numbers in the range `-1.0` to `1.0`.

A waveform tells you what is happening *now*, but it does not tell you what *frequencies* are present. A trumpet hitting a high note and a low note both look like wiggly lines; what differs is how fast they wiggle. To get at the frequency content you need a Fourier transform, which decomposes a chunk of waveform into a sum of pure sine waves and tells you how loud each one is.

A **spectrogram** is what you get when you slide a window across the waveform, take the Fourier transform of each window, and stack the results side by side as columns of a 2D image:

```
          time ▶
        ┌─────────────────────┐
freq ▲  │                     │
        │    spectrogram      │
        │                     │
        └─────────────────────┘
```

Each column is one moment in time. Each row is one frequency band. Brightness encodes how much energy lives in that band at that moment.

In SwiftSTT this step happens in ``FFTProcessor``. It takes a 25 ms window of samples (400 samples at 16 kHz), multiplies by a Hann window to reduce edge artefacts, and runs an FFT. The output is a power spectrum: the squared magnitude of each frequency bin. ``MelSpectrogram`` calls into ``FFTProcessor`` once per frame and stacks the results.

## Why the logarithm

A raw power spectrogram has a problem. Loud sounds and quiet sounds differ by enormous ratios. A whisper might sit at `0.0001` while a shout sits at `1.0`, a 10 000:1 ratio. If a model trains on those raw numbers, the loud frames dominate the loss and quiet ones contribute almost nothing.

Human ears do not work that way. We perceive loudness on roughly a logarithmic scale: doubling the perceived volume takes about a tenfold increase in actual power. Taking `log10` of the power spectrum compresses that 10 000:1 range down to a manageable 4 units, which the model can learn from evenly.

There is one practical issue: `log10(0)` is `-infinity`. Real audio buffers contain literal zeros during silence. ``MelSpectrogram`` clamps each value at `1e-10` before taking the log, so the lowest possible result is `log10(1e-10) = -10` rather than `-infinity`.

## Why the mel scale

The log spectrogram has frequency rows spaced linearly: bin 1 is 31 Hz, bin 2 is 62 Hz, bin 100 is 3125 Hz, and so on. Linear spacing is wrong for two reasons.

First, human hearing is logarithmic in frequency too. The difference between 200 Hz and 300 Hz sounds like a big jump (almost a perfect fifth musically). The difference between 5000 Hz and 5100 Hz sounds like nothing at all. A linear axis wastes resolution at the top and starves it at the bottom.

Second, speech information is concentrated in the low and mid frequencies. The fundamental frequency of a typical voice sits between 80 and 250 Hz. Vowel formants live between 200 and 3500 Hz. Almost everything above 4 kHz is consonant noise and harmonics, which need less resolution to identify.

The **mel scale** fixes both problems. It is a remapping of the frequency axis that:

- Spaces frequencies roughly the way humans perceive them (close to linear below 1 kHz, logarithmic above).
- Is named after "melody", because steps on the scale sound musically equal in size.

The conversion is a simple closed form. SwiftSTT uses the HTK formula:

```
mel(f) = 2595 * log10(1 + f / 700)
```

To turn a linear-frequency power spectrum into a mel spectrogram, we build a **mel filterbank**: a stack of 80 (or 128 for the large model) triangular filters, each one a weighted sum of nearby FFT bins. Filter 1 covers the lowest frequencies, filter 80 the highest. The triangles overlap by half so every FFT bin contributes to at least one filter. Multiplying the filterbank by the power spectrum is a single matrix-vector product, which is exactly what the `vDSP_mmul` call inside ``MelSpectrogram`` does.

## Putting it together

The log-mel spectrogram is what the Whisper encoder consumes. For each 30-second window of audio it expects a 80-by-3000 matrix of normalised log-mel values. The pipeline assembles that as follows:

1. Capture: 16 kHz mono Float32 audio in chunks (``AVAudioCapture``).
2. FFT: each 400-sample frame becomes a 257-bin power spectrum (``FFTProcessor``).
3. Mel filterbank: 257 power bins become 80 mel bands (``MelSpectrogram``).
4. Log: take `log10` with a `1e-10` floor.
5. Normalise: clamp at `max - 8.0`, shift, and divide so the output range is exactly 2 units wide regardless of input loudness.

The output goes straight into the encoder, which lifts it into the abstract embedding space the decoder reasons over.

## Why 100 frames per second

The frame rate matters for one practical reason: it sets the time resolution of the eventual transcription. SwiftSTT uses a 400-sample frame with a 160-sample hop, which works out to 100 frames per second (one frame every 10 ms). That is fine enough to capture the fastest phonemes in normal speech (about 50 ms each) without flooding the model with redundant data.

Doubling the hop length would halve the data the model has to process but blur the timestamps in the output. Halving it would do the opposite. The 10 ms hop is the choice the original Whisper paper made and SwiftSTT inherits.

## Further reading

- Davis, S. and Mermelstein, P., 1980. *Comparison of Parametric Representations for Monosyllabic Word Recognition in Continuously Spoken Sentences.* The original mel-cepstral coefficient paper that established mel features as a speech recognition staple.
- Stevens, S.S., Volkmann, J. and Newman, E.B., 1937. *A Scale for the Measurement of the Psychological Magnitude Pitch.* The original mel scale.
- Radford, A. et al., 2022. *Robust Speech Recognition via Large-Scale Weak Supervision.* The Whisper paper, including the exact log-mel preprocessing recipe.
