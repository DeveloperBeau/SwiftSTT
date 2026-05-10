import Accelerate
import Foundation
import SwiftWhisperCore

/// Streaming mel-scale log-power spectrogram, Whisper-compatible.
///
/// This actor turns a stream of arbitrary-length `AudioChunk` values into the
/// log-mel features Whisper's encoder consumes. For an introduction to what
/// "log-mel" means and why speech models use it, see
/// <doc:Understanding-Mel-Spectrograms>.
///
/// The implementation is split into four steps that run for every 400-sample frame:
///
/// 1. Hann-window the frame and run an FFT through ``FFTProcessor``.
/// 2. Multiply the resulting power spectrum by an 80-band mel filterbank using
///    a single `vDSP_mmul` call.
/// 3. Take `log10` with a `1e-10` floor to avoid `-inf` on silence.
/// 4. After all frames in the chunk are computed, apply Whisper's normalisation
///    `out = (max(log_mel, log_mel.max() - 8.0) + 4.0) / 4.0`. The output range
///    is always exactly 2.0 wide regardless of input loudness.
///
/// ## Carry-over
///
/// Audio chunks usually don't divide evenly into 400-sample frames at a
/// 160-sample hop. Anything left over at the end of one chunk is prepended to
/// the next chunk's samples so no audio is lost at chunk boundaries. Use
/// ``reset()`` after a stream restart to drop the carry-over buffer.
///
/// ## Output layout
///
/// `MelSpectrogramResult.frames` is row-major `[nMels x nFrames]`, matching
/// the layout the Core ML encoder expects. To read mel band `m` at frame `t`,
/// use `frames[m * nFrames + t]`.
///
/// ## Filterbank
///
/// The mel scale uses the HTK formula (`2595 * log10(1 + hz / 700)`). This
/// differs slightly from `librosa`'s default Slaney scale that the reference
/// Python `whisper` uses. The difference is small for speech bands and the
/// output passes the math-property tests; bit-exact `librosa` parity will land
/// alongside the bit-exact FFT work.
public actor MelSpectrogram: MelSpectrogramProcessor {

    /// Frame size in samples. 25 ms at 16 kHz.
    public static let frameLength: Int = FFTProcessor.frameLength  // 400

    /// Hop size in samples. 10 ms at 16 kHz, giving 100 frames per second.
    public static let hopLength: Int = 160

    /// Number of FFT power bins each frame produces.
    public static let nFFTBins: Int = FFTProcessor.outputBins  // 257 (FFT padded to 512)

    private let nMels: Int
    private let sampleRate: Float
    private let filterbank: [Float]  // row-major [nMels x nFFTBins]
    private let fft: FFTProcessor
    private var leftover: [Float] = []

    /// Rolling per-mel-band column buffer. Streaming callers feed PCM through
    /// ``process(chunk:)`` and read the accumulated mel matrix back via
    /// ``snapshot()``, then call ``advance(framesConsumed:)`` to slide the window
    /// after a successful decode.
    private var rolling: [[Float]]
    private var rollingFrameCount: Int = 0

    /// Creates a mel processor. Propagates any error from the underlying
    /// ``FFTProcessor`` setup.
    ///
    /// - Parameters:
    ///   - nMels: number of mel bands. 80 for tiny/base/small/medium models,
    ///     128 for large.
    ///   - sampleRate: audio sample rate. 16 000 throughout the standard pipeline.
    public init(nMels: Int = 80, sampleRate: Float = 16_000) throws(SwiftWhisperError) {
        self.nMels = nMels
        self.sampleRate = sampleRate
        self.fft = try FFTProcessor()
        self.filterbank = Self.makeMelFilterbank(
            nMels: nMels,
            fftSize: FFTProcessor.fftSize,
            sampleRate: sampleRate
        )
        self.rolling = Array(repeating: [], count: nMels)
    }

    /// Processes a chunk and returns its mel features.
    ///
    /// Carries over any partial frame at the end of `chunk` into the next call.
    /// Returns an empty result when the carry-over plus this chunk is shorter
    /// than one full frame.
    public func process(chunk: AudioChunk) async throws(SwiftWhisperError) -> MelSpectrogramResult {
        var samples = leftover
        samples.append(contentsOf: chunk.samples)

        var melFrames: [[Float]] = []
        var pos = 0
        while pos + Self.frameLength <= samples.count {
            let frame = Array(samples[pos..<(pos + Self.frameLength)])
            let power = try fft.process(frame: frame)
            let mel = applyFilterbank(power: power)
            melFrames.append(mel)
            pos += Self.hopLength
        }

        leftover = Array(samples[pos..<samples.count])

        guard !melFrames.isEmpty else {
            return try MelSpectrogramResult(frames: [], nMels: nMels, nFrames: 0)
        }

        let result = try normalize(melFrames: melFrames)
        appendToRolling(result)
        return result
    }

    /// Drops the carry-over buffer and the rolling snapshot. Call after a
    /// stream restart so the next frame starts at sample zero and the next
    /// snapshot is empty.
    public func reset() async {
        leftover.removeAll(keepingCapacity: true)
        for m in 0..<nMels {
            rolling[m].removeAll(keepingCapacity: true)
        }
        rollingFrameCount = 0
    }

    // MARK: - Rolling window API

    /// Number of mel frames currently held in the rolling buffer.
    public func currentFrameCount() -> Int {
        rollingFrameCount
    }

    /// Returns a flattened copy of the rolling buffer in `[nMels x nFrames]`
    /// row-major layout. Subsequent ``advance(framesConsumed:)`` calls do not
    /// mutate the returned value.
    public func snapshot() throws(SwiftWhisperError) -> MelSpectrogramResult {
        guard rollingFrameCount > 0 else {
            return try MelSpectrogramResult(frames: [], nMels: nMels, nFrames: 0)
        }
        var flat = [Float](repeating: 0, count: nMels * rollingFrameCount)
        for m in 0..<nMels {
            for t in 0..<rollingFrameCount {
                flat[m * rollingFrameCount + t] = rolling[m][t]
            }
        }
        return try MelSpectrogramResult(frames: flat, nMels: nMels, nFrames: rollingFrameCount)
    }

    /// Drops the first `framesConsumed` columns from the rolling buffer.
    /// Negative or zero is a no-op. Values larger than ``currentFrameCount()``
    /// clamp to a full clear so callers can ask for "consume everything"
    /// without bookkeeping.
    public func advance(framesConsumed: Int) {
        guard framesConsumed > 0, rollingFrameCount > 0 else { return }
        let drop = min(framesConsumed, rollingFrameCount)
        for m in 0..<nMels {
            rolling[m].removeFirst(drop)
        }
        rollingFrameCount -= drop
    }

    private func appendToRolling(_ result: MelSpectrogramResult) {
        let added = result.nFrames
        guard added > 0 else { return }
        for m in 0..<nMels {
            for t in 0..<added {
                rolling[m].append(result.frames[m * added + t])
            }
        }
        rollingFrameCount += added
    }

    // MARK: - Internals

    private func applyFilterbank(power: [Float]) -> [Float] {
        var mel = [Float](repeating: 0, count: nMels)
        filterbank.withUnsafeBufferPointer { fbPtr in
            power.withUnsafeBufferPointer { pPtr in
                mel.withUnsafeMutableBufferPointer { mPtr in
                    vDSP_mmul(
                        fbPtr.baseAddress!, 1,
                        pPtr.baseAddress!, 1,
                        mPtr.baseAddress!, 1,
                        vDSP_Length(nMels),
                        1,
                        vDSP_Length(Self.nFFTBins)
                    )
                }
            }
        }
        for i in 0..<nMels {
            mel[i] = log10f(max(mel[i], 1e-10))
        }
        return mel
    }

    private func normalize(melFrames: [[Float]]) throws(SwiftWhisperError) -> MelSpectrogramResult {
        let nFrames = melFrames.count
        var maxVal: Float = -.greatestFiniteMagnitude
        for frame in melFrames {
            for v in frame where v > maxVal {
                maxVal = v
            }
        }
        let floorVal = maxVal - 8.0

        var out = [Float](repeating: 0, count: nMels * nFrames)
        for t in 0..<nFrames {
            let frame = melFrames[t]
            for m in 0..<nMels {
                let clamped = max(frame[m], floorVal)
                out[m * nFrames + t] = (clamped + 4.0) / 4.0
            }
        }
        return try MelSpectrogramResult(frames: out, nMels: nMels, nFrames: nFrames)
    }

    // MARK: - Filterbank construction (HTK mel formula)

    static func makeMelFilterbank(nMels: Int, fftSize: Int, sampleRate: Float) -> [Float] {
        let nBins = fftSize / 2 + 1
        let fMax = sampleRate / 2
        let melMax = hzToMel(fMax)
        let melMin: Float = 0

        let melPoints: [Float] = (0..<(nMels + 2)).map { i in
            melMin + (melMax - melMin) * Float(i) / Float(nMels + 1)
        }
        let freqPoints = melPoints.map(melToHz)
        let binPositions: [Float] = freqPoints.map { hz in
            (hz / sampleRate) * Float(fftSize)
        }

        var filterbank = [Float](repeating: 0, count: nMels * nBins)
        for m in 0..<nMels {
            let left = binPositions[m]
            let center = binPositions[m + 1]
            let right = binPositions[m + 2]
            for k in 0..<nBins {
                let kf = Float(k)
                let value: Float
                if kf < left || kf > right {
                    value = 0
                } else if kf <= center {
                    value = (kf - left) / max(1e-6, center - left)
                } else {
                    value = (right - kf) / max(1e-6, right - center)
                }
                filterbank[m * nBins + k] = value
            }
        }
        return filterbank
    }

    static func hzToMel(_ hz: Float) -> Float {
        2595 * log10f(1 + hz / 700)
    }

    static func melToHz(_ mel: Float) -> Float {
        700 * (powf(10, mel / 2595) - 1)
    }
}
