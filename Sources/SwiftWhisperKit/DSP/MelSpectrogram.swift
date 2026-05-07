import Accelerate
import Foundation
import SwiftWhisperCore

/// Mel-scale log-power spectrogram computed in fixed-rate frames.
///
/// Output layout in `MelSpectrogramResult.frames`: row-major `[nMels x nFrames]`
/// (mel-major). Whisper-style normalization is applied:
/// `out = (max(log_mel, log_mel.max() - 8.0) + 4.0) / 4.0`, clamped to `(-1, 1]`.
public actor MelSpectrogram: MelSpectrogramProcessor {

    public static let frameLength: Int = FFTProcessor.frameLength   // 400
    public static let hopLength: Int = 160                          // 10 ms @ 16 kHz
    public static let nFFTBins: Int = FFTProcessor.outputBins       // 257 (FFT padded to 512)

    private let nMels: Int
    private let sampleRate: Float
    private let filterbank: [Float]   // row-major [nMels x nFFTBins]
    private let fft: FFTProcessor
    private var leftover: [Float] = []

    public init(nMels: Int = 80, sampleRate: Float = 16_000) {
        self.nMels = nMels
        self.sampleRate = sampleRate
        self.fft = FFTProcessor()
        self.filterbank = Self.makeMelFilterbank(
            nMels: nMels,
            fftSize: FFTProcessor.fftSize,
            sampleRate: sampleRate
        )
    }

    public func process(chunk: AudioChunk) async throws(SwiftWhisperError) -> MelSpectrogramResult {
        var samples = leftover
        samples.append(contentsOf: chunk.samples)

        var melFrames: [[Float]] = []   // each frame: [nMels] log-power
        var pos = 0
        while pos + Self.frameLength <= samples.count {
            let frame = Array(samples[pos..<(pos + Self.frameLength)])
            let power = fft.process(frame: frame)
            let mel = applyFilterbank(power: power)
            melFrames.append(mel)
            pos += Self.hopLength
        }

        // Carry over remainder for next call. Slice to keep memory bounded.
        leftover = Array(samples[pos..<samples.count])

        guard !melFrames.isEmpty else {
            return MelSpectrogramResult(frames: [], nMels: nMels, nFrames: 0)
        }

        return normalize(melFrames: melFrames)
    }

    public func reset() async {
        leftover.removeAll(keepingCapacity: true)
    }

    // MARK: - Internals

    private func applyFilterbank(power: [Float]) -> [Float] {
        // mel = filterbank · power, where filterbank is [nMels x nFFTBins] row-major.
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
        // log10 with floor
        for i in 0..<nMels {
            mel[i] = log10f(max(mel[i], 1e-10))
        }
        return mel
    }

    private func normalize(melFrames: [[Float]]) -> MelSpectrogramResult {
        let nFrames = melFrames.count
        var maxVal: Float = -.greatestFiniteMagnitude
        for frame in melFrames {
            for v in frame where v > maxVal {
                maxVal = v
            }
        }
        let floorVal = maxVal - 8.0

        // Output layout: row-major [nMels x nFrames] (mel-major).
        var out = [Float](repeating: 0, count: nMels * nFrames)
        for t in 0..<nFrames {
            let frame = melFrames[t]
            for m in 0..<nMels {
                let clamped = max(frame[m], floorVal)
                out[m * nFrames + t] = (clamped + 4.0) / 4.0
            }
        }
        return MelSpectrogramResult(frames: out, nMels: nMels, nFrames: nFrames)
    }

    // MARK: - Filterbank construction (HTK mel formula)

    static func makeMelFilterbank(nMels: Int, fftSize: Int, sampleRate: Float) -> [Float] {
        let nBins = fftSize / 2 + 1
        let fMax = sampleRate / 2
        let melMax = hzToMel(fMax)
        let melMin: Float = 0

        // (nMels + 2) mel-spaced points: lower edge, centers, upper edge.
        let melPoints: [Float] = (0..<(nMels + 2)).map { i in
            melMin + (melMax - melMin) * Float(i) / Float(nMels + 1)
        }
        let freqPoints = melPoints.map(melToHz)
        // Continuous bin positions for accurate triangle edges.
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
