import Accelerate
import Foundation

/// Real-input FFT for Whisper-style 400-sample frames at 16 kHz.
///
/// The Whisper preprocessing pipeline applies a Hann window to a 25 ms frame
/// (400 samples at 16 kHz), takes the FFT, and squares the magnitude to get a
/// power spectrum. That power spectrum is what the mel filterbank maps onto.
///
/// ## Why 512, not 400?
///
/// `vDSP.DiscreteFourierTransform` only accepts lengths of the form `f * 2^n`
/// where `f` is one of `{1, 3, 5, 15}`. 400 doesn't fit (it's `25 * 16`, and
/// 25 isn't on the list), so we zero-pad the windowed frame to 512 before
/// calling the transform. This changes the bin spacing from 40 Hz/bin to
/// 31.25 Hz/bin and gives 257 output bins instead of 201, but doesn't change
/// the underlying frequency content.
///
/// Switching to a true 400-point FFT to match the reference Python `whisper`
/// bit-exactly would need a non-Apple FFT library (KissFFT, FFTW, or a custom
/// Bluestein implementation). That's deferred until model integration shows
/// the bin offsets are causing accuracy regressions.
///
/// ## Performance
///
/// Both the Hann window and the DFT setup are precomputed once in
/// ``init()`` and reused across calls. Each ``process(frame:)`` allocates the
/// padded buffer, the imaginary input, and the two output buffers. Profiling
/// will eventually move those into a per-instance scratch area, but the current
/// shape keeps the code simple.
///
/// Not `Sendable`; hold as actor-isolated state (``MelSpectrogram`` does).
public struct FFTProcessor {

    /// Length of the input frame in samples. 25 ms at 16 kHz.
    public static let frameLength: Int = 400

    /// FFT length used internally after zero-padding.
    public static let fftSize: Int = 512

    /// Number of unique output bins. `fftSize/2 + 1`.
    public static let outputBins: Int = fftSize / 2 + 1   // 257

    private let dft: vDSP.DiscreteFourierTransform<Float>
    private let window: [Float]

    /// Builds the Hann window and DFT setup. Cheap enough to call per request,
    /// but typical use is one instance per pipeline.
    public init() {
        do {
            self.dft = try vDSP.DiscreteFourierTransform(
                count: Self.fftSize,
                direction: .forward,
                transformType: .complexComplex,
                ofType: Float.self
            )
        } catch {
            preconditionFailure("vDSP.DiscreteFourierTransform init failed: \(error)")
        }
        self.window = (0..<Self.frameLength).map { i in
            0.5 * (1 - cosf(2 * .pi * Float(i) / Float(Self.frameLength - 1)))
        }
    }

    /// Computes the power spectrum `|X[k]|^2` of a windowed frame.
    ///
    /// - Parameter frame: exactly ``frameLength`` real-valued samples.
    /// - Returns: ``outputBins`` non-negative power values, indexed from DC at
    ///   bin 0 to Nyquist at bin `outputBins - 1`.
    public func process(frame: [Float]) -> [Float] {
        precondition(frame.count == Self.frameLength, "frame must be \(Self.frameLength) samples")

        var padded = [Float](repeating: 0, count: Self.fftSize)
        padded.withUnsafeMutableBufferPointer { paddedPtr in
            frame.withUnsafeBufferPointer { framePtr in
                window.withUnsafeBufferPointer { winPtr in
                    vDSP_vmul(
                        framePtr.baseAddress!, 1,
                        winPtr.baseAddress!, 1,
                        paddedPtr.baseAddress!, 1,
                        vDSP_Length(Self.frameLength)
                    )
                }
            }
        }

        let imagIn = [Float](repeating: 0, count: Self.fftSize)
        var realOut = [Float](repeating: 0, count: Self.fftSize)
        var imagOut = [Float](repeating: 0, count: Self.fftSize)

        dft.transform(
            inputReal: padded,
            inputImaginary: imagIn,
            outputReal: &realOut,
            outputImaginary: &imagOut
        )

        var power = [Float](repeating: 0, count: Self.outputBins)
        for k in 0..<Self.outputBins {
            power[k] = realOut[k] * realOut[k] + imagOut[k] * imagOut[k]
        }
        return power
    }
}

