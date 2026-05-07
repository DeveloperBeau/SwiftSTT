import Accelerate
import Foundation

/// Real-valued FFT for Whisper-style 400-sample frames at 16 kHz, returning a power spectrum.
///
/// Input frames are 400 samples (25 ms @ 16 kHz, matching Whisper's n_fft semantics for hop
/// calculations) but are zero-padded to 512 internally because `vDSP.DFT` accepts only
/// counts of the form `f * 2^n` with `f ∈ {1, 3, 5, 15}`. This produces 257 output bins
/// at ~31.25 Hz/bin.
///
/// The Hann window (length 400) and `vDSP.DFT` setup are precomputed once and reused per
/// frame. The setup object is single-owner; transport across threads requires the owning
/// actor to serialize calls.
public struct FFTProcessor: Sendable {

    public static let frameLength: Int = 400
    public static let fftSize: Int = 512
    public static let outputBins: Int = fftSize / 2 + 1   // 257

    private let setup: DFTSetupBox
    private let window: [Float]

    public init() {
        self.setup = DFTSetupBox(count: Self.fftSize)
        self.window = (0..<Self.frameLength).map { i in
            0.5 * (1 - cosf(2 * .pi * Float(i) / Float(Self.frameLength - 1)))
        }
    }

    /// Computes the power spectrum `|X[k]|^2` of a windowed real-valued frame.
    /// - Parameter frame: exactly `frameLength` samples.
    /// - Returns: `outputBins` non-negative power values.
    public func process(frame: [Float]) -> [Float] {
        precondition(frame.count == Self.frameLength, "frame must be \(Self.frameLength) samples")

        // Window the 400-sample frame, then zero-pad to fftSize.
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

        setup.dft.transform(
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

/// `vDSP.DFT` setup object wrapped for `Sendable` transport. Single-owner usage required.
private final class DFTSetupBox: @unchecked Sendable {
    let dft: vDSP.DFT<Float>

    init(count: Int) {
        guard let dft = vDSP.DFT(
            count: count,
            direction: .forward,
            transformType: .complexComplex,
            ofType: Float.self
        ) else {
            preconditionFailure("vDSP.DFT init failed for count=\(count)")
        }
        self.dft = dft
    }
}
