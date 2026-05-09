import Foundation

/// All errors thrown across the SwiftWhisper pipeline.
///
/// Every public protocol uses typed throws (`throws(SwiftWhisperError)`), so callers
/// can match on these cases without bridging through a generic `Error`. New cases
/// should be added here rather than introducing additional error types per module.
public enum SwiftWhisperError: Error, Sendable, Equatable {

    /// A stub or unfinished method was called. Mostly useful during development;
    /// production code paths should never throw this.
    case notImplemented

    /// The user denied microphone access, or the system has restricted it (parental
    /// controls, MDM policy, etc).
    case micPermissionDenied

    /// A Core ML model could not be loaded from the given URL. The associated string
    /// explains why (file missing, ANE compile failure, format mismatch).
    case modelLoadFailed(String)

    /// `AVAudioConverter` refused to set up the requested format conversion. Usually
    /// means the input device is unavailable or the format combination is unsupported.
    case audioConversionFailed

    /// The capture pipeline could not be brought up. The associated string carries
    /// the underlying cause (audio session, engine start, missing input).
    case audioCaptureFailed(String)

    /// The decoder failed mid-generation. The associated string describes the failure
    /// (KV cache mismatch, NaN output, sampler error).
    case decoderFailure(String)

    /// `MelSpectrogramResult` was built with `frames.count != nMels * nFrames`.
    /// Carries the actual count and the count the layout requires.
    case invalidMelDimensions(framesCount: Int, expected: Int)

    /// `vDSP.DiscreteFourierTransform` rejected the requested setup. Carries the
    /// underlying error string.
    case fftSetupFailed(String)

    /// `FFTProcessor.process` was called with a frame whose length doesn't match
    /// the configured `frameLength`.
    case fftFrameSizeMismatch(got: Int, expected: Int)

    /// An HTTP or network error prevented a model download from completing.
    case modelDownloadFailed(String)

    /// A downloaded file's SHA256 hash does not match the expected value from
    /// the HuggingFace listing.
    case modelChecksumMismatch(file: String)

    /// An expected file was not found in the model cache directory after download.
    case modelFileMissing(String)
}
