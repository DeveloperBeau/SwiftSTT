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

    /// A model could not be loaded from the given URL or path. The associated string
    /// explains why (file missing, incompatible format, initialisation failure).
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

    /// An HTTP or network error prevented a model download from completing.
    case modelDownloadFailed(String)

    /// A downloaded file's SHA256 hash does not match the expected value from
    /// the HuggingFace listing.
    case modelChecksumMismatch(file: String)

    /// An expected file was not found in the model cache directory after download.
    case modelFileMissing(String)

    /// A `DecodingOptions` value was rejected by the decoder. The associated string
    /// names the offending field and the rule it violated (e.g. negative temperature,
    /// `beamSize < 1`, or an unsupported combination of beam search with sampling).
    case invalidDecodingOption(String)
}

extension SwiftWhisperError: LocalizedError {

    /// Human-readable description used by `error.localizedDescription` and
    /// surfaced via `NSError.bridge`.
    public var errorDescription: String? {
        switch self {
        case .notImplemented:
            return "Not implemented."
        case .micPermissionDenied:
            return "Microphone permission denied."
        case .modelLoadFailed(let reason):
            return "Model load failed: \(reason)"
        case .audioConversionFailed:
            return "Audio conversion failed."
        case .audioCaptureFailed(let reason):
            return "Audio capture failed: \(reason)"
        case .decoderFailure(let reason):
            return "Decoder failure: \(reason)"
        case .modelDownloadFailed(let reason):
            return "Model download failed: \(reason)"
        case .modelChecksumMismatch(let file):
            return "Checksum mismatch for \(file)."
        case .modelFileMissing(let file):
            return "Missing model file: \(file)"
        case .invalidDecodingOption(let reason):
            return "Invalid decoding option: \(reason)"
        }
    }
}
