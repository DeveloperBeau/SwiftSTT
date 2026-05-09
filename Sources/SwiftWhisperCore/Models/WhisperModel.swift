import Foundation

/// Available Whisper model sizes, matching the pre-converted Core ML models
/// published by argmaxinc on HuggingFace.
public enum WhisperModel: String, CaseIterable, Sendable, Identifiable {

    case tiny
    case base
    case small
    case largeV3Turbo // openai_whisper-large-v3-turbo

    public var id: String { rawValue }

    /// HuggingFace repository that hosts the Core ML-converted models.
    public static let huggingFaceRepo = "argmaxinc/whisperkit-coreml"

    /// Subdirectory name inside the HuggingFace repo for this model variant.
    public var huggingFacePath: String {
        switch self {
        case .tiny: "openai_whisper-tiny"
        case .base: "openai_whisper-base"
        case .small: "openai_whisper-small"
        case .largeV3Turbo: "openai_whisper-large-v3-turbo"
        }
    }

    public var displayName: String {
        switch self {
        case .tiny: "Tiny"
        case .base: "Base"
        case .small: "Small"
        case .largeV3Turbo: "Large V3 Turbo"
        }
    }

    /// Rough download size in bytes. Useful for showing a size estimate before
    /// the user commits to downloading.
    public var approximateSizeBytes: Int64 {
        switch self {
        case .tiny: 150_000_000
        case .base: 290_000_000
        case .small: 580_000_000
        case .largeV3Turbo: 800_000_000
        }
    }
}
