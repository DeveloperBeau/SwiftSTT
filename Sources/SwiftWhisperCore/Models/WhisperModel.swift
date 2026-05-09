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

    /// Approximate peak resident memory (in bytes) when the model is loaded
    /// and actively decoding. The figure is roughly twice the on-disk size to
    /// account for the encoder activations and the decoder KV cache held by
    /// `MLState`. Used by ``recommended(forPhysicalMemoryBytes:)`` to gate
    /// suggestions on smaller devices.
    public var approximatePeakRuntimeBytes: Int64 {
        switch self {
        case .tiny: 350_000_000
        case .base: 600_000_000
        case .small: 1_200_000_000
        case .largeV3Turbo: 1_700_000_000
        }
    }

    /// Picks the largest model that fits comfortably in the supplied physical
    /// memory budget, leaving headroom for the host app and the OS.
    ///
    /// The rule of thumb is `peakRuntime + 1 GB headroom <= physicalMemory`.
    /// Devices below 2 GB get ``tiny``, devices below 4 GB get ``base``,
    /// devices below 6 GB get ``small``, and anything above gets
    /// ``largeV3Turbo``. The thresholds are conservative on purpose since
    /// foreground apps on iOS are killed well before the device runs out of
    /// memory.
    public static func recommended(forPhysicalMemoryBytes bytes: UInt64) -> WhisperModel {
        let oneGB: UInt64 = 1_073_741_824
        if bytes < 2 * oneGB { return .tiny }
        if bytes < 4 * oneGB { return .base }
        if bytes < 6 * oneGB { return .small }
        return .largeV3Turbo
    }

    /// Convenience that asks the host process for `physicalMemory` and feeds
    /// it into ``recommended(forPhysicalMemoryBytes:)``. Returns the same
    /// model on every call for the lifetime of the process.
    public static func recommendedForCurrentDevice() -> WhisperModel {
        recommended(forPhysicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
    }
}
