import Foundation

/// Available Whisper model sizes, matching the ggml model files published by
/// ggerganov/whisper.cpp on HuggingFace.
///
/// ## What this enum does not cover
///
/// - **Speaker diarization**. Whisper produces a single transcription stream;
///   assigning each segment to a speaker requires a separate model such as
///   pyannote. Not in scope for this package.
/// - **Punctuation restoration on languages without it in the training data**.
///   Whisper inserts punctuation for English and the major European languages;
///   lower-resource languages need a dedicated text model.
/// - **Voice activity detection**. The package provides `EnergyVAD` and
///   `SileroVAD`. Neither is downloaded through this enum; Silero models live
///   at <https://github.com/snakers4/silero-vad> and must be Core ML
///   compiled by the caller and loaded via `SileroVAD.load(from:)`.
public enum WhisperModel: String, CaseIterable, Sendable, Identifiable {

    case tiny
    case base
    case small
    case largeV3Turbo  // openai_whisper-large-v3-turbo

    /// Stable identifier for the model variant.
    public var id: String { rawValue }

    /// HuggingFace repository that hosts the ggml model files.
    public static let huggingFaceRepo = "ggerganov/whisper.cpp"

    /// Clean identifier for the local cache directory and on-disk file name.
    ///
    /// For example, `tiny` maps to `tiny/tiny.bin`. Deliberately does not
    /// carry the `ggml-` prefix; that is an artifact of the upstream repo's
    /// file naming, not something callers should see locally.
    public var fileStem: String {
        switch self {
        case .tiny: "tiny"
        case .base: "base"
        case .small: "small"
        case .largeV3Turbo: "large-v3-turbo"
        }
    }

    /// The exact file name published in the `ggerganov/whisper.cpp` HF repo.
    ///
    /// These are the multilingual variants (99 languages). The English-only
    /// `.en` files are intentionally not used: at every size tier they are
    /// the same byte size as the multilingual file and only trade language
    /// coverage for marginally better English accuracy.
    public var ggmlFileName: String {
        "ggml-\(fileStem).bin"
    }

    /// User-facing display name for the model.
    ///
    /// These are quality-ladder labels rather than the upstream Whisper
    /// size names: a user picks by how good they want results, not by the
    /// underlying parameter count.
    public var displayName: String {
        switch self {
        case .tiny: "Tiny"
        case .base: "Small"
        case .small: "Default"
        case .largeV3Turbo: "Best"
        }
    }

    /// Rough size of the ggml weights file (`<stem>.bin`) in bytes.
    public var approximateSizeBytes: Int64 {
        switch self {
        case .tiny: 75_000_000
        case .base: 145_000_000
        case .small: 465_000_000
        case .largeV3Turbo: 1_620_000_000
        }
    }

    /// Rough size of the optional Core ML encoder zip
    /// (`<stem>-encoder.mlmodelc.zip`) in bytes, about a fifth of the
    /// ggml file.
    public var approximateEncoderSizeBytes: Int64 {
        approximateSizeBytes / 5
    }

    /// Rough total of everything a model download fetches.
    ///
    /// That is the ggml weights plus the Core ML encoder; use this for the
    /// pre-download size estimate shown to the user.
    public var approximateDownloadSizeBytes: Int64 {
        approximateSizeBytes + approximateEncoderSizeBytes
    }

    /// Approximate peak resident memory (in bytes) when the model is loaded
    /// and actively decoding.
    ///
    /// whisper.cpp typically uses roughly 2–3× the on-disk ggml size at
    /// runtime to account for KV cache and intermediate activations.
    /// Used by ``recommended(forPhysicalMemoryBytes:)`` to gate
    /// suggestions on smaller devices.
    public var approximatePeakRuntimeBytes: Int64 {
        switch self {
        case .tiny: 200_000_000
        case .base: 400_000_000
        case .small: 1_000_000_000
        case .largeV3Turbo: 3_200_000_000
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
    /// it into ``recommended(forPhysicalMemoryBytes:)``.
    ///
    /// Returns the same model on every call for the lifetime of the process.
    public static func recommendedForCurrentDevice() -> WhisperModel {
        recommended(forPhysicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
    }

    /// Whether this model fits comfortably in the supplied physical memory
    /// budget.
    ///
    /// Adds a 1 GB headroom for the host app and OS, matching the rule used
    /// by ``recommended(forPhysicalMemoryBytes:)``.
    public func canRun(onPhysicalMemoryBytes bytes: UInt64) -> Bool {
        let oneGB: UInt64 = 1_073_741_824
        return UInt64(approximatePeakRuntimeBytes) + oneGB <= bytes
    }

    /// Whether this model fits in the current device's physical memory.
    public var canRunOnCurrentDevice: Bool {
        canRun(onPhysicalMemoryBytes: ProcessInfo.processInfo.physicalMemory)
    }
}
