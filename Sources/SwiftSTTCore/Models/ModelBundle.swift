import Foundation

/// Paths to the components of a downloaded Whisper model.
///
/// Groups the ggml model file (and optional Core ML encoder) so they
/// can be passed around as a single value. Construction performs no
/// file-system validation; the caller (typically
/// `ModelDownloader.bundle(for:)`) is responsible for checking that
/// the files exist before building one.
public struct ModelBundle: Sendable, Equatable {

    /// Which model variant this bundle represents.
    public let model: WhisperModel

    /// Root directory where the model's files are cached.
    public let directory: URL

    /// Path to the ggml model file (e.g. `ggml-tiny.en.bin`).
    public let ggmlModelURL: URL

    /// Optional path to a Core ML encoder `.mlmodelc` directory.
    ///
    /// When present whisper.cpp uses it for the encoder step (much
    /// faster on Apple Silicon). When `nil`, the ggml encoder runs
    /// on CPU/Metal.
    public let coreMLEncoderURL: URL?

    /// Creates a new ModelBundle with the supplied values.
    public init(
        model: WhisperModel,
        directory: URL,
        ggmlModelURL: URL,
        coreMLEncoderURL: URL? = nil
    ) {
        self.model = model
        self.directory = directory
        self.ggmlModelURL = ggmlModelURL
        self.coreMLEncoderURL = coreMLEncoderURL
    }
}
