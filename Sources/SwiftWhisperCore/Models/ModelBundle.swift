import Foundation

/// Paths to the components of a downloaded Whisper model.
///
/// A bundle groups the encoder, decoder, and tokenizer URLs together so they
/// can be passed around as a single value. Construction is a plain memberwise
/// init with no file-system validation; the caller (typically
/// `ModelDownloader.bundle(for:)`) is responsible for checking that the files
/// exist before building one of these.
public struct ModelBundle: Sendable, Equatable {

    /// Which model variant this bundle represents.
    public let model: WhisperModel

    /// Root directory where the model's files are cached.
    public let directory: URL

    /// Path to the `AudioEncoder.mlmodelc` compiled model directory.
    public let encoderURL: URL

    /// Path to the `TextDecoder.mlmodelc` compiled model directory.
    public let decoderURL: URL

    /// Path to the `tokenizer.json` vocabulary file.
    public let tokenizerURL: URL

    public init(
        model: WhisperModel,
        directory: URL,
        encoderURL: URL,
        decoderURL: URL,
        tokenizerURL: URL
    ) {
        self.model = model
        self.directory = directory
        self.encoderURL = encoderURL
        self.decoderURL = decoderURL
        self.tokenizerURL = tokenizerURL
    }
}
