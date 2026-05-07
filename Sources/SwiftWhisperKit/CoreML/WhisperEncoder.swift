import CoreML
import SwiftWhisperCore

public actor WhisperEncoder: AudioEncoding {
    public init() {}

    public func encode(spectrogram: MelSpectrogramResult) async throws(SwiftWhisperError) -> MLMultiArray {
        throw .notImplemented
    }
}
