import CoreML
import SwiftWhisperCore

public protocol AudioEncoding: Actor {
    func encode(spectrogram: MelSpectrogramResult) async throws(SwiftWhisperError) -> MLMultiArray
}
