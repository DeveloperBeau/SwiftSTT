import Foundation
import SwiftWhisperCore

public actor MelSpectrogram: MelSpectrogramProcessor {
    public init(nMels: Int = 80) {}

    public func process(chunk: AudioChunk) async throws(SwiftWhisperError) -> MelSpectrogramResult {
        throw .notImplemented
    }

    public func reset() async {}
}
