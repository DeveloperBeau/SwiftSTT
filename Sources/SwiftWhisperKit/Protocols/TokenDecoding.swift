import CoreML
import SwiftWhisperCore

public protocol TokenDecoding: Actor {
    func decode(
        encoderOutput: MLMultiArray,
        options: DecodingOptions
    ) async throws(SwiftWhisperError) -> [WhisperToken]
}
