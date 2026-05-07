import CoreML
import SwiftWhisperCore

public actor WhisperDecoder: TokenDecoding {
    public init() {}

    public func decode(
        encoderOutput: MLMultiArray,
        options: DecodingOptions
    ) async throws(SwiftWhisperError) -> [WhisperToken] {
        throw .notImplemented
    }
}
