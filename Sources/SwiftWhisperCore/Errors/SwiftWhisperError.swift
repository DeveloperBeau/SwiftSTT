import Foundation

public enum SwiftWhisperError: Error, Sendable, Equatable {
    case notImplemented
    case micPermissionDenied
    case modelLoadFailed(String)
    case audioConversionFailed
    case audioCaptureFailed(String)
    case decoderFailure(String)
}
