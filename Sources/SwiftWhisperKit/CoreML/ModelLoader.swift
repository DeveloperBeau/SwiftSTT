import CoreML
import Foundation
import SwiftWhisperCore

public actor ModelLoader {
    public init() {}

    public func loadEncoder(at url: URL) async throws(SwiftWhisperError) -> MLModel {
        throw .notImplemented
    }

    public func loadDecoder(at url: URL) async throws(SwiftWhisperError) -> MLModel {
        throw .notImplemented
    }
}
