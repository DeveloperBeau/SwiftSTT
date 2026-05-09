import Testing
@testable import SwiftWhisperCore

@Suite("WhisperModel")
struct WhisperModelTests {

    @Test("All cases exist and are iterable")
    func allCases() {
        #expect(WhisperModel.allCases.count == 4)
    }

    @Test("Raw values are stable")
    func rawValues() {
        #expect(WhisperModel.tiny.rawValue == "tiny")
        #expect(WhisperModel.base.rawValue == "base")
        #expect(WhisperModel.small.rawValue == "small")
        #expect(WhisperModel.largeV3Turbo.rawValue == "largeV3Turbo")
    }

    @Test("HuggingFace paths map correctly")
    func huggingFacePaths() {
        #expect(WhisperModel.tiny.huggingFacePath == "openai_whisper-tiny")
        #expect(WhisperModel.largeV3Turbo.huggingFacePath == "openai_whisper-large-v3-turbo")
    }

    @Test("Approximate sizes are in reasonable range")
    func approximateSizes() {
        for model in WhisperModel.allCases {
            #expect(model.approximateSizeBytes > 100_000_000)
        }
        #expect(WhisperModel.tiny.approximateSizeBytes < WhisperModel.largeV3Turbo.approximateSizeBytes)
    }

    @Test("Display names are non-empty")
    func displayNames() {
        for model in WhisperModel.allCases {
            #expect(!model.displayName.isEmpty)
        }
    }

    @Test("Identifiable id matches raw value")
    func identifiable() {
        for model in WhisperModel.allCases {
            #expect(model.id == model.rawValue)
        }
    }
}
