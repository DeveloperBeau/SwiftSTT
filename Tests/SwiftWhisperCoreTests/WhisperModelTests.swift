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
        // Paths now match ggerganov/whisper.cpp ggml filenames (Task 6 replatform).
        #expect(WhisperModel.tiny.huggingFacePath == "ggml-tiny.en")
        #expect(WhisperModel.base.huggingFacePath == "ggml-base.en")
        #expect(WhisperModel.small.huggingFacePath == "ggml-small.en")
        #expect(WhisperModel.largeV3Turbo.huggingFacePath == "ggml-large-v3-turbo")
    }

    @Test("Approximate sizes are in reasonable range")
    func approximateSizes() {
        // Sizes are non-zero and ordered smallest to largest.
        for model in WhisperModel.allCases {
            #expect(model.approximateSizeBytes > 0)
        }
        #expect(
            WhisperModel.tiny.approximateSizeBytes < WhisperModel.largeV3Turbo.approximateSizeBytes)
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

    @Test("Recommended model picks tiny on devices below 2 GB")
    func recommendedTinyForLowMemory() {
        let oneGB: UInt64 = 1_073_741_824
        #expect(WhisperModel.recommended(forPhysicalMemoryBytes: oneGB) == .tiny)
        #expect(WhisperModel.recommended(forPhysicalMemoryBytes: 2 * oneGB - 1) == .tiny)
    }

    @Test("Recommended model picks base for 2-4 GB devices")
    func recommendedBaseForMidMemory() {
        let oneGB: UInt64 = 1_073_741_824
        #expect(WhisperModel.recommended(forPhysicalMemoryBytes: 2 * oneGB) == .base)
        #expect(WhisperModel.recommended(forPhysicalMemoryBytes: 3 * oneGB) == .base)
    }

    @Test("Recommended model picks small for 4-6 GB devices")
    func recommendedSmallForHighMemory() {
        let oneGB: UInt64 = 1_073_741_824
        #expect(WhisperModel.recommended(forPhysicalMemoryBytes: 4 * oneGB) == .small)
        #expect(WhisperModel.recommended(forPhysicalMemoryBytes: 5 * oneGB) == .small)
    }

    @Test("Recommended model picks largeV3Turbo on 6+ GB devices")
    func recommendedLargeForHighEnd() {
        let oneGB: UInt64 = 1_073_741_824
        #expect(WhisperModel.recommended(forPhysicalMemoryBytes: 6 * oneGB) == .largeV3Turbo)
        #expect(WhisperModel.recommended(forPhysicalMemoryBytes: 16 * oneGB) == .largeV3Turbo)
    }

    @Test("recommendedForCurrentDevice returns a known case")
    func recommendedForCurrentDevice() {
        let model = WhisperModel.recommendedForCurrentDevice()
        #expect(WhisperModel.allCases.contains(model))
    }

    @Test("approximatePeakRuntimeBytes is non-zero and increasing by size")
    func peakRuntimeIsIncreasing() {
        let order: [WhisperModel] = [.tiny, .base, .small, .largeV3Turbo]
        var previous: Int64 = 0
        for model in order {
            #expect(model.approximatePeakRuntimeBytes > previous)
            previous = model.approximatePeakRuntimeBytes
        }
    }
}
