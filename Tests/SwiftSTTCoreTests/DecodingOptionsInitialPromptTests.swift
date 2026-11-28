import Testing

@testable import SwiftSTTCore

@Suite("DecodingOptions initialPrompt")
struct DecodingOptionsInitialPromptTests {

    @Test("The default options carry no prompt")
    func defaultHasNoPrompt() {
        #expect(DecodingOptions.default.initialPrompt == nil)
        #expect(DecodingOptions().initialPrompt == nil)
    }

    @Test("An initialiser call written before the prompt existed still compiles and has no prompt")
    func existingInitialiserCallsHaveNoPrompt() {
        let options = DecodingOptions(
            language: "en",
            task: .translate,
            temperature: 0.2,
            beamSize: 2,
            usePrefixTimestamps: false,
            suppressBlank: false,
            suppressTokens: [1, 2],
            withoutTimestamps: false,
            temperatureFallback: [0.0],
            logProbThreshold: -2.0,
            noSpeechThreshold: 0.5,
            compressionRatioThreshold: 2.0,
            topP: 0.9,
            repetitionPenalty: 1.1,
            logitBias: [3: 0.5]
        )
        #expect(options.language == "en")
        #expect(options.logitBias == [3: 0.5])
        #expect(options.initialPrompt == nil)
    }

    @Test("The initialiser stores the prompt exactly as given")
    func initialiserStoresPrompt() {
        let options = DecodingOptions(initialPrompt: "  Zoltan Szabo\n")
        #expect(options.initialPrompt == "  Zoltan Szabo\n")
    }

    @Test("The prompt can be set after creation")
    func promptIsMutable() {
        var options = DecodingOptions.default
        options.initialPrompt = "Kubernetes"
        #expect(options.initialPrompt == "Kubernetes")
    }

    @Test("Options that differ only in the prompt are not equal")
    func equalityIncludesPrompt() {
        #expect(DecodingOptions(initialPrompt: "a") == DecodingOptions(initialPrompt: "a"))
        #expect(DecodingOptions(initialPrompt: "a") != DecodingOptions(initialPrompt: "b"))
        #expect(DecodingOptions(initialPrompt: "") != DecodingOptions())
        #expect(DecodingOptions() == DecodingOptions.default)
    }
}
