import Foundation
import SwiftSTTCore
import Testing
@preconcurrency import whisper

@testable import SwiftSTTKit

/// Tests for the `initial_prompt` mapping, read back through ``WhisperCppParams/withCStrings``,
/// the closure that hands the final params to `whisper_full`.
@Suite("WhisperCppPrompt")
struct WhisperCppPromptTests {

    // MARK: - fullParams with no prompt

    @Test("fullParams for default options sets the same values as before prompts existed")
    func fullParamsDefaultsUnchanged() {
        let p = WhisperCppParams.fullParams(from: DecodingOptions())
        #expect(p.strategy == WHISPER_SAMPLING_GREEDY)
        #expect(p.translate == false)
        #expect(p.no_context == true)
        #expect(p.single_segment == false)
        #expect(p.print_special == false)
        #expect(p.print_progress == false)
        #expect(p.print_realtime == false)
        #expect(p.print_timestamps == false)
        #expect(p.detect_language == false)
        #expect(p.temperature == 0.0)
        #expect(p.suppress_blank == true)
        #expect(p.n_threads == Int32(min(ProcessInfo.processInfo.activeProcessorCount, 4)))
        #expect(p.max_tokens == 0)
        #expect(p.initial_prompt == nil)
        #expect(p.carry_initial_prompt == false)
        #expect(p.prompt_tokens == nil)
        #expect(p.prompt_n_tokens == 0)
    }

    @Test("fullParams keeps no_context true when a prompt is supplied")
    func fullParamsKeepsNoContextWithPrompt() {
        let p = WhisperCppParams.fullParams(from: DecodingOptions(initialPrompt: "Zoltan Szabo"))
        #expect(p.no_context == true)
        #expect(p.carry_initial_prompt == false)
        #expect(p.prompt_tokens == nil)
        #expect(p.prompt_n_tokens == 0)
    }

    // MARK: - Seam: the C strings the decoder receives

    @Test("A prompt reaches initial_prompt byte for byte")
    func promptArrivesByteForByte() {
        let bytes = deliveredPrompt(DecodingOptions(initialPrompt: "Zoltan Szabo, Kubernetes."))
        #expect(bytes == Array("Zoltan Szabo, Kubernetes.".utf8))
    }

    @Test("A nil prompt gives a nil initial_prompt pointer")
    func nilPromptGivesNilPointer() {
        let isNil = deliver(DecodingOptions()) { $0.initial_prompt == nil }
        #expect(isNil)
    }

    @Test("An empty prompt is passed through as an empty C string, not mapped to nil")
    func emptyPromptPassesThroughAsEmptyCString() {
        let bytes = deliveredPrompt(DecodingOptions(initialPrompt: ""))
        #expect(bytes == [])
    }

    @Test("Leading and trailing whitespace reaches the decoder untouched")
    func whitespaceIsNotTrimmed() {
        let prompt = "  \n\tZoltan  Szabo \r\n "
        let bytes = deliveredPrompt(DecodingOptions(initialPrompt: prompt))
        #expect(
            bytes == [
                0x20, 0x20, 0x0A, 0x09, 0x5A, 0x6F, 0x6C, 0x74, 0x61, 0x6E, 0x20, 0x20, 0x53, 0x7A,
                0x61,
                0x62, 0x6F, 0x20, 0x0D, 0x0A, 0x20,
            ])
    }

    @Test("Unicode text reaches the decoder as its UTF-8 bytes")
    func unicodeArrivesAsUTF8() {
        // "Zoë 東京" with a precomposed e-diaeresis.
        let bytes = deliveredPrompt(DecodingOptions(initialPrompt: "Zo\u{00EB} \u{6771}\u{4EAC}"))
        #expect(bytes == [0x5A, 0x6F, 0xC3, 0xAB, 0x20, 0xE6, 0x9D, 0xB1, 0xE4, 0xBA, 0xAC])
    }

    @Test("Decomposed unicode is not normalised")
    func decomposedUnicodeIsNotNormalised() {
        // "e" followed by a combining diaeresis must not become the precomposed form.
        let bytes = deliveredPrompt(DecodingOptions(initialPrompt: "e\u{0308}"))
        #expect(bytes == [0x65, 0xCC, 0x88])
    }

    @Test("A long prompt arrives whole, the library does not cut it")
    func longPromptArrivesWhole() {
        let prompt = String(repeating: "abc ", count: 25_000)
        let bytes = deliveredPrompt(DecodingOptions(initialPrompt: prompt))
        #expect(bytes?.count == 100_000)
        #expect(bytes == Array(prompt.utf8))
    }

    @Test("An embedded NUL ends the prompt at the C boundary")
    func embeddedNulTruncates() {
        let bytes = deliveredPrompt(DecodingOptions(initialPrompt: "Zoltan\u{0}Szabo"))
        #expect(bytes == Array("Zoltan".utf8))
    }

    @Test("The prompt pointer carries the prompt and the language pointer carries the language")
    func promptAndLanguageStayApart() {
        let options = DecodingOptions(language: "de", initialPrompt: "Zoltan")
        let (language, prompt, distinct) = deliver(options) { params in
            (
                cBytes(params.language), cBytes(params.initial_prompt),
                params.language != params.initial_prompt
            )
        }
        #expect(language == Array("de".utf8))
        #expect(prompt == Array("Zoltan".utf8))
        #expect(distinct)
    }

    @Test("A prompt does not change a nil language, and a language does not create a prompt")
    func promptAndLanguageAreIndependent() {
        let promptOnly = deliver(DecodingOptions(initialPrompt: "Zoltan")) { $0.language == nil }
        let languageOnly = deliver(DecodingOptions(language: "en")) { $0.initial_prompt == nil }
        #expect(promptOnly)
        #expect(languageOnly)
    }

    @Test("A prompt leaves every other mapped field as it was")
    func promptLeavesOtherFieldsAlone() {
        let options = DecodingOptions(
            task: .translate,
            temperature: 0.3,
            suppressBlank: false,
            initialPrompt: "Zoltan"
        )
        let params = WhisperCppParams.fullParams(from: options)
        let fields = WhisperCppParams.withCStrings(from: options, appliedTo: params) { p in
            MappedFields(p)
        }
        #expect(
            fields
                == MappedFields(
                    translate: true,
                    temperature: 0.3,
                    suppressBlank: false,
                    noContext: true,
                    detectLanguage: false,
                    singleSegment: false,
                    maxTokens: 0,
                    nThreads: Int32(min(ProcessInfo.processInfo.activeProcessorCount, 4)),
                    carryInitialPrompt: false,
                    promptNTokens: 0
                ))
    }

    @Test("The body runs exactly once and its result is returned")
    func bodyRunsOnce() {
        var calls = 0
        let options = DecodingOptions(initialPrompt: "Zoltan")
        let result = WhisperCppParams.withCStrings(
            from: options,
            appliedTo: WhisperCppParams.fullParams(from: options)
        ) { _ -> Int32 in
            calls += 1
            return 42
        }
        #expect(calls == 1)
        #expect(result == 42)
    }

    // MARK: - Fuzz

    @Test("Fuzz: random prompts arrive intact, never invent a pointer, never touch other fields")
    func fuzzPrompts() {
        let seed =
            ProcessInfo.processInfo.environment["SWIFTSTT_FUZZ_SEED"].flatMap { UInt64($0) }
            ?? 0x5EED_F00D
        print("WhisperCppPromptTests.fuzzPrompts seed=\(seed)")
        var rng = PromptFuzzGenerator(seed: seed)
        let languages: [String?] = [nil, "en", "de", "ja"]
        var seen = (nilPrompts: 0, empty: 0, withNul: 0, long: 0)

        for iteration in 0..<500 {
            let prompt = Self.randomPrompt(&rng)
            let language = languages[Int(rng.next() % UInt64(languages.count))]
            let translate = rng.next() % 2 == 0
            let temperature = Float(rng.next() % 11) / 10
            let suppressBlank = rng.next() % 2 == 0
            let options = DecodingOptions(
                language: language,
                task: translate ? .translate : .transcribe,
                temperature: temperature,
                suppressBlank: suppressBlank,
                initialPrompt: prompt
            )
            let context = "seed \(seed) iteration \(iteration)"

            let (promptBytes, languageBytes, fields) = WhisperCppParams.withCStrings(
                from: options,
                appliedTo: WhisperCppParams.fullParams(from: options)
            ) { p in
                (cBytes(p.initial_prompt), cBytes(p.language), MappedFields(p))
            }

            if prompt == nil { seen.nilPrompts += 1 }
            if prompt == "" { seen.empty += 1 }
            if prompt?.utf8.contains(0) == true { seen.withNul += 1 }
            if (prompt?.utf8.count ?? 0) > 4_096 { seen.long += 1 }

            switch prompt {
            case .none:
                // False positive check: no prompt never produces a pointer.
                #expect(promptBytes == nil, "\(context): nil prompt produced a pointer")
            case .some(let text):
                let utf8 = Array(text.utf8)
                if let nul = utf8.firstIndex(of: 0) {
                    // False positive check: nothing after a NUL is ever delivered.
                    #expect(
                        promptBytes == Array(utf8[..<nul]), "\(context): bytes after NUL leaked")
                } else {
                    // False negative check: every prompt without a NUL arrives whole.
                    #expect(promptBytes == utf8, "\(context): prompt bytes changed")
                }
            }
            #expect(
                languageBytes == language.map { Array($0.utf8) }, "\(context): language changed")
            #expect(
                fields
                    == MappedFields(
                        translate: translate,
                        temperature: temperature,
                        suppressBlank: suppressBlank,
                        noContext: true,
                        detectLanguage: false,
                        singleSegment: false,
                        maxTokens: 0,
                        nThreads: Int32(min(ProcessInfo.processInfo.activeProcessorCount, 4)),
                        carryInitialPrompt: false,
                        promptNTokens: 0
                    ),
                "\(context): a mapped field changed"
            )
        }
        // The generator must have produced every kind of prompt the checks above rely on.
        #expect(seen.nilPrompts > 0 && seen.empty > 0 && seen.withNul > 0 && seen.long > 0)
        print("WhisperCppPromptTests.fuzzPrompts coverage \(seen)")
    }

    // MARK: - Helpers

    private static let fuzzAlphabet: [Unicode.Scalar] = {
        var scalars: [Unicode.Scalar] = []
        scalars += (0x20...0x7E).compactMap(Unicode.Scalar.init)
        scalars += (0x01...0x1F).compactMap(Unicode.Scalar.init)
        scalars += [
            0x7F, 0xA0, 0xE9, 0xEB, 0x308, 0x5D0, 0x627, 0x200B, 0x2028, 0x3000, 0x6771, 0xFEFF,
        ]
        .compactMap(Unicode.Scalar.init)
        scalars += [0x1F399, 0x1F600, 0x1F1E6].compactMap(Unicode.Scalar.init)
        return scalars
    }()

    /// Returns nil, an empty string, a very long string, or a short random string.
    ///
    /// About one in five non-empty strings has an embedded NUL.
    private static func randomPrompt(_ rng: inout PromptFuzzGenerator) -> String? {
        switch rng.next() % 10 {
        case 0: return nil
        case 1: return ""
        default: break
        }
        let length =
            rng.next() % 10 == 0 ? 5_000 + Int(rng.next() % 5_000) : 1 + Int(rng.next() % 64)
        var scalars = String.UnicodeScalarView()
        for _ in 0..<length {
            scalars.append(fuzzAlphabet[Int(rng.next() % UInt64(fuzzAlphabet.count))])
        }
        if rng.next() % 5 == 0 {
            let index = scalars.index(
                scalars.startIndex, offsetBy: Int(rng.next() % UInt64(length)))
            scalars.insert("\u{0}", at: index)
        }
        return String(scalars)
    }

    private func deliver<R>(_ options: DecodingOptions, _ read: (whisper_full_params) -> R) -> R {
        WhisperCppParams.withCStrings(
            from: options,
            appliedTo: WhisperCppParams.fullParams(from: options),
            read
        )
    }

    private func deliveredPrompt(_ options: DecodingOptions) -> [UInt8]? {
        deliver(options) { cBytes($0.initial_prompt) }
    }
}

/// Reads a C string's bytes up to its terminating NUL, the way whisper.cpp does.
private func cBytes(_ pointer: UnsafePointer<CChar>?) -> [UInt8]? {
    guard let pointer else { return nil }
    return Array(UnsafeRawBufferPointer(start: pointer, count: strlen(pointer)))
}

/// The `whisper_full_params` fields that a prompt must never change.
private struct MappedFields: Equatable {
    var translate: Bool
    var temperature: Float
    var suppressBlank: Bool
    var noContext: Bool
    var detectLanguage: Bool
    var singleSegment: Bool
    var maxTokens: Int32
    var nThreads: Int32
    var carryInitialPrompt: Bool
    var promptNTokens: Int32

    init(
        translate: Bool,
        temperature: Float,
        suppressBlank: Bool,
        noContext: Bool,
        detectLanguage: Bool,
        singleSegment: Bool,
        maxTokens: Int32,
        nThreads: Int32,
        carryInitialPrompt: Bool,
        promptNTokens: Int32
    ) {
        self.translate = translate
        self.temperature = temperature
        self.suppressBlank = suppressBlank
        self.noContext = noContext
        self.detectLanguage = detectLanguage
        self.singleSegment = singleSegment
        self.maxTokens = maxTokens
        self.nThreads = nThreads
        self.carryInitialPrompt = carryInitialPrompt
        self.promptNTokens = promptNTokens
    }

    init(_ p: whisper_full_params) {
        self.init(
            translate: p.translate,
            temperature: p.temperature,
            suppressBlank: p.suppress_blank,
            noContext: p.no_context,
            detectLanguage: p.detect_language,
            singleSegment: p.single_segment,
            maxTokens: p.max_tokens,
            nThreads: p.n_threads,
            carryInitialPrompt: p.carry_initial_prompt,
            promptNTokens: p.prompt_n_tokens
        )
    }
}

/// Small deterministic generator so a fuzz failure can be replayed from its seed.
private struct PromptFuzzGenerator: RandomNumberGenerator {
    private var state: UInt64

    init(seed: UInt64) { self.state = seed }

    mutating func next() -> UInt64 {
        // SplitMix64.
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
