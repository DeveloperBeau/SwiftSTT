import Foundation
import SwiftWhisperCore
import Testing
@preconcurrency import whisper
@testable import SwiftWhisperKit

@Suite("WhisperCppParams")
struct WhisperCppParamsTests {

    @Test("translate flag mirrors options.task")
    func translateFlag() {
        let transcribe = WhisperCppParams.fullParams(from: DecodingOptions(task: .transcribe))
        let translate = WhisperCppParams.fullParams(from: DecodingOptions(task: .translate))
        #expect(transcribe.translate == false)
        #expect(translate.translate == true)
    }

    @Test("detect_language is always false (it would exit without transcribing)")
    func detectLanguage() {
        // `detect_language == true` makes whisper.cpp detect the language and
        // return *without* transcribing. Auto-detection is driven by a nil
        // `language` instead, so this flag must stay false regardless.
        let auto = WhisperCppParams.fullParams(from: DecodingOptions(language: nil))
        let english = WhisperCppParams.fullParams(from: DecodingOptions(language: "en"))
        #expect(auto.detect_language == false)
        #expect(english.detect_language == false)
    }

    @Test("temperature flows through")
    func temperature() {
        let p = WhisperCppParams.fullParams(from: DecodingOptions(temperature: 0.5))
        #expect(p.temperature == 0.5)
    }

    @Test("suppress_blank mirrors options")
    func suppressBlank() {
        let p1 = WhisperCppParams.fullParams(from: DecodingOptions(suppressBlank: true))
        let p2 = WhisperCppParams.fullParams(from: DecodingOptions(suppressBlank: false))
        #expect(p1.suppress_blank == true)
        #expect(p2.suppress_blank == false)
    }

    @Test("n_threads is set to a sane positive value")
    func nThreads() {
        let p = WhisperCppParams.fullParams(from: DecodingOptions())
        // We deliberately cap at 4 threads. whisper.cpp documents this as the
        // sweet spot beyond which contention costs outweigh parallelism gains.
        #expect(p.n_threads > 0)
        #expect(p.n_threads <= 4)
    }

    @Test("unconditional defaults are set correctly")
    func unconditionalDefaults() {
        let p = WhisperCppParams.fullParams(from: .default)
        #expect(p.no_context == true)
        #expect(p.single_segment == false)
        #expect(p.print_special == false)
        #expect(p.print_progress == false)
        #expect(p.print_realtime == false)
        #expect(p.print_timestamps == false)
        #expect(p.max_tokens == 0)
    }
}
