import Testing

@testable import SwiftWhisperCore

@Suite("Models")
struct ModelsTests {

    @Test("AudioChunk equality")
    func audioChunkEquality() {
        let a = AudioChunk(samples: [0.0, 1.0], sampleRate: 16_000, timestamp: 0)
        let b = AudioChunk(samples: [0.0, 1.0], sampleRate: 16_000, timestamp: 0)
        #expect(a == b)
    }

    @Test("MelSpectrogramResult valid dimensions")
    func melSpectrogramFramesLength() throws {
        let result = try MelSpectrogramResult(
            frames: Array(repeating: 0, count: 80 * 100), nMels: 80, nFrames: 100)
        #expect(result.frames.count == 80 * 100)
    }

    @Test("MelSpectrogramResult invalid dimensions throws")
    func melSpectrogramInvalid() throws {
        do {
            _ = try MelSpectrogramResult(frames: [0, 0, 0], nMels: 2, nFrames: 5)
            Issue.record("expected throw")
        } catch {
            #expect(error == .invalidMelDimensions(framesCount: 3, expected: 10))
        }
    }

    @Test("DecodingOptions defaults")
    func decodingOptionsDefaults() {
        let opts = DecodingOptions.default
        #expect(opts.task == .transcribe)
        #expect(opts.temperature == 0.0)
        #expect(opts.beamSize == 1)
        #expect(opts.suppressBlank == true)
        #expect(opts.withoutTimestamps == true)
    }

    @Test("TaskKind cases")
    func taskKindCases() {
        #expect(TaskKind.transcribe.rawValue == "transcribe")
        #expect(TaskKind.translate.rawValue == "translate")
    }

    @Test("WhisperToken initialization")
    func whisperTokenInit() {
        let token = WhisperToken(id: 42, text: "hello")
        #expect(token.id == 42)
        #expect(token.text == "hello")
        #expect(token.probability == 1.0)
        #expect(token.timestamp == nil)
    }

    @Test("TranscriptionResult defaults")
    func transcriptionResultDefaults() {
        let result = TranscriptionResult(text: "hi")
        #expect(result.text == "hi")
        #expect(result.hypothesis == "")
        #expect(result.isFinal == false)
        #expect(result.segments.isEmpty)
    }
}
