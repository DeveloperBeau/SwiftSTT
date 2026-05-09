@preconcurrency import CoreML
import Foundation
import Testing
@testable import SwiftWhisperKit
import SwiftWhisperCore

/// Coverage-only tests for stubs that throw `.notImplemented` or return zero
/// values. They do not assert behaviour beyond the stub contract; once the
/// real implementations land in M4/M5 these get rewritten or replaced.
@Suite("Stub coverage")
struct StubTests {

    // MARK: - WhisperEncoder

    @Test("WhisperEncoder.encode throws notImplemented")
    func encoderThrows() async throws {
        let encoder = WhisperEncoder()
        let mel = try MelSpectrogramResult(frames: [], nMels: 0, nFrames: 0)
        do {
            _ = try await encoder.encode(spectrogram: mel)
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            #expect(error == .notImplemented)
        }
    }

    // MARK: - WhisperDecoder

    @Test("WhisperDecoder.decode throws notImplemented")
    func decoderThrows() async throws {
        let decoder = WhisperDecoder()
        // Build a minimal valid MLMultiArray so we get past the type check.
        let array = try MLMultiArray(shape: [1], dataType: .float32)
        do {
            _ = try await decoder.decode(encoderOutput: array, options: .default)
            Issue.record("expected throw")
        } catch {
            #expect(error == .notImplemented)
        }
    }

    // MARK: - WhisperTokenizer

    @Test("WhisperTokenizer stub returns empty/false values")
    func tokenizerStub() async {
        let tok = WhisperTokenizer()
        await #expect(tok.encode(text: "hello") == [])
        await #expect(tok.decode(tokens: [1, 2, 3]) == "")
        await #expect(tok.isTimestamp(token: 0) == false)
        await #expect(tok.isSpecial(token: 0) == false)
        await #expect(tok.endOfTextToken == 50_257)
        await #expect(tok.startOfTranscriptToken == 50_258)
    }

    // MARK: - SileroVAD

    @Test("SileroVAD stub always reports silence")
    func sileroStub() async {
        let vad = SileroVAD()
        let chunk = AudioChunk(samples: [0.5, 0.5], sampleRate: 16_000, timestamp: 0)
        let result = await vad.isSpeech(chunk: chunk)
        #expect(result == false)
        await vad.reset()
    }
}
