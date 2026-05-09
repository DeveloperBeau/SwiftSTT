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
