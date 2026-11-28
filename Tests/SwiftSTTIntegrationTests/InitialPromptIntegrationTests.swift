import AVFoundation
import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTKit

/// Real-model decodes of the committed `prompt-zoltan-szabo.wav` fixture.
///
/// The fixture is macOS `say -v Samantha` speaking "Please ask Zoltan Szabo to review the
/// Kubernetes cluster.", converted with `afconvert -f WAVE -d LEI16@16000 -c 1`. The expected
/// strings are the tiny model's output, written down by hand from a run of the unmodified decoder.
///
/// Disabled by default. Run with `SWIFTSTT_RUN_INTEGRATION=1 swift test --filter InitialPromptIntegration`.
@Suite(
    "InitialPromptIntegration",
    .disabled(if: ProcessInfo.processInfo.environment["SWIFTSTT_RUN_INTEGRATION"] == nil),
    .timeLimit(.minutes(10))
)
struct InitialPromptIntegrationTests {

    /// The tiny model's text for the fixture with no prompt, recorded before the prompt field existed.
    static let noPromptGolden = " Please ask Soltons Abbo to review the Kubernetes cluster."

    @Test("A decode with default options matches the text recorded before prompts existed")
    func defaultOptionsMatchRecordedGolden() async throws {
        let context = try await tinyContext()
        let samples = try PromptFixture.samples()

        let text = try await decodedText(context, samples, DecodingOptions())

        #expect(text == Self.noPromptGolden)
    }

    // MARK: - Helpers

    private func decodedText(
        _ context: WhisperCppContext,
        _ samples: [Float],
        _ options: DecodingOptions
    ) async throws -> String {
        try await context.transcribe(samples: samples, options: options)
            .map(\.text)
            .joined()
    }

    /// Loads the tiny model through the same download flow as the main integration suite.
    private func tinyContext() async throws -> WhisperCppContext {
        let downloader = ModelDownloader()
        if await !downloader.isDownloaded(.tiny) {
            let stream = try await downloader.download(.tiny)
            for try await progress in stream {
                if progress.phase == .complete { break }
            }
        }
        let bundle = try await downloader.bundle(for: .tiny)
        return try WhisperCppContext(ggmlModelURL: bundle.ggmlModelURL)
    }
}

/// Reads the committed prompt fixture without any model.
enum PromptFixture {

    static let url = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .appendingPathComponent("Fixtures/prompt-zoltan-szabo.wav")

    /// Decodes the fixture into 16 kHz mono Float samples.
    static func samples() throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        guard format.sampleRate == 16_000, format.channelCount == 1,
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format,
                frameCapacity: AVAudioFrameCount(file.length)
            )
        else {
            throw FixtureError.unexpectedFormat
        }
        try file.read(into: buffer)
        guard let channel = buffer.floatChannelData?[0] else {
            throw FixtureError.unexpectedFormat
        }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    enum FixtureError: Error {
        case unexpectedFormat
    }
}

/// Checks the fixture itself, so a damaged file fails the default gate without a model.
@Suite("PromptFixture")
struct PromptFixtureTests {

    @Test("The prompt fixture is 16 kHz mono with the recorded length")
    func fixtureFormatAndLength() throws {
        let samples = try PromptFixture.samples()
        // afinfo reports 55697 packets of one Int16 sample each.
        #expect(samples.count == 55_697)
    }
}
