@preconcurrency import AVFoundation
import Foundation
import SwiftSTTCore
import SwiftSTTKit
import Testing

/// Locations of the committed accuracy fixture.
private enum AccuracyFixture {
    static var audioURL: URL {
        get throws {
            try #require(
                Bundle.module.url(
                    forResource: "jfk", withExtension: "wav", subdirectory: "Fixtures"))
        }
    }

    static var expectedTranscript: String {
        get throws {
            let url = try #require(
                Bundle.module.url(
                    forResource: "jfk", withExtension: "txt", subdirectory: "Fixtures"))
            return try String(contentsOf: url, encoding: .utf8)
        }
    }
}

/// Checks the committed fixture is in the format the file input expects.
///
/// Runs in the default gate. It reads only the committed file.
@Suite("Accuracy fixture format")
struct AccuracyFixtureFormatTests {

    @Test("jfk.wav is 11 seconds of 16 kHz mono 16-bit PCM")
    func fixtureFormat() throws {
        let file = try AVAudioFile(forReading: try AccuracyFixture.audioURL)
        let format = file.fileFormat
        #expect(format.sampleRate == 16_000)
        #expect(format.channelCount == 1)
        #expect(format.streamDescription.pointee.mFormatID == kAudioFormatLinearPCM)
        #expect(format.streamDescription.pointee.mBitsPerChannel == 16)
        #expect(file.length == 176_000)
    }
}

/// Transcribes the committed fixture with the tiny model and scores it against the
/// hand-written transcript.
///
/// Disabled by default, like ``IntegrationTests``. Run with:
///
/// ```
/// SWIFTSTT_RUN_INTEGRATION=1 swift test --filter SwiftSTTIntegrationTests
/// ```
///
/// The only network use is the model download when the tiny model is not cached.
/// The audio is decoded on this machine.
@Suite(
    "Accuracy fixture",
    .disabled(if: ProcessInfo.processInfo.environment["SWIFTSTT_RUN_INTEGRATION"] == nil)
)
struct AccuracyFixtureTests {

    /// The tiny model's word error rate on `jfk.wav` must stay below this.
    ///
    /// The reference has 22 words, so 0.10 allows at most two word errors.
    static let wordErrorRateThreshold = 0.10

    @Test(
        "Tiny model transcribes the committed fixture within the expected word error rate",
        .timeLimit(.minutes(10))
    )
    func tinyModelTranscribesFixture() async throws {
        // The threshold is set for the tiny model, so this test ignores
        // SWIFTSTT_INTEGRATION_MODEL.
        let model = WhisperModel.tiny
        let downloader = ModelDownloader()
        if await !downloader.isDownloaded(model) {
            let stream = try await downloader.download(model)
            for try await progress in stream {
                if progress.phase == .complete { break }
            }
        }
        let bundle = try await downloader.bundle(for: model)
        let context = try WhisperCppContext(ggmlModelURL: bundle.ggmlModelURL)

        let samples = try await readAllSamples(at: try AccuracyFixture.audioURL)
        #expect(samples.count == 176_000)

        var options = DecodingOptions.default
        options.language = "en"
        let segments = try await context.transcribe(samples: samples, options: options)

        let hypothesis = segments.map(\.text).joined(separator: " ")
        let reference = try AccuracyFixture.expectedTranscript
        let wer = WordErrorRate(reference: reference, hypothesis: hypothesis)
        let percent = (wer.rate * 100).formatted(.number.precision(.fractionLength(1)))
        print(
            "jfk.wav tiny model WER: \(percent)% (\(wer.errors) errors in"
                + " \(wer.referenceWordCount) words, threshold"
                + " \(Int(Self.wordErrorRateThreshold * 100))%)"
        )
        print("jfk.wav hypothesis: \(hypothesis.trimmingCharacters(in: .whitespaces))")

        #expect(
            wer.rate < Self.wordErrorRateThreshold,
            "WER \(percent)% is not below the threshold. Hypothesis: \(hypothesis)"
        )
    }

    /// Reads the file through the public file input at the 16 kHz rate the context takes.
    private func readAllSamples(at url: URL) async throws -> [Float] {
        let (chunks, continuation) = AsyncStream<[Float]>.makeStream()
        let input = AudioFileInput(fileURL: url)
        try await input.start(targetSampleRate: 16_000, bufferDurationSeconds: 1.0) { chunk in
            continuation.yield(chunk)
        }
        Task {
            await input.waitUntilComplete()
            await input.stop()
            continuation.finish()
        }
        var samples: [Float] = []
        for await chunk in chunks {
            samples.append(contentsOf: chunk)
        }
        return samples
    }
}
