@preconcurrency import AVFoundation
import Foundation
import SwiftSTTCore
import SwiftSTTKit
import Testing

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

    /// The tiny model, cached or downloaded before the test body runs.
    ///
    /// Set up here so a missing model is downloaded or fails the test. The body has no
    /// way to see an uncached model.
    private let prepared: PreparedModel

    init() async throws {
        // The threshold is set for the tiny model, so this suite ignores
        // SWIFTSTT_INTEGRATION_MODEL.
        prepared = try await PreparedModel.prepare(.tiny, using: ModelDownloader())
    }

    @Test(
        "Tiny model transcribes the committed fixture within the expected word error rate",
        .timeLimit(.minutes(10))
    )
    func tinyModelTranscribesFixture() async throws {
        #expect(FileManager.default.fileExists(atPath: prepared.bundle.ggmlModelURL.path))

        // The confirmation fails the test if the body returns before scoring.
        try await confirmation("fixture decoded and scored", expectedCount: 1) { scored in
            let context = try WhisperCppContext(ggmlModelURL: prepared.bundle.ggmlModelURL)

            let samples = try await readAllSamples(at: try AccuracyFixture.audioURL)
            #expect(samples.count == 176_000)

            var options = DecodingOptions.default
            options.language = "en"
            let segments = try await context.transcribe(samples: samples, options: options)
            try #require(!segments.isEmpty, "Decoding produced no segments")

            let hypothesis = segments.map(\.text).joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            try #require(!hypothesis.isEmpty, "Decoding produced no text")

            let score = try AccuracyFixtureScore(hypothesis: hypothesis)
            let wer = score.wordErrorRate
            let percent = (wer.rate * 100).formatted(.number.precision(.fractionLength(1)))
            print(
                "jfk.wav tiny model WER: \(percent)% (\(wer.errors) errors in"
                    + " \(wer.referenceWordCount) words, threshold"
                    + " \(Int(AccuracyFixtureScore.wordErrorRateThreshold * 100))%)"
            )
            print("jfk.wav hypothesis: \(hypothesis)")

            #expect(
                score.reference
                    == "And so my fellow Americans, ask not what your country can do for you,"
                    + " ask what you can do for your country.\n"
            )
            #expect(wer.referenceWordCount == 22)
            #expect(
                score.meetsThreshold,
                "WER \(percent)% is not below the threshold. Hypothesis: \(hypothesis)"
            )
            scored()
        }
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
