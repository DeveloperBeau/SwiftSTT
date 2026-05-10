@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Synchronization
import Testing

@testable import SwiftWhisperKit

// MARK: - Test doubles

/// Audio input that delivers all configured chunks during `start()`.
private actor SyncAudioInput: AudioInputProvider {

    private let chunks: [[Float]]
    private(set) var stopCount = 0

    init(chunks: [[Float]]) {
        self.chunks = chunks
    }

    func start(
        targetSampleRate: Double,
        bufferDurationSeconds: Double,
        onChunk: @Sendable @escaping ([Float]) -> Void
    ) async throws(SwiftWhisperError) {
        for chunk in chunks {
            onChunk(chunk)
        }
    }

    func stop() async {
        stopCount += 1
    }
}

private actor AlwaysSpeechVAD: VoiceActivityDetector {
    func isSpeech(chunk: AudioChunk) async -> Bool { true }
    func reset() async {}
}

/// `MelSpectrogramProcessor` that succeeds at `process` (so the pipeline
/// reaches `runDecodeCycle`) but fails at `snapshot()`. Used to cover the
/// otherwise-unreachable mel snapshot error branch.
private actor FailingSnapshotMel: MelSpectrogramProcessor {

    private let nMels: Int
    private var frames: Int = 0

    init(nMels: Int = 80) {
        self.nMels = nMels
    }

    func process(chunk: AudioChunk) async throws(SwiftWhisperError) -> MelSpectrogramResult {
        // Record one frame so the pipeline thinks there is content to decode.
        frames += 1
        return try MelSpectrogramResult(frames: [], nMels: nMels, nFrames: 0)
    }

    func reset() async {
        frames = 0
    }

    func currentFrameCount() async -> Int { frames }

    func snapshot() async throws(SwiftWhisperError) -> MelSpectrogramResult {
        throw .invalidMelDimensions(framesCount: 0, expected: 1)
    }

    func advance(framesConsumed: Int) async {
        frames = max(0, frames - framesConsumed)
    }
}

// MARK: - Mock CoreML runners (mirrors of what TranscriptionPipelineTests use)

private final class FailingEncoderRunner: CoreMLModelRunner, @unchecked Sendable {
    func predict(features: any MLFeatureProvider) async throws(SwiftWhisperError)
        -> any MLFeatureProvider
    {
        throw .decoderFailure("encoder boom")
    }
}

private final class FailingDecoderRunner: StatefulCoreMLModelRunner, @unchecked Sendable {
    func resetState() async {}
    func predict(features: any MLFeatureProvider) async throws(SwiftWhisperError)
        -> any MLFeatureProvider
    {
        throw .decoderFailure("decoder boom")
    }
}

/// Encoder runner that returns a valid (but trivial) embedding.
private final class TrivialEncoderRunner: CoreMLModelRunner, @unchecked Sendable {
    func predict(features: any MLFeatureProvider) async throws(SwiftWhisperError)
        -> any MLFeatureProvider
    {
        do {
            let array = try MLMultiArray(shape: [1, 1500, 16], dataType: .float32)
            let count = 1500 * 16
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: count)
            for i in 0..<count { pointer[i] = 0 }
            return try MLDictionaryFeatureProvider(dictionary: [
                WhisperEncoder.outputFeatureName: array
            ])
        } catch {
            throw .decoderFailure("trivial encoder: \(error.localizedDescription)")
        }
    }
}

// MARK: - Helpers

private let vocabSize = 50_400

private let defaultSpecials: [String: Int] = [
    "<|endoftext|>": 50_257,
    "<|startoftranscript|>": 50_258,
    "<|en|>": 50_259,
    "<|transcribe|>": 50_359,
    "<|translate|>": 50_358,
    "<|notimestamps|>": 50_363,
]

private func makeTokenizer() -> WhisperTokenizer {
    WhisperTokenizer(specialTokens: defaultSpecials)
}

private func speech(count: Int = 512) -> [Float] {
    (0..<count).map { Float(sin(Double($0) * 0.1)) * 0.5 }
}

private actor SegmentCollector {
    var segments: [TranscriptionSegment] = []

    func append(_ s: TranscriptionSegment) { segments.append(s) }
}

@Suite("TranscriptionPipeline error paths")
struct TranscriptionPipelineErrorTests {

    @Test("Encoder failure terminates the stream")
    func encoderFailureTerminatesStream() async throws {
        let tokenizer = makeTokenizer()
        let encoder = WhisperEncoder(runner: FailingEncoderRunner())
        let decoder = WhisperDecoder(
            runner: FailingDecoderRunner(),
            tokenizer: tokenizer
        )
        let pipeline = TranscriptionPipeline(
            audioInput: SyncAudioInput(chunks: [speech(), speech()]),
            vad: AlwaysSpeechVAD(),
            melSpectrogram: try MelSpectrogram(),
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: LocalAgreementPolicy(),
            decodeIntervalSeconds: 0
        )

        let stream = try await pipeline.start()
        let collector = SegmentCollector()
        let consume = Task {
            for await s in stream { await collector.append(s) }
        }
        try await Task.sleep(for: .milliseconds(500))
        await pipeline.stop()
        await consume.value

        let segments = await collector.segments
        #expect(segments.isEmpty)
    }

    @Test("Decoder failure terminates the stream")
    func decoderFailureTerminatesStream() async throws {
        let tokenizer = makeTokenizer()
        let encoder = WhisperEncoder(runner: TrivialEncoderRunner())
        let decoder = WhisperDecoder(
            runner: FailingDecoderRunner(),
            tokenizer: tokenizer
        )
        let pipeline = TranscriptionPipeline(
            audioInput: SyncAudioInput(chunks: [speech(), speech()]),
            vad: AlwaysSpeechVAD(),
            melSpectrogram: try MelSpectrogram(),
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: LocalAgreementPolicy(),
            decodeIntervalSeconds: 0
        )

        let stream = try await pipeline.start()
        let collector = SegmentCollector()
        let consume = Task {
            for await s in stream { await collector.append(s) }
        }
        try await Task.sleep(for: .milliseconds(500))
        await pipeline.stop()
        await consume.value

        let segments = await collector.segments
        #expect(segments.isEmpty)
    }

    @Test("Mel snapshot failure terminates the stream")
    func melSnapshotFailureTerminatesStream() async throws {
        let tokenizer = makeTokenizer()
        let encoder = WhisperEncoder(runner: TrivialEncoderRunner())
        let decoder = WhisperDecoder(
            runner: FailingDecoderRunner(),
            tokenizer: tokenizer
        )
        let pipeline = TranscriptionPipeline(
            audioInput: SyncAudioInput(chunks: [speech()]),
            vad: AlwaysSpeechVAD(),
            melSpectrogram: FailingSnapshotMel(),
            encoder: encoder,
            decoder: decoder,
            tokenizer: tokenizer,
            policy: LocalAgreementPolicy(),
            decodeIntervalSeconds: 0,
            maxBufferedFrames: 1
        )

        let stream = try await pipeline.start()
        let collector = SegmentCollector()
        let consume = Task {
            for await s in stream { await collector.append(s) }
        }
        try await Task.sleep(for: .milliseconds(500))
        await pipeline.stop()
        await consume.value

        let segments = await collector.segments
        #expect(segments.isEmpty)
    }
}
