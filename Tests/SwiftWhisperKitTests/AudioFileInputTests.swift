@preconcurrency import AVFoundation
import Foundation
import Synchronization
import Testing
@testable import SwiftWhisperKit
import SwiftWhisperCore

@Suite("AudioFileInput")
struct AudioFileInputTests {

    /// Writes a synthetic mono Float32 WAV to a temp file and returns the URL.
    /// The waveform is a 440 Hz sine so the test data is non-trivial.
    static func makeSineWave(
        sampleRate: Double,
        durationSeconds: Double,
        frequencyHz: Double = 440
    ) throws -> URL {
        let frames = Int(sampleRate * durationSeconds)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("audio-file-input-\(UUID().uuidString).wav")

        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: sampleRate,
                channels: 1,
                interleaved: false
            )
        else {
            throw NSError(domain: "test", code: 1)
        }

        let file = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )

        guard
            let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames))
        else {
            throw NSError(domain: "test", code: 2)
        }
        buffer.frameLength = AVAudioFrameCount(frames)
        let channel = buffer.floatChannelData![0]
        for i in 0..<frames {
            channel[i] = Float(0.25 * sin(2 * .pi * frequencyHz * Double(i) / sampleRate))
        }
        try file.write(from: buffer)
        return url
    }

    /// Thread-safe collector for the @Sendable onChunk callback.
    private final class ChunkCollector: @unchecked Sendable {
        private let mutex = Mutex<[[Float]]>([])
        func append(_ samples: [Float]) { mutex.withLock { $0.append(samples) } }
        var chunks: [[Float]] { mutex.withLock { $0 } }
        var totalSamples: Int { mutex.withLock { $0.reduce(0) { $0 + $1.count } } }
    }

    @Test("Emits chunks at the requested sample rate and reaches EOF")
    func emitsChunksAndCompletes() async throws {
        let sampleRate: Double = 16_000
        let url = try Self.makeSineWave(sampleRate: sampleRate, durationSeconds: 0.5)
        defer { try? FileManager.default.removeItem(at: url) }

        let input = AudioFileInput(fileURL: url)
        let collector = ChunkCollector()

        try await input.start(
            targetSampleRate: sampleRate,
            bufferDurationSeconds: 0.064
        ) { @Sendable samples in
            collector.append(samples)
        }
        await input.waitUntilComplete()
        await input.stop()

        #expect(!collector.chunks.isEmpty)
        let expected = Int(sampleRate * 0.5)
        let total = collector.totalSamples
        #expect(abs(total - expected) <= 64)
    }

    @Test("waitUntilComplete returns after EOF")
    func waitUntilCompleteReturns() async throws {
        let url = try Self.makeSineWave(sampleRate: 16_000, durationSeconds: 0.25)
        defer { try? FileManager.default.removeItem(at: url) }

        let input = AudioFileInput(fileURL: url)
        try await input.start(
            targetSampleRate: 16_000,
            bufferDurationSeconds: 0.064
        ) { @Sendable _ in }

        await input.waitUntilComplete()
        await input.stop()
    }

    @Test("stop cancels mid-stream")
    func stopCancelsMidStream() async throws {
        let url = try Self.makeSineWave(sampleRate: 16_000, durationSeconds: 5.0)
        defer { try? FileManager.default.removeItem(at: url) }

        let input = AudioFileInput(fileURL: url)
        let collector = ChunkCollector()

        try await input.start(
            targetSampleRate: 16_000,
            bufferDurationSeconds: 0.064
        ) { @Sendable samples in
            collector.append(samples)
        }

        await input.stop()
        let countAfterStop = collector.chunks.count
        try await Task.sleep(for: .milliseconds(100))
        #expect(collector.chunks.count == countAfterStop)
    }

    @Test("Missing file throws audioCaptureFailed")
    func missingFileThrows() async {
        let url = URL(fileURLWithPath: "/nonexistent/path/missing-audio-\(UUID()).wav")
        let input = AudioFileInput(fileURL: url)

        do {
            try await input.start(
                targetSampleRate: 16_000,
                bufferDurationSeconds: 0.064
            ) { @Sendable _ in }
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            switch error {
            case .audioCaptureFailed:
                break
            default:
                Issue.record("expected audioCaptureFailed, got \(error)")
            }
        }
    }

    @Test("Double start is a no-op")
    func doubleStart() async throws {
        let url = try Self.makeSineWave(sampleRate: 16_000, durationSeconds: 0.1)
        defer { try? FileManager.default.removeItem(at: url) }

        let input = AudioFileInput(fileURL: url)
        try await input.start(
            targetSampleRate: 16_000,
            bufferDurationSeconds: 0.064
        ) { @Sendable _ in }
        try await input.start(
            targetSampleRate: 16_000,
            bufferDurationSeconds: 0.064
        ) { @Sendable _ in }
        await input.waitUntilComplete()
        await input.stop()
    }
}
