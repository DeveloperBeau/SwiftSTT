@preconcurrency import AVFoundation
import Foundation
import SwiftWhisperCore
import Synchronization
import Testing

@testable import SwiftWhisperKit

@Suite("AudioFileInput format coverage")
struct AudioFormatTests {

    /// Thread-safe collector for the @Sendable onChunk callback.
    private final class ChunkCollector: @unchecked Sendable {
        private let mutex = Mutex<[[Float]]>([])
        func append(_ samples: [Float]) { mutex.withLock { $0.append(samples) } }
        var totalSamples: Int { mutex.withLock { $0.reduce(0) { $0 + $1.count } } }
    }

    /// Synthesizes a 440 Hz sine wave and writes it through `AVAudioFile` in
    /// the requested container/codec settings.
    ///
    /// Returns the file URL.
    static func writeSine(
        sampleRate: Double,
        durationSeconds: Double,
        settings: [String: Any],
        ext: String
    ) throws -> URL {
        let frames = Int(sampleRate * durationSeconds)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("audio-format-\(UUID().uuidString).\(ext)")

        guard
            let pcmFormat = AVAudioFormat(
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
            settings: settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )

        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: pcmFormat,
                frameCapacity: AVAudioFrameCount(frames)
            )
        else {
            throw NSError(domain: "test", code: 2)
        }
        buffer.frameLength = AVAudioFrameCount(frames)
        let channel = buffer.floatChannelData![0]
        for i in 0..<frames {
            channel[i] = Float(0.25 * sin(2 * .pi * 440 * Double(i) / sampleRate))
        }
        try file.write(from: buffer)
        return url
    }

    @Test("AIFF input is decoded and emits chunks at the target rate")
    func aiffDecodes() async throws {
        let sampleRate: Double = 16_000
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsBigEndianKey: true,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        let url = try Self.writeSine(
            sampleRate: sampleRate,
            durationSeconds: 0.5,
            settings: settings,
            ext: "aiff"
        )
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

        let expected = Int(sampleRate * 0.5)
        let total = collector.totalSamples
        #expect(abs(total - expected) <= 256)
    }

    @Test("CAF input is decoded and emits chunks")
    func cafDecodes() async throws {
        let sampleRate: Double = 16_000
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        let url = try Self.writeSine(
            sampleRate: sampleRate,
            durationSeconds: 0.25,
            settings: settings,
            ext: "caf"
        )
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

        #expect(collector.totalSamples > 0)
    }

    @Test("M4A input decodes when platform codec is available", .disabled(if: !m4aWritable()))
    func m4aDecodes() async throws {
        let sampleRate: Double = 16_000
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 32_000,
        ]
        let url: URL
        do {
            url = try Self.writeSine(
                sampleRate: sampleRate,
                durationSeconds: 0.5,
                settings: settings,
                ext: "m4a"
            )
        } catch {
            // Codec unavailable on this host. Skip rather than fail.
            return
        }
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

        #expect(collector.totalSamples > 0)
    }

    /// Probes whether `AVAudioFile` can write an AAC `.m4a` on this host.
    ///
    /// The CI runner usually can; some sandboxed environments cannot. Used to gate
    /// the M4A test case so unavailable codecs don't fail the suite.
    static func m4aWritable() -> Bool {
        let probeURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("audio-format-probe-\(UUID().uuidString).m4a")
        defer { try? FileManager.default.removeItem(at: probeURL) }
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: 16_000,
            AVNumberOfChannelsKey: 1,
            AVEncoderBitRateKey: 32_000,
        ]
        do {
            _ = try AVAudioFile(
                forWriting: probeURL,
                settings: settings,
                commonFormat: .pcmFormatFloat32,
                interleaved: false
            )
            return true
        } catch {
            return false
        }
    }
}
