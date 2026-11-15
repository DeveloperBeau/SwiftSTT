import AVFoundation
import Testing

@testable import SwiftSTTKit

@Suite("OneShotFlag")
struct OneShotFlagTests {

    @Test("claim succeeds once, then fails")
    func claimsOnce() {
        let flag = OneShotFlag()
        #expect(flag.claim())
        #expect(!flag.claim())
        #expect(!flag.claim())
    }

    @Test("concurrent claims produce exactly one winner")
    func oneWinnerUnderContention() async {
        let flag = OneShotFlag()
        let winners = await withTaskGroup(of: Bool.self) { group in
            for _ in 0..<64 { group.addTask { flag.claim() } }
            return await group.reduce(0) { $0 + ($1 ? 1 : 0) }
        }
        #expect(winners == 1)
    }

    @Test("provide hands over the buffer once, then reports no data")
    func provideHandsOverOnce() throws {
        let format = try #require(AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 16))
        let flag = OneShotFlag()
        var status = AVAudioConverterInputStatus.endOfStream

        let first = flag.provide(buffer, status: &status)
        #expect(first === buffer)
        #expect(status == .haveData)

        let second = flag.provide(buffer, status: &status)
        #expect(second == nil)
        #expect(status == .noDataNow)
    }

    @Test("CaptureBox converts one tap buffer exactly once")
    func captureBoxConvertsOnce() throws {
        let source = try #require(AVAudioFormat(standardFormatWithSampleRate: 48_000, channels: 1))
        let target = try #require(AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1))
        let buffer = try #require(AVAudioPCMBuffer(pcmFormat: source, frameCapacity: 4_800))
        buffer.frameLength = 4_800
        let box = CaptureBox()
        box.converter = AVAudioConverter(from: source, to: target)
        box.outputFormat = target

        let samples = try #require(
            box.convert(buffer: buffer, sourceRate: 48_000, targetRate: 16_000))
        #expect(samples.count <= 1_600)
        #expect(samples.count > 1_000)
    }
}
