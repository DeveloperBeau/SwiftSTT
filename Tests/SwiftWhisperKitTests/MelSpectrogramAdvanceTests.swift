import Foundation
import Testing
@testable import SwiftWhisperKit
import SwiftWhisperCore

@Suite("MelSpectrogram rolling window")
struct MelSpectrogramAdvanceTests {

    private static func sine(samples: Int) -> AudioChunk {
        let buf = (0..<samples).map { i in
            Float(0.5) * sinf(2 * .pi * 440 * Float(i) / 16_000)
        }
        return AudioChunk(samples: buf, sampleRate: 16_000, timestamp: 0)
    }

    @Test("advance(0) is no-op")
    func advanceZeroIsNoOp() async throws {
        let mel = try MelSpectrogram()
        _ = try await mel.process(chunk: Self.sine(samples: 16_000))
        let before = await mel.currentFrameCount()
        await mel.advance(framesConsumed: 0)
        let after = await mel.currentFrameCount()
        #expect(before == after)
        #expect(after > 0)
    }

    @Test("advance(N) drops first N frames")
    func advanceDropsFrames() async throws {
        let mel = try MelSpectrogram()
        _ = try await mel.process(chunk: Self.sine(samples: 16_000))
        let before = await mel.currentFrameCount()
        await mel.advance(framesConsumed: 10)
        let after = await mel.currentFrameCount()
        #expect(after == before - 10)
    }

    @Test("advance(currentFrameCount) empties buffer")
    func advanceAllEmpties() async throws {
        let mel = try MelSpectrogram()
        _ = try await mel.process(chunk: Self.sine(samples: 16_000))
        let count = await mel.currentFrameCount()
        await mel.advance(framesConsumed: count)
        let after = await mel.currentFrameCount()
        #expect(after == 0)
    }

    @Test("advance beyond currentFrameCount clamps")
    func advanceBeyondClamps() async throws {
        let mel = try MelSpectrogram()
        _ = try await mel.process(chunk: Self.sine(samples: 16_000))
        await mel.advance(framesConsumed: 10_000_000)
        let after = await mel.currentFrameCount()
        #expect(after == 0)
    }

    @Test("snapshot returns a copy unaffected by advance")
    func snapshotIsCopy() async throws {
        let mel = try MelSpectrogram()
        _ = try await mel.process(chunk: Self.sine(samples: 16_000))
        let snap = try await mel.snapshot()
        let snapFrames = snap.nFrames
        await mel.advance(framesConsumed: snapFrames)
        let after = await mel.currentFrameCount()
        #expect(after == 0)
        #expect(snap.nFrames == snapFrames)
        #expect(snap.frames.count == snap.nMels * snapFrames)
    }

    @Test("currentFrameCount reflects state after add and advance")
    func currentFrameCountTracks() async throws {
        let mel = try MelSpectrogram()
        let initial = await mel.currentFrameCount()
        #expect(initial == 0)

        _ = try await mel.process(chunk: Self.sine(samples: 16_000))
        let afterAdd = await mel.currentFrameCount()
        #expect(afterAdd == 98)

        await mel.advance(framesConsumed: 30)
        let afterAdvance = await mel.currentFrameCount()
        #expect(afterAdvance == 68)

        _ = try await mel.process(chunk: Self.sine(samples: 16_000))
        let afterSecondAdd = await mel.currentFrameCount()
        #expect(afterSecondAdd == 68 + 100)
    }

    @Test("reset clears rolling buffer")
    func resetClearsRolling() async throws {
        let mel = try MelSpectrogram()
        _ = try await mel.process(chunk: Self.sine(samples: 16_000))
        await mel.reset()
        let count = await mel.currentFrameCount()
        #expect(count == 0)
        let snap = try await mel.snapshot()
        #expect(snap.nFrames == 0)
        #expect(snap.frames.isEmpty)
    }

    @Test("snapshot layout matches process output for a single batch")
    func snapshotLayoutMatchesProcess() async throws {
        let mel = try MelSpectrogram()
        let processed = try await mel.process(chunk: Self.sine(samples: 16_000))
        let snap = try await mel.snapshot()
        #expect(snap.nFrames == processed.nFrames)
        #expect(snap.nMels == processed.nMels)
        #expect(snap.frames == processed.frames)
    }
}
