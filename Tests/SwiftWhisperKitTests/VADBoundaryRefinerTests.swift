import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

@Suite("VADBoundaryRefiner")
struct VADBoundaryRefinerTests {

    // 512 samples at 16 kHz = 32 ms per frame.
    static let frameSamples = 512

    @Test("Single isSpeech=true emits no boundary (need rising + falling)")
    func singleSpeechFrameEmitsNothing() async {
        let refiner = VADBoundaryRefiner(startConsecutive: 3, endConsecutive: 5)
        let boundary = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        #expect(boundary == nil)
    }

    @Test("Rising + falling sequence emits one boundary")
    func risingAndFallingEmitsBoundary() async {
        let refiner = VADBoundaryRefiner(startConsecutive: 2, endConsecutive: 2)
        // Rising: two speech frames committee start.
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        // Inside speech.
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        // Falling: two silence frames close the segment.
        _ = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        let final = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)

        guard let boundary = final else {
            Issue.record("Expected a boundary")
            return
        }
        #expect(boundary.startTime >= 0)
        #expect(boundary.endTime > boundary.startTime)
    }

    @Test("Hysteresis: 1-frame speech among silence does not trigger start")
    func singleSpeechFrameSuppressed() async {
        let refiner = VADBoundaryRefiner(startConsecutive: 3, endConsecutive: 3)
        _ = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)  // start of pending
        _ = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)  // resets streak
        _ = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        // No boundary should ever be emitted because we never reached startConsecutive speech in a row.
        let result = await refiner.flush()
        #expect(result == nil)
    }

    @Test("Hysteresis: 1-frame silence among speech does not trigger end")
    func singleSilenceFrameSuppressed() async {
        let refiner = VADBoundaryRefiner(startConsecutive: 2, endConsecutive: 3)
        // Establish speech.
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        // Inject one silence followed by speech: should not close.
        let after1Silence = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        #expect(after1Silence == nil)
        let backToSpeech = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        #expect(backToSpeech == nil)

        // Now flush should emit an in-progress boundary.
        let flushed = await refiner.flush()
        #expect(flushed != nil)
    }

    @Test("Boundary timestamps reflect accumulated audio time")
    func timestampsReflectAudioTime() async {
        let refiner = VADBoundaryRefiner(startConsecutive: 1, endConsecutive: 1, sampleRate: 16_000)
        // Ingest one silence frame (32ms), then a speech frame (start at 0.032), then silence.
        _ = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        let boundary = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)

        guard let boundary else {
            Issue.record("Expected boundary")
            return
        }
        let frameDuration = Double(Self.frameSamples) / 16_000
        #expect(abs(boundary.startTime - frameDuration) < 1e-9)
        // End is at the start of the silence frame which is 2 * frameDuration.
        #expect(abs(boundary.endTime - 2 * frameDuration) < 1e-9)
    }

    @Test("flush() emits in-progress boundary if currently in-speech")
    func flushEmitsInProgress() async {
        let refiner = VADBoundaryRefiner(startConsecutive: 1, endConsecutive: 5)
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        let final = await refiner.flush()
        guard let final else {
            Issue.record("Expected flush boundary")
            return
        }
        #expect(final.startTime == 0)
        #expect(final.endTime > 0)
    }

    @Test("flush() returns nil if currently in-silence")
    func flushReturnsNilInSilence() async {
        let refiner = VADBoundaryRefiner(startConsecutive: 3, endConsecutive: 1)
        _ = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        _ = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        let result = await refiner.flush()
        #expect(result == nil)
    }

    @Test("reset() clears state and elapsed time")
    func resetClearsState() async {
        let refiner = VADBoundaryRefiner(startConsecutive: 1, endConsecutive: 1)
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        _ = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        await refiner.reset()

        // After reset, a single speech then silence at startConsecutive=1 emits a boundary at t=0.
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        let boundary = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        guard let boundary else {
            Issue.record("Expected boundary after reset")
            return
        }
        #expect(boundary.startTime == 0)
    }

    @Test("Multiple back-to-back speech segments emit multiple boundaries")
    func multipleBoundaries() async {
        let refiner = VADBoundaryRefiner(startConsecutive: 1, endConsecutive: 1)
        var boundaries: [SpeechBoundary] = []

        // Segment 1: speech, silence
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        if let b = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples) {
            boundaries.append(b)
        }
        // Gap
        _ = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        // Segment 2: speech, silence
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        if let b = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples) {
            boundaries.append(b)
        }

        #expect(boundaries.count == 2)
        #expect(boundaries[0].endTime <= boundaries[1].startTime)
    }

    @Test("Custom thresholds tighten behaviour")
    func customThresholds() async {
        // startConsecutive=5 requires a long warmup before committing to speech.
        let refiner = VADBoundaryRefiner(startConsecutive: 5, endConsecutive: 1)
        for _ in 0..<4 {
            _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        }
        // Not yet committed.
        let flushedBefore = await refiner.flush()
        #expect(flushedBefore == nil)

        // 5th speech frame commits.
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        let flushedAfter = await refiner.flush()
        #expect(flushedAfter != nil)
    }

    @Test("Init clamps startConsecutive and endConsecutive to at least 1")
    func initClampsToOne() async {
        let refiner = VADBoundaryRefiner(startConsecutive: 0, endConsecutive: -3)
        // Effective startConsecutive=1, endConsecutive=1.
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        let boundary = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        #expect(boundary != nil)
    }

    @Test("Pending end resets when speech resumes mid-silence-streak")
    func pendingEndResets() async {
        let refiner = VADBoundaryRefiner(startConsecutive: 1, endConsecutive: 3)
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        _ = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        _ = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        // Speech resumes - end streak resets.
        _ = await refiner.ingest(isSpeech: true, sampleCount: Self.frameSamples)
        // Now need 3 more silent in a row to close.
        let silence1 = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        let silence2 = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        #expect(silence1 == nil)
        #expect(silence2 == nil)
        let silence3 = await refiner.ingest(isSpeech: false, sampleCount: Self.frameSamples)
        #expect(silence3 != nil)
    }
}
