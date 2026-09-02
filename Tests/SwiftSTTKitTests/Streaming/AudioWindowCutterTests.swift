import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTKit

/// Returns a scripted verdict per call, `false` once the script runs out.
private actor ScriptedVoiceActivityDetector: VoiceActivityDetector {
    private let verdicts: [Bool]
    private var index = 0
    init(_ verdicts: [Bool]) { self.verdicts = verdicts }
    func isSpeech(chunk: AudioChunk) async -> Bool {
        defer { index += 1 }
        return index < verdicts.count && verdicts[index]
    }
    func reset() async { index = 0 }
}

private struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed &* 6_364_136_223_846_793_005 &+ 1 }
    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }
}

private let bufferSize = 1_600  // 0.1 s at 16 kHz

private func buffer(_ value: Float = 0.5) -> [Float] {
    Array(repeating: value, count: bufferSize)
}

private func makeCutter(
    policy: StreamingWindowPolicy,
    verdicts: [Bool]
) -> AudioWindowCutter {
    AudioWindowCutter(
        policy: policy,
        detector: ScriptedVoiceActivityDetector(verdicts),
        refiner: VADBoundaryRefiner(startConsecutive: 1, endConsecutive: 1, sampleRate: 16_000)
    )
}

@Suite("AudioWindowCutter")
struct AudioWindowCutterTests {

    @Test("C1: silence cut")
    func silenceCut() async throws {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 10, minimumWindowDuration: 0.5, overlapDuration: 1)
        let cutter = makeCutter(policy: policy, verdicts: [true, true, true, true, true, false])

        var lastWindow: AudioWindow?
        for _ in 0..<6 {
            lastWindow = await cutter.ingest(buffer())
        }

        let window = try #require(lastWindow)
        #expect(window.cut == .silence)
        #expect(window.samples.count == 9_600)
        #expect(window.startTime == 0)
    }

    @Test("C2: false-positive, the minimum guard suppresses the cut")
    func minimumGuardSuppressesCut() async {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 10, minimumWindowDuration: 2.0, overlapDuration: 1)
        let cutter = makeCutter(policy: policy, verdicts: [true, true, true, true, true, false])

        for _ in 0..<6 {
            let window = await cutter.ingest(buffer())
            #expect(window == nil)
        }
    }

    @Test("C3: false-negative, the minimum guard eventually allows the cut")
    func minimumGuardEventuallyAllowsCut() async throws {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 10, minimumWindowDuration: 2.0, overlapDuration: 1)
        var verdicts = Array(repeating: true, count: 25)
        verdicts.append(false)
        let cutter = makeCutter(policy: policy, verdicts: verdicts)

        var lastWindow: AudioWindow?
        for _ in 0..<26 {
            lastWindow = await cutter.ingest(buffer())
        }

        let window = try #require(lastWindow)
        #expect(window.cut == .silence)
        #expect(window.samples.count == 41_600)
    }

    @Test("C4: forced cut and carry-over")
    func forcedCutAndCarryOver() async throws {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 1.0, minimumWindowDuration: 0.5, overlapDuration: 0.3)
        let cutter = makeCutter(policy: policy, verdicts: Array(repeating: true, count: 20))

        var firstWindow: AudioWindow?
        for i in 0..<10 {
            let window = await cutter.ingest(buffer())
            if i < 9 {
                #expect(window == nil)
            } else {
                firstWindow = window
            }
        }
        let window1 = try #require(firstWindow)
        #expect(window1.cut == .maximumDuration)
        #expect(window1.samples.count == 16_000)
        #expect(window1.startTime == 0)

        var secondWindow: AudioWindow?
        for i in 0..<7 {
            let window = await cutter.ingest(buffer())
            if i < 6 {
                #expect(window == nil)
            } else {
                secondWindow = window
            }
        }
        let window2 = try #require(secondWindow)
        #expect(window2.startTime == 0.7)
        #expect(window2.samples.count == 16_000)
    }

    @Test("C5: false-positive, speechless audio never returns a window")
    func speechlessAudioNeverReturnsWindow() async {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 1.0, minimumWindowDuration: 0.5, overlapDuration: 0.3)
        let cutter = makeCutter(policy: policy, verdicts: Array(repeating: false, count: 20))

        for _ in 0..<20 {
            let window = await cutter.ingest(buffer())
            #expect(window == nil)
        }
        let flushed = await cutter.flush()
        #expect(flushed == nil)
    }

    @Test("C6: false-negative, dropped windows still advance windowStartTime")
    func droppedWindowsAdvanceStartTime() async throws {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 1.0, minimumWindowDuration: 0.5, overlapDuration: 0.3)
        var verdicts = Array(repeating: false, count: 20)
        verdicts.append(contentsOf: Array(repeating: true, count: 10))
        let cutter = makeCutter(policy: policy, verdicts: verdicts)

        var lastWindow: AudioWindow?
        for _ in 0..<30 {
            lastWindow = await cutter.ingest(buffer())
        }

        let window = try #require(lastWindow)
        #expect(window.cut == .maximumDuration)
        #expect(window.startTime == 2.0)
    }

    @Test("C7: flush ignores the minimum")
    func flushIgnoresMinimum() async throws {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 10, minimumWindowDuration: 5.0, overlapDuration: 1)
        let cutter = makeCutter(policy: policy, verdicts: [true, true, true])

        for _ in 0..<3 {
            let window = await cutter.ingest(buffer())
            #expect(window == nil)
        }

        let flushed = try #require(await cutter.flush())
        #expect(flushed.cut == .stop)
        #expect(flushed.samples.count == 4_800)
    }

    @Test("C8: failure case, flush with nothing buffered")
    func flushWithNothingBuffered() async {
        let policy = StreamingWindowPolicy.default
        let cutter = makeCutter(policy: policy, verdicts: [])
        let flushed = await cutter.flush()
        #expect(flushed == nil)
    }

    @Test("C9: failure case, flush with only silence buffered")
    func flushWithOnlySilenceBuffered() async {
        let policy = StreamingWindowPolicy.default
        let cutter = makeCutter(policy: policy, verdicts: Array(repeating: false, count: 5))

        for _ in 0..<5 {
            _ = await cutter.ingest(buffer())
        }
        let flushed = await cutter.flush()
        #expect(flushed == nil)
    }

    @Test("C11: ordering, silence beats duration when both fire on the same buffer")
    func silenceBeatsDurationOnTheSameBuffer() async throws {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 0.5, minimumWindowDuration: 0.2, overlapDuration: 0.2)
        let cutter = makeCutter(policy: policy, verdicts: [true, true, true, true, false, true])

        var lastWindow: AudioWindow?
        for _ in 0..<5 {
            lastWindow = await cutter.ingest(buffer())
        }
        let window = try #require(lastWindow)
        #expect(window.cut == .silence)

        // Sixth buffer starts the next window; flush it to observe startTime
        // without waiting for it to fill or close on its own. The silence cut
        // carried its last overlapDuration (0.2s) of the 0.5s window forward,
        // so the next window starts at 0.3 rather than at 0.5: the carried
        // audio is deliberately re-covered, which is what gives the model
        // context for the first word after the boundary.
        _ = await cutter.ingest(buffer())
        let flushed = try #require(await cutter.flush())
        #expect(flushed.startTime == 0.3)
    }

    @Test("C12: failure case, empty buffer")
    func emptyBufferIsIgnored() async throws {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 10, minimumWindowDuration: 0.5, overlapDuration: 1)
        let cutter = makeCutter(policy: policy, verdicts: [true, true, true, true, true, false])

        let emptyResult = await cutter.ingest([])
        #expect(emptyResult == nil)

        var lastWindow: AudioWindow?
        for _ in 0..<6 {
            lastWindow = await cutter.ingest(buffer())
        }
        let window = try #require(lastWindow)
        #expect(window.startTime == 0)
    }

    @Test("C13: failure case, a buffer longer than the whole window")
    func bufferLongerThanWholeWindow() async throws {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 1.0, minimumWindowDuration: 0.5, overlapDuration: 0.25)
        let cutter = makeCutter(policy: policy, verdicts: [true])

        let bigBuffer = Array(repeating: Float(0.5), count: 48_000)
        let window = try #require(await cutter.ingest(bigBuffer))
        #expect(window.samples.count == 48_000)
        #expect(window.cut == .maximumDuration)

        let flushed = try #require(await cutter.flush())
        #expect(flushed.startTime == 2.75)
    }

    @Test("C14: fuzz A, exact tiling, speech throughout")
    func fuzzExactTilingSpeechThroughout() async {
        var generator = SeededGenerator(seed: 0xF00D_BEEF)
        for iteration in 0..<200 {
            let bufferCount = Int.random(in: 1...400, using: &generator)
            let maximumWindowDuration = [0.5, 1.0, 2.0, 5.0].randomElement(using: &generator)!
            let minimumWindowDuration = [0.1, 0.5, 1.0].randomElement(using: &generator)!
            let overlapDuration = [0.1, 0.25, 0.5].randomElement(using: &generator)!
            let policy = StreamingWindowPolicy(
                maximumWindowDuration: maximumWindowDuration,
                minimumWindowDuration: minimumWindowDuration,
                overlapDuration: overlapDuration
            )
            let cutter = makeCutter(
                policy: policy, verdicts: Array(repeating: true, count: bufferCount))

            var expectedStart: TimeInterval = 0
            var totalIngested: TimeInterval = 0
            for _ in 0..<bufferCount {
                totalIngested += Double(bufferSize) / 16_000.0
                if let window = await cutter.ingest(buffer()) {
                    if abs(window.startTime - expectedStart) >= 1e-9 {
                        Issue.record(
                            "seed 0xF00D_BEEF iteration \(iteration): startTime \(window.startTime) != expected \(expectedStart), policy=\(policy)"
                        )
                    }
                    expectedStart +=
                        window.duration
                        - (window.cut == .maximumDuration ? policy.overlapDuration : 0)
                }
            }
            if let flushedWindow = await cutter.flush() {
                if abs(flushedWindow.startTime - expectedStart) >= 1e-9 {
                    Issue.record(
                        "seed 0xF00D_BEEF iteration \(iteration): flushed startTime \(flushedWindow.startTime) != expected \(expectedStart)"
                    )
                }
                if abs(expectedStart + flushedWindow.duration - totalIngested) >= 1e-9 {
                    Issue.record(
                        "seed 0xF00D_BEEF iteration \(iteration): tiling total mismatch: expectedStart=\(expectedStart) flushedDuration=\(flushedWindow.duration) totalIngested=\(totalIngested)"
                    )
                }
            } else {
                if abs(expectedStart - totalIngested) >= 1e-9 {
                    Issue.record(
                        "seed 0xF00D_BEEF iteration \(iteration): no flush window but expectedStart=\(expectedStart) != totalIngested=\(totalIngested)"
                    )
                }
            }
        }
    }

    @Test("C18: carried audio does not count toward the minimum window duration")
    func carriedAudioDoesNotCountTowardTheMinimum() async throws {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 10, minimumWindowDuration: 0.4, overlapDuration: 0.2)
        let cutter = makeCutter(
            policy: policy,
            verdicts: [true, true, true, true, false, true, true, false, true, true, false])

        // First boundary: 0.5s of speech, none of it carried, so it cuts.
        var window: AudioWindow?
        for _ in 0..<5 {
            window = await cutter.ingest(buffer())
        }
        let first = try #require(window)
        #expect(abs(first.duration - 0.5) < 1e-9)

        // Second boundary: 0.5s pending, but 0.2s of that was carried from the
        // first window. Only 0.3s is new, short of the 0.4s minimum, so the
        // boundary is ignored. Counting the carry here is what produced the
        // 1.0s-of-new-audio windows that came back mangled on real speech.
        for _ in 0..<3 {
            window = await cutter.ingest(buffer())
        }
        #expect(window == nil, "a window of mostly carried audio must not close on a boundary")

        // Third boundary: now 0.6s is new, so it cuts, and it starts where the
        // first window's carry began rather than where that window ended.
        for _ in 0..<3 {
            window = await cutter.ingest(buffer())
        }
        let second = try #require(window)
        #expect(abs(second.startTime - 0.3) < 1e-9)
        #expect(abs(second.duration - 0.8) < 1e-9)
    }

    @Test("C17: flush emits the trailing fragment the detector called silence")
    func flushEmitsTrailingFragmentTheDetectorCalledSilence() async throws {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 1.0, minimumWindowDuration: 0.2, overlapDuration: 0.2)
        // Three speech buffers, a falling edge that closes a silence window,
        // then two buffers the detector calls silence. On real speech those
        // last two hold the end of the final sentence: an energy detector
        // reading a quiet tail as silence is what cost this recording its last
        // two words before the flush learned to emit them.
        let cutter = makeCutter(policy: policy, verdicts: [true, true, true, false, false, false])

        for _ in 0..<4 {
            _ = await cutter.ingest(buffer())
        }
        for _ in 0..<2 {
            _ = await cutter.ingest(buffer())
        }

        let flushed = try #require(
            await cutter.flush(),
            "the last window must not be dropped on the detector's word alone")
        #expect(flushed.cut == .stop)
        // 0.2s carried from the silence cut, plus the 0.2s the detector
        // dismissed. The carry is what makes this safe: the window holds
        // confirmed speech, so it is never a decode of pure silence.
        #expect(abs(flushed.duration - 0.4) < 1e-9)
        #expect(abs(flushed.startTime - 0.2) < 1e-9)
    }

    @Test("C15: fuzz B, random verdicts, bounds only")
    func fuzzRandomVerdictsBoundsOnly() async {
        var generator = SeededGenerator(seed: 0xF00D_BEEF)
        for iteration in 0..<200 {
            let bufferCount = Int.random(in: 1...400, using: &generator)
            let maximumWindowDuration = [0.5, 1.0, 2.0, 5.0].randomElement(using: &generator)!
            let minimumWindowDuration = [0.1, 0.5, 1.0].randomElement(using: &generator)!
            let overlapDuration = [0.1, 0.25, 0.5].randomElement(using: &generator)!
            let policy = StreamingWindowPolicy(
                maximumWindowDuration: maximumWindowDuration,
                minimumWindowDuration: minimumWindowDuration,
                overlapDuration: overlapDuration
            )
            let verdicts = (0..<bufferCount).map { _ in Bool.random(using: &generator) }
            let cutter = makeCutter(policy: policy, verdicts: verdicts)

            var previousWindow: AudioWindow?
            for _ in 0..<bufferCount {
                guard let window = await cutter.ingest(buffer()) else { continue }

                if window.duration > policy.maximumWindowDuration + 0.1 + 1e-9 {
                    Issue.record(
                        "seed 0xF00D_BEEF iteration \(iteration): duration \(window.duration) overshoots max \(policy.maximumWindowDuration)"
                    )
                }
                if window.cut != .stop && window.duration < policy.minimumWindowDuration - 1e-9 {
                    Issue.record(
                        "seed 0xF00D_BEEF iteration \(iteration): duration \(window.duration) undershoots min \(policy.minimumWindowDuration)"
                    )
                }
                if let previousWindow, window.startTime <= previousWindow.startTime {
                    Issue.record(
                        "seed 0xF00D_BEEF iteration \(iteration): startTime not strictly increasing: \(previousWindow.startTime) -> \(window.startTime)"
                    )
                }
                // Every cut but the last carries audio forward, so a window
                // starts before the previous one ended by design. What must
                // hold is that the carry is bounded: never more than the
                // policy's overlap, and never so much that the window is
                // re-covered rather than advanced.
                if let previousWindow, previousWindow.cut != .stop {
                    let previousEnd = previousWindow.startTime + previousWindow.duration
                    // A gap is not checked for. `ingest` returns nil for a
                    // window that held no speech, having already consumed and
                    // discarded its audio, so two windows that arrive back to
                    // back here can have any amount of dropped silence
                    // between them.
                    let carried = max(previousEnd - window.startTime, 0)
                    if carried > policy.overlapDuration + 1e-9 {
                        Issue.record(
                            "seed 0xF00D_BEEF iteration \(iteration): carried \(carried) exceeds overlap \(policy.overlapDuration)"
                        )
                    }
                    if carried > previousWindow.duration / 2 + 1e-9 {
                        Issue.record(
                            "seed 0xF00D_BEEF iteration \(iteration): carried \(carried) is more than half the \(previousWindow.duration) window it came from"
                        )
                    }
                }
                previousWindow = window
            }
        }
    }

    @Test(
        "C16: a policy built from a negative maximumWindowDuration still forces a valid cut on the first speech buffer, not a crash"
    )
    func negativeMaximumDurationForcesAValidCutNotACrash() async throws {
        // StreamingWindowPolicy floors a non-positive maximumWindowDuration
        // rather than passing it through (see StreamingWindowPolicyTests.P8).
        // Confirming that here, through the cutter, is the regression that
        // matters: unfloored, `overlapDuration` clamps to a negative half of
        // a negative maximum, `carryCount` goes negative, and
        // `pending.suffix(carryCount)` traps, reachable from the very first
        // speech buffer, since the floored maximum (0.1s) equals one buffer.
        let policy = StreamingWindowPolicy(maximumWindowDuration: -1)
        let cutter = makeCutter(policy: policy, verdicts: [true])

        let window = try #require(await cutter.ingest(buffer()))
        #expect(window.cut == .maximumDuration)
        #expect(window.samples.count == bufferSize)
    }
}
