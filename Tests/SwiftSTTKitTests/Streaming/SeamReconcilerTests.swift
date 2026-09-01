import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTKit

private func window(
    startingAt startTime: TimeInterval,
    lasting duration: TimeInterval,
    cut: WindowCutCause
) -> AudioWindow {
    AudioWindow(
        samples: Array(repeating: 0, count: Int(duration * 16_000)),
        startTime: startTime,
        cut: cut
    )
}

private func segment(_ text: String, _ start: TimeInterval, _ end: TimeInterval) -> TranscriptionSegment {
    TranscriptionSegment(text: text, start: start, end: end)
}

private struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed &* 6_364_136_223_846_793_005 &+ 1 }
    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }
}

@Suite("SeamReconciler")
struct SeamReconcilerTests {

    @Test("R1: silence cut passes everything through with the offset applied")
    func silenceCutPassesEverythingThrough() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        let win = window(startingAt: 12.0, lasting: 5.0, cut: .silence)
        let decoded = [segment("hello", 0.5, 1.5), segment("world", 1.5, 3.0)]

        let emitted = reconciler.reconcile(decoded, from: win).segments

        #expect(emitted == [segment("hello", 12.5, 13.5), segment("world", 13.5, 15.0)])
    }

    @Test("R2: a forced cut emits its own tail, because only the next window can know it repeats")
    func forcedCutEmitsItsOwnTail() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        let win = window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        let decoded = [
            segment("alpha", 4.0, 5.0),
            segment("bravo", 9.0, 10.0),
            segment("charlie", 12.0, 13.0),
            segment("delta", 14.6, 14.8),
        ]

        // The carried audio is re-decoded by the window after this one, and
        // the duplicate is removed there. Holding "delta" back here instead
        // would lose it outright if capture stopped before another window.
        let emitted = reconciler.reconcile(decoded, from: win).segments

        #expect(emitted == decoded)
    }

    @Test("R2b: the longest matching run is dropped, not the first one found")
    func longestMatchingRunIsDropped() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        _ = reconciler.reconcile(
            [segment("the window carries the tail", 12.0, 15.0)],
            from: window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        )

        // "the" alone matches the emitted tail's last word, and so does the
        // three-word run "carries the tail". Taking the short match would
        // re-emit "carries the tail" a second time, so the longest run wins.
        let emitted = reconciler.reconcile(
            [segment("carries the tail onward now", 0.0, 2.0)],
            from: window(startingAt: 13.5, lasting: 5, cut: .silence)
        ).segments

        #expect(emitted.count == 1)
        #expect(emitted.first?.text == "onward now")
    }

    @Test("R3: no candidate at or after preferredSeam, then nothing lost across the gap")
    func noCandidateAtOrAfterPreferredSeam() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        let win = window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        let decoded = [segment("a", 1.0, 2.0), segment("b", 3.0, 4.0)]

        let emitted = reconciler.reconcile(decoded, from: win).segments
        #expect(emitted == decoded)

        let nextWindow = window(startingAt: 13.5, lasting: 5, cut: .stop)
        let nextDecoded = [segment("next", 0.0, 1.0)]
        let nextEmitted = reconciler.reconcile(nextDecoded, from: nextWindow).segments
        #expect(nextEmitted == [segment("next", 13.5, 14.5)])
    }

    @Test("R4: the next window drops the words the previous one already emitted")
    func nextWindowDropsWhatWasAlreadyEmitted() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        let win = window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        _ = reconciler.reconcile(
            [
                segment("alpha", 4.0, 5.0),
                segment("bravo", 9.0, 10.0),
                segment("charlie", 12.0, 13.0),
                segment("delta echo", 13.8, 14.8),
            ],
            from: win
        )

        // The carried 1.5 seconds hold "delta echo", so the next window
        // decodes them again before reaching new speech. Punctuation and case
        // differ between the two readings, as they do in practice, and must
        // not stop the match.
        let nextWindow = window(startingAt: 13.5, lasting: 5, cut: .silence)
        let nextDecoded = [segment("Delta echo,", 0.3, 1.3), segment("fresh", 1.3, 2.5)]
        let emitted = reconciler.reconcile(nextDecoded, from: nextWindow).segments

        #expect(emitted == [segment("fresh", 14.8, 16.0)])
    }

    @Test("R5: a segment the overlap ends partway through keeps its remaining words")
    func straddlerKeepsItsRemainingWords() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        let win = window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        _ = reconciler.reconcile(
            [segment("alpha bravo charlie", 12.0, 14.8)],
            from: win
        )

        // The model does not have to break the second reading where the first
        // one broke: here the repeated "charlie" and the new speech land in a
        // single segment. Emitting it whole would duplicate "charlie", and
        // dropping it whole would lose the rest, so it is trimmed.
        let nextWindow = window(startingAt: 13.5, lasting: 5, cut: .silence)
        let emitted = reconciler.reconcile(
            [segment("charlie delta echo", 0.5, 2.5)], from: nextWindow).segments

        #expect(emitted.count == 1)
        #expect(emitted.first?.text == "delta echo")
        #expect(emitted.first?.end == 16.0)
        // The start advances past the dropped word rather than staying at 14.0.
        #expect((emitted.first?.start ?? 0) > 14.0)
    }

    @Test("R6: a window starting exactly where the previous ended is not searched for overlap")
    func adjacentWindowIsNotSearchedForOverlap() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        let win = window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        _ = reconciler.reconcile(
            [
                segment("a", 4.0, 5.0),
                segment("b", 9.0, 10.0),
                segment("c", 12.0, 13.0),
                segment("d", 14.6, 14.8),
            ],
            from: win
        )

        _ = reconciler.reconcile([], from: window(startingAt: 13.5, lasting: 5, cut: .silence))

        // 13.5 + 5 == 18.5, so this window re-covers nothing and every word
        // it decodes is new by construction.
        let finalWindow = window(startingAt: 18.5, lasting: 5, cut: .stop)
        let emitted = reconciler.reconcile([segment("early", 0.0, 0.2)], from: finalWindow).segments

        #expect(emitted == [segment("early", 18.5, 18.7)])
    }

    @Test("R7: degenerate — one segment spanning the whole window")
    func degenerateSingleSegmentSpansWindow() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        let win = window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        let emitted = reconciler.reconcile(
            [segment("one long sentence", 0.0, 15.0)], from: win).segments
        #expect(emitted == [segment("one long sentence", 0.0, 15.0)])

        let nextWindow = window(startingAt: 13.5, lasting: 5, cut: .stop)
        let nextDecoded = [segment("sentence", 0.0, 1.5), segment("after", 1.5, 3.0)]
        let nextEmitted = reconciler.reconcile(nextDecoded, from: nextWindow).segments
        #expect(nextEmitted == [segment("after", 15.0, 16.5)])
    }

    @Test("R8: failure case — empty decode does not crash on candidates.min()")
    func emptyDecodeDoesNotCrash() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        let forcedWindow = window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        let emitted = reconciler.reconcile([], from: forcedWindow).segments
        #expect(emitted == [])

        let nextWindow = window(startingAt: 13.5, lasting: 5, cut: .stop)
        let nextEmitted = reconciler.reconcile([segment("after", 0.0, 1.0)], from: nextWindow).segments
        #expect(nextEmitted == [segment("after", 13.5, 14.5)])
    }

    @Test("R9: failure case — zero-duration window")
    func zeroDurationWindow() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        let zeroWindow = AudioWindow(samples: [], startTime: 4.0, cut: .maximumDuration)
        let emitted = reconciler.reconcile([], from: zeroWindow).segments
        #expect(emitted == [])

        let followingWindow = window(startingAt: 4.0, lasting: 2, cut: .stop)
        let followingEmitted = reconciler.reconcile(
            [segment("after", 0.0, 0.5)], from: followingWindow).segments
        #expect(followingEmitted == [segment("after", 4.0, 4.5)])
    }

    @Test("R13: the unconfirmed tail of a window is retracted when the next one reads it better")
    func unconfirmedTailIsRetracted() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        var transcript: [TranscriptionSegment] = []
        reconciler.reconcile(
            [segment("when speech runs low", 12.0, 15.0)],
            from: window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        ).apply(to: &transcript)
        #expect(transcript.map(\.text) == ["when speech runs low"])

        // Real disagreement, taken from what whisper.cpp returned for this
        // audio: the first reading ends the truncated window on "low", the
        // second hears the whole word as "long". Three words agree, so the
        // fourth is the only thing nothing corroborates, and the reading that
        // heard the rest of the sentence gets to replace it.
        let update = reconciler.reconcile(
            [segment("when speech runs long without pausing", 0.0, 3.0)],
            from: window(startingAt: 13.5, lasting: 5, cut: .silence)
        )

        #expect(update.retractingFrom == 12.0)
        update.apply(to: &transcript)
        #expect(transcript.map(\.text) == ["when speech runs", "long without pausing"])
    }

    @Test("R13b: a tail the next window agrees with is left alone")
    func confirmedTailIsNotRetracted() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        var transcript: [TranscriptionSegment] = []
        reconciler.reconcile(
            [segment("carries overlap forward the seam", 12.0, 15.0)],
            from: window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        ).apply(to: &transcript)

        // Also real: the second window opens on "the lap forward", having
        // mangled the words it met with no audio in front of them, where the
        // first window had them right. Neither reading is better as a rule.
        // Every word of the first window's tail is agreed with here, so there
        // is nothing unconfirmed and nothing is taken back.
        let update = reconciler.reconcile(
            [segment("the lap forward the seam reconciler then drops", 0.0, 3.0)],
            from: window(startingAt: 13.5, lasting: 5, cut: .silence)
        )

        #expect(update.retractingFrom == nil)
        update.apply(to: &transcript)
        #expect(
            transcript.map(\.text) == ["carries overlap forward the seam", "reconciler then drops"])
    }

    @Test("R15: nothing is retracted when the next window supplies no replacement")
    func nothingRetractedWithoutAReplacement() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        var transcript: [TranscriptionSegment] = []
        reconciler.reconcile(
            [segment("alpha bravo charlie delta", 12.0, 15.0)],
            from: window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        ).apply(to: &transcript)

        // The second window re-reads the overlap and stops short, returning
        // only its first two words. That happens: the model can drop the start
        // of a window it was handed mid-phrase. Retracting the tail here would
        // take back "charlie delta" and put nothing in their place, which is
        // the one outcome worse than leaving a doubtful word standing.
        let update = reconciler.reconcile(
            [segment("alpha bravo", 0.0, 1.0)],
            from: window(startingAt: 13.5, lasting: 5, cut: .silence)
        )

        #expect(update.retractingFrom == nil)
        update.apply(to: &transcript)
        #expect(transcript.map(\.text) == ["alpha bravo charlie delta"])
    }

    @Test("R14: a single common word away from the seam is not treated as overlap")
    func loneCommonWordAwayFromSeamIsNotOverlap() {
        var reconciler = SeamReconciler(overlapDuration: 1.5)
        _ = reconciler.reconcile(
            [segment("alpha bravo charlie", 12.0, 15.0)],
            from: window(startingAt: 0, lasting: 15, cut: .maximumDuration)
        )

        // "charlie" appears in both, but three words into the new window
        // rather than at its start. Treating that as the seam would drop
        // "delta echo", which was never emitted. One word only counts when it
        // is the seam itself, which R7 covers.
        let decoded = [segment("delta echo charlie foxtrot", 0.0, 3.0)]
        let emitted = reconciler.reconcile(
            decoded, from: window(startingAt: 13.5, lasting: 5, cut: .silence)
        ).segments

        #expect(emitted.count == 1)
        #expect(emitted.first?.text == "delta echo charlie foxtrot")
    }

    // MARK: - R11 / R12 fuzz harness

    private struct GroundTruthSegment {
        let text: String
        let start: TimeInterval
        let end: TimeInterval
    }

    private struct GeneratedWindow {
        let audioStart: TimeInterval
        let audioEnd: TimeInterval
        let cut: WindowCutCause
        let decoded: [TranscriptionSegment]
    }

    private struct FuzzCase {
        let seed: UInt64
        let policyMaximum: TimeInterval
        let policyOverlap: TimeInterval
        let truth: [GroundTruthSegment]
        let windows: [GeneratedWindow]
    }

    private static func generateCase(seed: UInt64) -> FuzzCase {
        var generator = SeededGenerator(seed: seed)

        let segmentCount = Int.random(in: 4...40, using: &generator)
        var truth: [GroundTruthSegment] = []
        var cursor: TimeInterval = 0
        for index in 0..<segmentCount {
            let rawDuration = TimeInterval.random(in: 0.2...4.0, using: &generator)
            let duration = (rawDuration / 0.05).rounded() * 0.05
            truth.append(GroundTruthSegment(text: "w\(index)", start: cursor, end: cursor + duration))
            cursor += duration
        }
        let total = cursor

        let maximumWindowDuration = [6.0, 10.0, 15.0].randomElement(using: &generator)!
        let overlapDuration = [0.5, 1.5, 3.0].randomElement(using: &generator)!

        var windows: [GeneratedWindow] = []
        var windowCursor: TimeInterval = 0
        var safetyCounter = 0
        while true {
            safetyCounter += 1
            let forced = Bool.random(using: &generator)
            // A real AudioWindowCutter only forces a cut once accumulated
            // audio reaches maximumWindowDuration, so a forced window is
            // never shorter than that; drawing it shorter (as an early draft
            // of this generator did) lets consecutive forced cuts cascade
            // into non-adjacent overlaps the cutter's overlapDuration <=
            // maximumWindowDuration/2 clamp guarantees can never happen.
            // Silence/stop windows have no such floor beyond the minimum.
            let windowLength =
                forced
                ? maximumWindowDuration
                : TimeInterval.random(in: 3.0...maximumWindowDuration, using: &generator)
            let audioStart = windowCursor
            let audioEnd = windowCursor + windowLength

            let contained = truth.filter { $0.start >= audioStart && $0.end <= audioEnd }
            let decoded = contained.map {
                TranscriptionSegment(text: $0.text, start: $0.start - audioStart, end: $0.end - audioStart)
            }

            // Every cut but the last carries audio forward, silence cuts
            // included, and the cutter caps the carry at half the closed
            // window so it always makes progress. A generator that overlapped
            // only forced cuts would leave the silence seam, the one real
            // audio actually produces, untested.
            let carried = min(overlapDuration, windowLength / 2)
            let tentativeNextCursor = audioEnd - carried
            let isLast = tentativeNextCursor >= total || safetyCounter > 1_000
            let cut: WindowCutCause = isLast ? .stop : (forced ? .maximumDuration : .silence)

            windows.append(
                GeneratedWindow(audioStart: audioStart, audioEnd: audioEnd, cut: cut, decoded: decoded))

            if isLast {
                break
            }
            windowCursor = tentativeNextCursor
        }

        return FuzzCase(
            seed: seed,
            policyMaximum: maximumWindowDuration,
            policyOverlap: overlapDuration,
            truth: truth,
            windows: windows
        )
    }

    @Test("R11: fuzz — exact-once tiling across 200 generated cases")
    func fuzzExactOnceTiling() {
        for iteration in 0..<200 {
            let seed = 0xF00D_BEEF &+ UInt64(iteration)
            let testCase = Self.generateCase(seed: seed)

            var reconciler = SeamReconciler(overlapDuration: testCase.policyOverlap)
            var emitted: [TranscriptionSegment] = []
            for generated in testCase.windows {
                let win = AudioWindow(
                    samples: Array(
                        repeating: Float(0), count: Int((generated.audioEnd - generated.audioStart) * 16_000)),
                    startTime: generated.audioStart,
                    cut: generated.cut
                )
                reconciler.reconcile(generated.decoded, from: win).apply(to: &emitted)
            }

            let emittedTexts = Set(emitted.map(\.text))
            let containedTruthTexts = Set(
                testCase.windows.flatMap { $0.decoded }.map(\.text))

            // No loss: every true segment contained in at least one window
            // must appear at least once in the emitted output.
            let lostTexts = containedTruthTexts.subtracting(emittedTexts)
            if !lostTexts.isEmpty {
                Issue.record(
                    "seed \(testCase.seed) lost segments \(lostTexts); policy max=\(testCase.policyMaximum) overlap=\(testCase.policyOverlap); truth=\(testCase.truth.map { ($0.text, $0.start, $0.end) })"
                )
            }

            // No duplication: no emitted text appears twice.
            var seenTexts = Set<String>()
            var duplicates = Set<String>()
            for item in emitted {
                if !seenTexts.insert(item.text).inserted {
                    duplicates.insert(item.text)
                }
            }
            if !duplicates.isEmpty {
                Issue.record(
                    "seed \(testCase.seed) duplicated segments \(duplicates); policy max=\(testCase.policyMaximum) overlap=\(testCase.policyOverlap)"
                )
            }

            // Monotone: emitted start values are non-decreasing.
            for index in 1..<max(emitted.count, 1) where index < emitted.count {
                if emitted[index].start < emitted[index - 1].start {
                    Issue.record(
                        "seed \(testCase.seed) start values not monotone at index \(index): \(emitted.map(\.start))"
                    )
                    break
                }
            }

            // Absolute time is preserved: each emitted segment's (start, end)
            // equals the ground-truth segment's (start, end).
            let truthByText = Dictionary(uniqueKeysWithValues: testCase.truth.map { ($0.text, $0) })
            for item in emitted {
                guard let truthSegment = truthByText[item.text] else { continue }
                if abs(item.start - truthSegment.start) > 1e-6 || abs(item.end - truthSegment.end) > 1e-6 {
                    Issue.record(
                        "seed \(testCase.seed) time mismatch for \(item.text): got (\(item.start), \(item.end)), expected (\(truthSegment.start), \(truthSegment.end))"
                    )
                }
            }
        }
    }

    @Test("R12: false-positive validation — a pass-through stub must fail R11's no-duplication invariant")
    func passThroughFailsNoDuplicationInvariant() {
        var anyDuplicateFound = false
        for iteration in 0..<200 {
            let seed = 0xF00D_BEEF &+ UInt64(iteration)
            let testCase = Self.generateCase(seed: seed)

            // Pass-through: emit every window's segments in absolute time
            // with no filtering at all.
            var emitted: [TranscriptionSegment] = []
            for generated in testCase.windows {
                for segment in generated.decoded {
                    emitted.append(
                        TranscriptionSegment(
                            text: segment.text,
                            start: segment.start + generated.audioStart,
                            end: segment.end + generated.audioStart
                        ))
                }
            }

            var seenTexts = Set<String>()
            for item in emitted {
                if !seenTexts.insert(item.text).inserted {
                    anyDuplicateFound = true
                    break
                }
            }
            if anyDuplicateFound { break }
        }

        #expect(anyDuplicateFound, "pass-through stub should duplicate at least one segment across 200 cases")
    }
}
