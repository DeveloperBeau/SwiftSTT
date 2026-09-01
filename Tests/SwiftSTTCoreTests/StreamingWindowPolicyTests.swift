import Foundation
import Testing

@testable import SwiftSTTCore

private struct SeededGenerator: RandomNumberGenerator {
    private var state: UInt64
    init(seed: UInt64) { self.state = seed &* 6_364_136_223_846_793_005 &+ 1 }
    mutating func next() -> UInt64 {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return state
    }
}

@Suite("StreamingWindowPolicy")
struct StreamingWindowPolicyTests {

    @Test("P1: default values, hand-written")
    func defaultsAreHandWritten() {
        let policy = StreamingWindowPolicy.default
        #expect(policy.maximumWindowDuration == 15)
        #expect(policy.minimumWindowDuration == 2)
        #expect(policy.overlapDuration == 1.5)
        #expect(policy.sampleRate == 16_000)
    }

    @Test("P2: minimum clamps to maximum")
    func minimumClampsToMaximum() {
        let policy = StreamingWindowPolicy(maximumWindowDuration: 3, minimumWindowDuration: 10)
        #expect(policy.minimumWindowDuration == 3)
    }

    @Test("P3: overlap clamps to half of maximum")
    func overlapClampsToHalfMaximum() {
        let policy = StreamingWindowPolicy(maximumWindowDuration: 4, overlapDuration: 9)
        #expect(policy.overlapDuration == 2)
    }

    @Test("P4: false-positive validation — clamps do not fire on valid input")
    func clampsLeaveValidInputAlone() {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 20,
            minimumWindowDuration: 1,
            overlapDuration: 2
        )
        #expect(policy.maximumWindowDuration == 20)
        #expect(policy.minimumWindowDuration == 1)
        #expect(policy.overlapDuration == 2)
    }

    @Test("P5: false-negative validation — just past each limit still clamps")
    func clampsFireJustPastLimit() {
        let policy = StreamingWindowPolicy(
            maximumWindowDuration: 4,
            minimumWindowDuration: 4.0001,
            overlapDuration: 2.0001
        )
        #expect(policy.minimumWindowDuration == 4)
        #expect(policy.overlapDuration == 2)
    }

    @Test("P6: fuzz — clamp invariants hold for 500 generated policies")
    func fuzzClampInvariants() {
        var generator = SeededGenerator(seed: 0xF00D_BEEF)
        for iteration in 0..<500 {
            // Range includes non-positive values on purpose: the clamp this
            // invariant checks (`maximumWindowDuration > 0`) must be enforced
            // by `init`, not merely true because every drawn value already
            // satisfies it.
            let maximum = TimeInterval.random(in: -5...120, using: &generator)
            let minimum = TimeInterval.random(in: 0.01...120, using: &generator)
            let overlap = TimeInterval.random(in: 0.01...120, using: &generator)
            let sampleRate = [8_000, 16_000, 44_100].randomElement(using: &generator)!

            let policy = StreamingWindowPolicy(
                maximumWindowDuration: maximum,
                minimumWindowDuration: minimum,
                overlapDuration: overlap,
                sampleRate: sampleRate
            )

            guard policy.minimumWindowDuration <= policy.maximumWindowDuration,
                policy.overlapDuration <= policy.maximumWindowDuration / 2,
                policy.maximumWindowDuration > 0,
                policy.minimumWindowDuration > 0,
                policy.overlapDuration > 0
            else {
                Issue.record(
                    "seed 0xF00D_BEEF iteration \(iteration) failed: max=\(maximum) min=\(minimum) overlap=\(overlap) -> \(policy)"
                )
                continue
            }
        }
    }

    @Test("P8: boundary — a non-positive maximumWindowDuration is floored, not passed through")
    func nonPositiveMaximumIsFloored() {
        let zero = StreamingWindowPolicy(maximumWindowDuration: 0)
        #expect(zero.maximumWindowDuration == 0.1)
        #expect(zero.minimumWindowDuration == 0.1)
        #expect(zero.overlapDuration == 0.05)

        let negative = StreamingWindowPolicy(maximumWindowDuration: -1)
        #expect(negative.maximumWindowDuration == 0.1)
        #expect(negative.minimumWindowDuration == 0.1)
        #expect(negative.overlapDuration == 0.05)
    }

    @Test("P7: Equatable")
    func equatable() {
        let a = StreamingWindowPolicy(
            maximumWindowDuration: 10, minimumWindowDuration: 2, overlapDuration: 1, sampleRate: 16_000)
        let b = StreamingWindowPolicy(
            maximumWindowDuration: 10, minimumWindowDuration: 2, overlapDuration: 1, sampleRate: 16_000)
        #expect(a == b)
        let c = StreamingWindowPolicy(
            maximumWindowDuration: 10, minimumWindowDuration: 2, overlapDuration: 1.2, sampleRate: 16_000)
        #expect(a != c)
    }
}
