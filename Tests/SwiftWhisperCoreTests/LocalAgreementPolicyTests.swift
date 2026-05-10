import Testing

@testable import SwiftWhisperCore

// MARK: - Helpers

private func token(_ id: Int, text: String = "") -> WhisperToken {
    WhisperToken(id: id, text: text)
}

// MARK: - Tests

@Suite("LocalAgreementPolicy")
struct LocalAgreementPolicyTests {

    // MARK: - ingest (stateful streaming API)

    @Test("ingest with empty tokens returns nothing")
    func ingestEmpty() async {
        let policy = LocalAgreementPolicy()
        let result = await policy.ingest(tokens: [])
        #expect(result.isEmpty)
    }

    @Test("Single ingest returns nothing because N=2 needs two confirmations")
    func ingestSingleReturnsNothing() async {
        let policy = LocalAgreementPolicy()
        let result = await policy.ingest(tokens: [token(1, text: "Hello")])
        #expect(result.isEmpty)
    }

    @Test("Two identical sequences emit the full common prefix")
    func ingestTwoIdentical() async {
        let policy = LocalAgreementPolicy()
        let tokens = [token(1, text: "Hello"), token(2, text: " world")]
        _ = await policy.ingest(tokens: tokens)
        let stable = await policy.ingest(tokens: tokens)
        #expect(stable.map(\.id) == [1, 2])
        #expect(stable.map(\.text) == ["Hello", " world"])
    }

    @Test("Divergent suffix is held back")
    func ingestDivergentSuffix() async {
        let policy = LocalAgreementPolicy()
        let first = [token(1, text: "A"), token(2, text: "B"), token(3, text: "C")]
        let second = [token(1, text: "A"), token(2, text: "B"), token(4, text: "D")]
        _ = await policy.ingest(tokens: first)
        let stable = await policy.ingest(tokens: second)
        #expect(stable.map(\.id) == [1, 2])
    }

    @Test("Reset clears history so next ingest behaves like first")
    func resetClearsHistory() async {
        let policy = LocalAgreementPolicy()
        let tokens = [token(1, text: "X")]
        _ = await policy.ingest(tokens: tokens)
        _ = await policy.ingest(tokens: tokens)
        await policy.reset()
        let after = await policy.ingest(tokens: tokens)
        #expect(after.isEmpty)
    }

    @Test("N=3 requires three confirmations before any stable text")
    func ingestN3() async {
        let policy = LocalAgreementPolicy(agreementCount: 3)
        let tokens = [token(10, text: "ok")]
        _ = await policy.ingest(tokens: tokens)
        let afterTwo = await policy.ingest(tokens: tokens)
        #expect(afterTwo.isEmpty)
        let afterThree = await policy.ingest(tokens: tokens)
        #expect(afterThree.map(\.id) == [10])
    }

    @Test("Stable prefix only emits newly-stable tokens, no duplication")
    func ingestNoDuplication() async {
        let policy = LocalAgreementPolicy()
        let first = [token(1, text: "A"), token(2, text: "B")]
        _ = await policy.ingest(tokens: first)
        let batch1 = await policy.ingest(tokens: first)
        #expect(batch1.map(\.id) == [1, 2])

        let extended = [token(1, text: "A"), token(2, text: "B"), token(3, text: "C")]
        _ = await policy.ingest(tokens: extended)
        let batch2 = await policy.ingest(tokens: extended)
        #expect(batch2.map(\.id) == [3])
    }

    @Test("Token equality uses .id, not text or probability")
    func ingestMatchesById() async {
        let policy = LocalAgreementPolicy()
        let first = [WhisperToken(id: 5, text: "cat", probability: 0.9)]
        let second = [WhisperToken(id: 5, text: "dog", probability: 0.1)]
        _ = await policy.ingest(tokens: first)
        let stable = await policy.ingest(tokens: second)
        #expect(stable.count == 1)
        #expect(stable[0].id == 5)
    }

    @Test("Completely divergent second hypothesis emits nothing")
    func ingestFullyDivergent() async {
        let policy = LocalAgreementPolicy()
        _ = await policy.ingest(tokens: [token(1), token(2)])
        let stable = await policy.ingest(tokens: [token(3), token(4)])
        #expect(stable.isEmpty)
    }

    @Test("N=3 with divergence at third call does not confirm")
    func n3DivergentThirdCall() async {
        let policy = LocalAgreementPolicy(agreementCount: 3)
        let a = [token(1), token(2)]
        let b = [token(1), token(2)]
        let c = [token(1), token(3)]
        _ = await policy.ingest(tokens: a)
        _ = await policy.ingest(tokens: b)
        let stable = await policy.ingest(tokens: c)
        #expect(stable.map(\.id) == [1])
    }

    @Test("agreementCount below 2 is clamped to 2")
    func clampedAgreementCount() async {
        let policy = LocalAgreementPolicy(agreementCount: 0)
        let tokens = [token(1, text: "X")]
        _ = await policy.ingest(tokens: tokens)
        let stable = await policy.ingest(tokens: tokens)
        #expect(stable.count == 1)
    }

    // MARK: - process (stateless legacy API)

    @Test("process with empty inputs returns empty output")
    func processEmpty() async {
        let policy = LocalAgreementPolicy()
        let result = await policy.process(previous: [], current: [])
        #expect(result.confirmed == "")
        #expect(result.hypothesis == "")
    }

    @Test("process with empty previous returns full hypothesis")
    func processEmptyPrevious() async {
        let policy = LocalAgreementPolicy()
        let current = [token(1, text: "Hello"), token(2, text: " world")]
        let result = await policy.process(previous: [], current: current)
        #expect(result.confirmed == "")
        #expect(result.hypothesis == "Hello world")
    }

    @Test("process with matching prefix splits correctly")
    func processMatchingPrefix() async {
        let policy = LocalAgreementPolicy()
        let prev = [token(1, text: "A"), token(2, text: "B"), token(3, text: "C")]
        let curr = [token(1, text: "A"), token(2, text: "B"), token(4, text: "D")]
        let result = await policy.process(previous: prev, current: curr)
        #expect(result.confirmed == "AB")
        #expect(result.hypothesis == "D")
    }

    @Test("process with identical sequences confirms all")
    func processIdentical() async {
        let policy = LocalAgreementPolicy()
        let tokens = [token(1, text: "X"), token(2, text: "Y")]
        let result = await policy.process(previous: tokens, current: tokens)
        #expect(result.confirmed == "XY")
        #expect(result.hypothesis == "")
    }

    @Test("process does not modify actor state")
    func processIsStateless() async {
        let policy = LocalAgreementPolicy()
        let tokens = [token(1, text: "A")]
        _ = await policy.process(previous: tokens, current: tokens)
        let ingestResult = await policy.ingest(tokens: tokens)
        #expect(ingestResult.isEmpty)
    }

    // MARK: - Output equality

    @Test("Output equality")
    func outputEquality() {
        let a = LocalAgreementPolicy.Output(confirmed: "hi", hypothesis: "there")
        let b = LocalAgreementPolicy.Output(confirmed: "hi", hypothesis: "there")
        let c = LocalAgreementPolicy.Output(confirmed: "hi", hypothesis: "other")
        #expect(a == b)
        #expect(a != c)
    }
}
