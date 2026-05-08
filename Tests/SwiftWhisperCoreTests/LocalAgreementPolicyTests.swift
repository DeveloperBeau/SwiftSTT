import Testing
@testable import SwiftWhisperCore

@Suite("LocalAgreement Policy")
struct LocalAgreementPolicyTests {

    @Test("Stub returns full hypothesis as text, empty confirmed")
    func stubBehaviour() async {
        let policy = LocalAgreementPolicy()
        let tokens = [
            WhisperToken(id: 1, text: "Hello"),
            WhisperToken(id: 2, text: " world"),
        ]
        let result = await policy.process(previous: [], current: tokens)
        #expect(result.confirmed == "")
        #expect(result.hypothesis == "Hello world")
    }

    @Test("Process with empty tokens")
    func emptyTokens() async {
        let policy = LocalAgreementPolicy()
        let result = await policy.process(previous: [], current: [])
        #expect(result.confirmed == "")
        #expect(result.hypothesis == "")
    }

    @Test("Reset clears state")
    func reset() async {
        let policy = LocalAgreementPolicy()
        _ = await policy.process(previous: [], current: [WhisperToken(id: 1, text: "x")])
        await policy.reset()
        let result = await policy.process(previous: [], current: [])
        #expect(result.hypothesis == "")
    }

    @Test("Output equality")
    func outputEquality() {
        let a = LocalAgreementPolicy.Output(confirmed: "hi", hypothesis: "there")
        let b = LocalAgreementPolicy.Output(confirmed: "hi", hypothesis: "there")
        #expect(a == b)
    }
}
