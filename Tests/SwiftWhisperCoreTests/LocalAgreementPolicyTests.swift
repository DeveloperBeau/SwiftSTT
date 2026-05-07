import Testing
@testable import SwiftWhisperCore

@Suite("LocalAgreement Policy")
struct LocalAgreementPolicyTests {

    @Test("Stub returns empty confirmed", .disabled("real impl in Phase 8"))
    func stubReturnsEmptyConfirmed() async {
        let policy = LocalAgreementPolicy()
        let result = await policy.process(
            previous: [WhisperToken(id: 1, text: "Hello")],
            current: [WhisperToken(id: 1, text: "Hello")]
        )
        #expect(result.confirmed == "Hello")
    }
}
