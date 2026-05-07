import Foundation

public actor LocalAgreementPolicy {
    public struct Output: Sendable, Equatable {
        public let confirmed: String
        public let hypothesis: String

        public init(confirmed: String, hypothesis: String) {
            self.confirmed = confirmed
            self.hypothesis = hypothesis
        }
    }

    private var previousHypothesis: [WhisperToken] = []

    public init() {}

    public func reset() {
        previousHypothesis = []
    }

    public func process(previous: [WhisperToken], current: [WhisperToken]) -> Output {
        // Real impl in Phase 8. Stub returns empty confirmed + full current as hypothesis.
        let hypothesisText = current.map(\.text).joined()
        return Output(confirmed: "", hypothesis: hypothesisText)
    }
}
