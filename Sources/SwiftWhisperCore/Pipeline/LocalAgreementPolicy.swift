import Foundation

/// Stabilises streaming transcription by waiting for the model to agree with itself.
///
/// Real-time decoding produces a fresh hypothesis every time new audio arrives.
/// Naively showing each hypothesis flickers the UI and confuses users when the
/// model rewrites words it had already produced. The local-agreement policy
/// fixes this with a simple rule: a token is confirmed when it appears at the
/// same position in two successive hypotheses.
///
/// Concretely, on each update the policy:
///
/// 1. Compares the new hypothesis to the previous one.
/// 2. Finds the longest common prefix.
/// 3. Promotes that prefix to confirmed text.
/// 4. Emits the rest of the new hypothesis as tentative.
/// 5. Stores the new hypothesis for the next round.
///
/// Confirmed text never changes, which is exactly what UI code needs.
///
/// ## References
///
/// Liu, D. and Niehues, J., 2020. *Low-Latency Sequence-to-Sequence Speech
/// Recognition and Translation by Partial Hypothesis Selection.* The local
/// agreement variant used here is described in section 3.2.
///
/// > Important: The current implementation is a stub. Real prefix-matching
/// > arrives in milestone M6 (streaming pipeline).
public actor LocalAgreementPolicy {

    /// The result of one policy step.
    public struct Output: Sendable, Equatable {

        /// Tokens that match the prior hypothesis and can be shown as final.
        public let confirmed: String

        /// New tokens that may still change in the next update.
        public let hypothesis: String

        public init(confirmed: String, hypothesis: String) {
            self.confirmed = confirmed
            self.hypothesis = hypothesis
        }
    }

    private var previousHypothesis: [WhisperToken] = []

    public init() {}

    /// Drops any remembered prior hypothesis. Use after the audio source restarts.
    public func reset() {
        previousHypothesis = []
    }

    /// Compares the new hypothesis to the prior one and splits into confirmed plus
    /// tentative parts.
    ///
    /// > Note: Stub returns an empty confirmed string. The real algorithm lands in M6.
    public func process(previous: [WhisperToken], current: [WhisperToken]) -> Output {
        let hypothesisText = current.map(\.text).joined()
        return Output(confirmed: "", hypothesis: hypothesisText)
    }
}
