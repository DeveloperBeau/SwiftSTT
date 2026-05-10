import Foundation

/// Stabilises streaming transcription by waiting for the model to agree with itself.
///
/// Real-time decoding produces a fresh hypothesis every time new audio arrives.
/// Naively showing each hypothesis flickers the UI and confuses users when the
/// model rewrites words it had already produced. The local-agreement policy
/// fixes this with a simple rule: a token is confirmed when it appears at the
/// same position across N successive hypotheses (default N=2).
///
/// The stateful ``ingest(tokens:)`` method is the primary API for streaming
/// pipelines. It returns only the *newly* stable tokens since the last call,
/// making it safe to append them directly to a running transcript.
///
/// The stateless ``process(previous:current:)`` method exists for one-shot
/// comparisons where the caller manages its own hypothesis history.
///
/// ## References
///
/// Liu, D. and Niehues, J., 2020. *Low-Latency Sequence-to-Sequence Speech
/// Recognition and Translation by Partial Hypothesis Selection.* The local
/// agreement variant used here is described in section 3.2.
public actor LocalAgreementPolicy {

    /// The result of one policy step.
    public struct Output: Sendable, Equatable {

        /// Tokens that match the prior hypothesis and can be shown as final.
        public let confirmed: String

        /// New tokens that may still change in the next update.
        public let hypothesis: String

        /// Creates a new Output with the supplied values.
        public init(confirmed: String, hypothesis: String) {
            self.confirmed = confirmed
            self.hypothesis = hypothesis
        }
    }

    private let agreementCount: Int
    private var hypothesisRing: [[WhisperToken]] = []
    private var confirmedCount: Int = 0

    /// - Parameter agreementCount: how many consecutive hypotheses must agree
    ///   on a token position before it becomes stable. Must be at least 2.
    public init(agreementCount: Int = 2) {
        self.agreementCount = max(2, agreementCount)
    }

    /// Drops all remembered hypotheses and confirmed state.
    ///
    /// Use after the audio source restarts or the VAD detects a silence transition.
    public func reset() {
        hypothesisRing.removeAll(keepingCapacity: true)
        confirmedCount = 0
    }

    /// Feeds a new hypothesis into the ring and returns the tokens that became
    /// stable as a result.
    ///
    /// Returns an empty array until the ring contains ``agreementCount``
    /// hypotheses. After that, returns the delta between the previously
    /// confirmed prefix length and the new common prefix length.
    ///
    /// Token equality uses ``WhisperToken/id`` so that the same vocabulary
    /// entry from two runs always matches, even if probability or timestamp
    /// annotations differ.
    public func ingest(tokens: [WhisperToken]) -> [WhisperToken] {
        hypothesisRing.append(tokens)
        if hypothesisRing.count > agreementCount {
            hypothesisRing.removeFirst(hypothesisRing.count - agreementCount)
        }

        guard hypothesisRing.count >= agreementCount else { return [] }

        let prefixLength = longestCommonPrefixLength(hypothesisRing)

        guard prefixLength > confirmedCount else { return [] }

        let newlyStable = Array(tokens[confirmedCount..<prefixLength])
        confirmedCount = prefixLength
        return newlyStable
    }

    /// Stateless comparison of two hypotheses.
    ///
    /// Returns the longest common prefix as confirmed text and the remainder of `current` as the
    /// hypothesis.
    ///
    /// This method does not modify actor state. For streaming use, prefer
    /// ``ingest(tokens:)`` which tracks the cumulative confirmed prefix
    /// across calls.
    public func process(previous: [WhisperToken], current: [WhisperToken]) -> Output {
        let prefixLen = Self.commonPrefixLength(previous, current)
        let confirmed = current.prefix(prefixLen).map(\.text).joined()
        let hypothesis = current.dropFirst(prefixLen).map(\.text).joined()
        return Output(confirmed: confirmed, hypothesis: hypothesis)
    }

    // MARK: - Internals

    private func longestCommonPrefixLength(_ sequences: [[WhisperToken]]) -> Int {
        guard let first = sequences.first else { return 0 }
        let minLen = sequences.map(\.count).min() ?? 0
        for i in 0..<minLen {
            let id = first[i].id
            for seq in sequences.dropFirst() {
                if seq[i].id != id { return i }
            }
        }
        return minLen
    }

    private static func commonPrefixLength(
        _ a: [WhisperToken],
        _ b: [WhisperToken]
    ) -> Int {
        let minLen = min(a.count, b.count)
        for i in 0..<minLen where a[i].id != b[i].id {
            return i
        }
        return minLen
    }
}
