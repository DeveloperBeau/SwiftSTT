/// A transcription of the fixture scored against the bundled hand-written transcript.
///
/// The accuracy test and its default-gate checks both go through this type, so the
/// reference and the pass or fail decision have one owner.
struct AccuracyFixtureScore {

    /// The tiny model's word error rate on `jfk.wav` must stay below this.
    ///
    /// The reference has 22 words, so 0.10 allows at most two word errors.
    static let wordErrorRateThreshold = 0.10

    /// The reference the hypothesis was scored against.
    let reference: String

    /// Word error rate of the hypothesis against ``reference``.
    let wordErrorRate: WordErrorRate

    /// Whether ``wordErrorRate`` is below the threshold.
    var meetsThreshold: Bool {
        Self.meetsThreshold(wordErrorRate)
    }

    /// Scores `hypothesis` against the bundled `jfk.txt`.
    init(hypothesis: String) throws {
        reference = try AccuracyFixture.expectedTranscript
        wordErrorRate = WordErrorRate(reference: reference, hypothesis: hypothesis)
    }

    /// The pass or fail decision: a rate passes only when it is below the threshold.
    static func meetsThreshold(_ wordErrorRate: WordErrorRate) -> Bool {
        wordErrorRate.rate < wordErrorRateThreshold
    }
}
