import Testing

/// Hand-built hypotheses scored through the same path as the accuracy test.
///
/// Runs in the default gate. It reads only the committed transcript and needs no model.
@Suite("Accuracy fixture score")
struct AccuracyFixtureScoreTests {

    private let handWrittenTranscript =
        "And so my fellow Americans, ask not what your country can do for you,"
        + " ask what you can do for your country.\n"

    @Test("Bundled jfk.txt is the hand-written transcript and is the scoring reference")
    func referenceIsBundledTranscript() throws {
        #expect(try AccuracyFixture.expectedTranscript == handWrittenTranscript)
        let score = try AccuracyFixtureScore(hypothesis: "ask not")
        #expect(score.reference == handWrittenTranscript)
        #expect(score.wordErrorRate == WordErrorRate(errors: 20, referenceWordCount: 22))
    }

    @Test("Three word errors in 22 (13.6%) fail the accuracy threshold")
    func threeErrorsFail() throws {
        // Americans to citizens, you to me, your to the.
        let score = try AccuracyFixtureScore(
            hypothesis: "And so my fellow citizens, ask not what your country can do for me,"
                + " ask what you can do for the country."
        )
        #expect(score.wordErrorRate == WordErrorRate(errors: 3, referenceWordCount: 22))
        #expect(!score.meetsThreshold)
    }

    @Test("Two word errors in 22 (9.1%) meet the accuracy threshold")
    func twoErrorsPass() throws {
        // Americans to citizens, you to me.
        let score = try AccuracyFixtureScore(
            hypothesis: "And so my fellow citizens, ask not what your country can do for me,"
                + " ask what you can do for your country."
        )
        #expect(score.wordErrorRate == WordErrorRate(errors: 2, referenceWordCount: 22))
        #expect(score.meetsThreshold)
    }

    @Test("A rate exactly at the threshold fails")
    func rateAtThresholdFails() {
        #expect(
            !AccuracyFixtureScore.meetsThreshold(WordErrorRate(errors: 1, referenceWordCount: 10)))
    }

    @Test("Empty hypothesis fails the accuracy threshold")
    func emptyHypothesisFails() throws {
        let score = try AccuracyFixtureScore(hypothesis: "")
        #expect(score.wordErrorRate == WordErrorRate(errors: 22, referenceWordCount: 22))
        #expect(!score.meetsThreshold)
    }
}
