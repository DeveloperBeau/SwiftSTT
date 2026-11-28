import Testing

/// Hand-worked cases for the word error rate helper used by the accuracy fixture test.
///
/// Runs in the default gate. It needs no model and no network.
@Suite("Word error rate")
struct WordErrorRateTests {

    private let reference = "ask not what your country can do for you"

    @Test("Identical text scores zero errors")
    func identical() {
        let wer = WordErrorRate(reference: reference, hypothesis: reference)
        #expect(wer == WordErrorRate(errors: 0, referenceWordCount: 9))
        #expect(wer.rate == 0)
    }

    @Test("Case and punctuation differences are not errors")
    func caseAndPunctuationIgnored() {
        let wer = WordErrorRate(
            reference: reference,
            hypothesis: "Ask not, what your COUNTRY can do for you."
        )
        #expect(wer == WordErrorRate(errors: 0, referenceWordCount: 9))
    }

    @Test("One substituted word scores one error in nine")
    func oneSubstitution() {
        let wer = WordErrorRate(
            reference: reference,
            hypothesis: "ask not what your county can do for you"
        )
        #expect(wer == WordErrorRate(errors: 1, referenceWordCount: 9))
        #expect(wer.rate == 1.0 / 9.0)
    }

    @Test("One inserted word scores one error in nine")
    func oneInsertion() {
        let wer = WordErrorRate(
            reference: reference,
            hypothesis: "ask not what your country can really do for you"
        )
        #expect(wer == WordErrorRate(errors: 1, referenceWordCount: 9))
    }

    @Test("One deleted word scores one error in nine")
    func oneDeletion() {
        let wer = WordErrorRate(
            reference: reference,
            hypothesis: "ask not what your country can do you"
        )
        #expect(wer == WordErrorRate(errors: 1, referenceWordCount: 9))
    }

    @Test("Empty hypothesis deletes every reference word")
    func emptyHypothesis() {
        let wer = WordErrorRate(reference: reference, hypothesis: "")
        #expect(wer == WordErrorRate(errors: 9, referenceWordCount: 9))
        #expect(wer.rate == 1)
    }

    @Test("Empty reference and empty hypothesis score zero")
    func bothEmpty() {
        let wer = WordErrorRate(reference: "", hypothesis: " . ")
        #expect(wer == WordErrorRate(errors: 0, referenceWordCount: 0))
        #expect(wer.rate == 0)
    }

    @Test("Empty reference counts each hypothesis word as a full error")
    func emptyReference() {
        let wer = WordErrorRate(reference: "", hypothesis: "thank you")
        #expect(wer == WordErrorRate(errors: 2, referenceWordCount: 0))
        #expect(wer.rate == 2)
    }
}
