import Foundation

/// Word error rate of a hypothesis against a hand-written reference transcript.
///
/// Both texts are normalised for case and punctuation only, then split on whitespace.
/// The rate is the word-level edit distance (substitutions, insertions and deletions)
/// divided by the number of reference words. An empty reference divides by one, so an
/// empty hypothesis scores 0 and every extra word counts as one full error.
struct WordErrorRate: Equatable {

    /// Word-level edit distance between the normalised texts.
    let errors: Int

    /// Number of words in the normalised reference.
    let referenceWordCount: Int

    /// Errors divided by the reference word count.
    var rate: Double {
        Double(errors) / Double(max(referenceWordCount, 1))
    }

    /// Wraps counts worked out by hand, for comparing against a computed result.
    init(errors: Int, referenceWordCount: Int) {
        self.errors = errors
        self.referenceWordCount = referenceWordCount
    }

    /// Scores `hypothesis` against `reference`.
    init(reference: String, hypothesis: String) {
        let referenceWords = Self.words(in: reference)
        let hypothesisWords = Self.words(in: hypothesis)
        self.errors = Self.editDistance(referenceWords, hypothesisWords)
        self.referenceWordCount = referenceWords.count
    }

    /// Lowercases the text, drops punctuation, and splits it on whitespace.
    static func words(in text: String) -> [String] {
        let kept = text.lowercased().unicodeScalars.filter {
            !CharacterSet.punctuationCharacters.contains($0)
        }
        return String(String.UnicodeScalarView(kept))
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
    }

    /// Levenshtein distance over words, one row at a time.
    private static func editDistance(_ reference: [String], _ hypothesis: [String]) -> Int {
        var previous = Array(0...hypothesis.count)
        for (i, referenceWord) in reference.enumerated() {
            var current = [i + 1]
            for (j, hypothesisWord) in hypothesis.enumerated() {
                let substitution = previous[j] + (referenceWord == hypothesisWord ? 0 : 1)
                let deletion = previous[j + 1] + 1
                let insertion = current[j] + 1
                current.append(min(substitution, deletion, insertion))
            }
            previous = current
        }
        return previous[hypothesis.count]
    }
}
