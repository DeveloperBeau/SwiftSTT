import Foundation
import Testing

/// Locations of the committed accuracy fixture.
enum AccuracyFixture {

    /// The committed recording, `Fixtures/jfk.wav`.
    static var audioURL: URL {
        get throws {
            try #require(
                Bundle.module.url(
                    forResource: "jfk", withExtension: "wav", subdirectory: "Fixtures"))
        }
    }

    /// The contents of the hand-written transcript, `Fixtures/jfk.txt`.
    static var expectedTranscript: String {
        get throws {
            let url = try #require(
                Bundle.module.url(
                    forResource: "jfk", withExtension: "txt", subdirectory: "Fixtures"))
            return try String(contentsOf: url, encoding: .utf8)
        }
    }
}
