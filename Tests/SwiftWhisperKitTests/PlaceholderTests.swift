import Testing

@testable import SwiftWhisperKit

@Suite("Kit Placeholder")
struct PlaceholderTests {

    @Test("Kit target compiles")
    func kitCompiles() throws {
        let _ = AudioConverter()
        let _ = try FFTProcessor()
        let _ = BPETokenizer()
    }
}
