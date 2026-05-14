import Testing

@testable import SwiftWhisperKit

@Suite("Kit Placeholder")
struct PlaceholderTests {

    @Test("Kit target compiles")
    func kitCompiles() {
        let _ = AudioConverter()
    }
}
