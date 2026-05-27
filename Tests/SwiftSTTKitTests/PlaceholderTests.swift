import Testing

@testable import SwiftSTTKit

@Suite("Kit Placeholder")
struct PlaceholderTests {

    @Test("Kit target compiles")
    func kitCompiles() {
        let _ = AudioConverter()
    }
}
