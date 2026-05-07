import Foundation

public actor WhisperTokenizer {
    public init() {}

    public func encode(text: String) -> [Int] { [] }
    public func decode(tokens: [Int]) -> String { "" }
    public func isTimestamp(token: Int) -> Bool { false }
    public func isSpecial(token: Int) -> Bool { false }
    public var endOfTextToken: Int { 50_257 }
    public var startOfTranscriptToken: Int { 50_258 }
}
