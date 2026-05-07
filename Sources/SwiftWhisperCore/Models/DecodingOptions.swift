import Foundation

public enum TaskKind: String, Sendable, Equatable {
    case transcribe
    case translate
}

public struct DecodingOptions: Sendable, Equatable {
    public var language: String?
    public var task: TaskKind
    public var temperature: Float
    public var beamSize: Int
    public var usePrefixTimestamps: Bool
    public var suppressBlank: Bool
    public var suppressTokens: [Int]

    public init(
        language: String? = nil,
        task: TaskKind = .transcribe,
        temperature: Float = 0.0,
        beamSize: Int = 5,
        usePrefixTimestamps: Bool = true,
        suppressBlank: Bool = true,
        suppressTokens: [Int] = []
    ) {
        self.language = language
        self.task = task
        self.temperature = temperature
        self.beamSize = beamSize
        self.usePrefixTimestamps = usePrefixTimestamps
        self.suppressBlank = suppressBlank
        self.suppressTokens = suppressTokens
    }

    public static let `default` = DecodingOptions()
}
