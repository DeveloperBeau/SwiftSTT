import ArgumentParser
import SwiftWhisperCore

/// Bridges ``WhisperModel`` into ArgumentParser so subcommands can accept a
/// model name like `tiny` or `base` as a positional argument.
extension WhisperModel: ExpressibleByArgument {

    public nonisolated init?(argument: String) {
        self.init(rawValue: argument)
    }

    public nonisolated static var allValueStrings: [String] {
        WhisperModel.allCases.map(\.rawValue)
    }
}
