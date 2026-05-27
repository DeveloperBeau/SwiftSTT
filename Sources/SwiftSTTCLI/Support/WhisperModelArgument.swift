import ArgumentParser
import SwiftSTTCore

/// Bridges ``WhisperModel`` into ArgumentParser so subcommands can accept a
/// model name like `tiny` or `base` as a positional argument.
extension WhisperModel: ExpressibleByArgument {

    /// Creates a new WhisperModel with the supplied values.
    public nonisolated init?(argument: String) {
        self.init(rawValue: argument)
    }

    /// All accepted string values for command-line argument completion.
    public nonisolated static var allValueStrings: [String] {
        WhisperModel.allCases.map(\.rawValue)
    }
}
