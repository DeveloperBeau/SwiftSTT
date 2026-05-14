import Foundation

/// Persists the user's selected default model via UserDefaults.
///
/// Murmur and the CLI share this so they agree on which model to load.
/// The underlying key is `"SwiftWhisper.defaultModel"`.
///
/// Reads and writes are not synchronised across processes; if both the
/// CLI and the host app are running and both write, last-writer wins.
public final class DefaultModelStorage: @unchecked Sendable {

    private static let key = "SwiftWhisper.defaultModel"
    private let defaults: UserDefaults

    /// Creates a storage backed by the supplied UserDefaults suite.
    ///
    /// Pass a custom suite for tests so they don't pollute `.standard`.
    public init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    /// The currently selected model, or `nil` if none is selected.
    public var model: WhisperModel? {
        get {
            guard let raw = defaults.string(forKey: Self.key) else { return nil }
            return WhisperModel(rawValue: raw)
        }
        set {
            defaults.set(newValue?.rawValue, forKey: Self.key)
        }
    }
}
