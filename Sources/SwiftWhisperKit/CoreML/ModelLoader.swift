import Foundation

/// Deprecated: model loading is now handled by `WhisperCppContext`.
///
/// This file remains to satisfy compile-time references and will be
/// deleted in Task 9.
@available(*, deprecated, message: "Use WhisperCppContext directly.")
public struct ModelLoader: Sendable {
    /// Creates a no-op ModelLoader.
    ///
    /// Use `WhisperCppContext` instead.
    public init() {}
}

/// Deprecated: removed; use WhisperCppContext.
@available(*, deprecated, message: "Removed; use WhisperCppContext.")
public struct LoadedModels: Sendable {
    /// Creates an empty LoadedModels placeholder.
    ///
    /// This type is deprecated; use `WhisperCppContext` instead.
    public init() {}
}
