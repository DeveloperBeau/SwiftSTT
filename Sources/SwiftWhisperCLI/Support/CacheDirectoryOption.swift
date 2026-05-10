import Foundation

/// Resolves the user-supplied `--cache-dir` value to an absolute `URL`.
///
/// Precedence (highest first):
/// 1. Explicit `--cache-dir <path>` flag passed on the command line.
/// 2. `SWIFTWHISPER_CACHE_DIR` environment variable.
/// 3. Built-in default applied by `ModelDownloader`
///    (`~/Library/Application Support/SwiftWhisper/Models/`).
///
/// Returns `nil` only when no source provides a value. Callers pass `nil`
/// straight through to `ModelDownloader(cacheDirectory:)`, which falls back
/// to the platform default.
///
/// Tilde expansion is honored on every source so `~/whisper-cache` works
/// without requiring shell expansion (useful for env-supplied paths).
enum CacheDirectoryOption {

    /// Environment variable name checked when no explicit `--cache-dir` flag
    /// is supplied. Made into a constant so tests can refer to it without a
    /// magic string.
    nonisolated static let environmentVariableName = "SWIFTWHISPER_CACHE_DIR"

    nonisolated static func resolve(
        _ raw: String?,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> URL? {
        if let raw, !raw.isEmpty {
            return expand(raw)
        }
        if let envValue = environment[environmentVariableName], !envValue.isEmpty {
            return expand(envValue)
        }
        return nil
    }

    nonisolated private static func expand(_ raw: String) -> URL {
        let expanded = (raw as NSString).expandingTildeInPath
        return URL(fileURLWithPath: expanded, isDirectory: true)
    }
}
