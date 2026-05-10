import Foundation

/// Resolves the user-supplied `--cache-dir` value to an absolute `URL`.
///
/// Returns `nil` when the user did not supply a directory; callers pass that
/// straight through to `ModelDownloader(cacheDirectory:)` which falls back to
/// the platform default (`~/Library/Application Support/SwiftWhisper/Models/`).
///
/// Tilde expansion is honored so users can type `--cache-dir ~/whisper-cache`
/// without the shell having to expand it (useful when the value comes from a
/// config file or env var).
enum CacheDirectoryOption {

    nonisolated static func resolve(_ raw: String?) -> URL? {
        guard let raw, !raw.isEmpty else { return nil }
        let expanded = (raw as NSString).expandingTildeInPath
        return URL(fileURLWithPath: expanded, isDirectory: true)
    }
}
