import ArgumentParser
import Foundation

/// Sink for transcription output.
///
/// Wraps either standard output or a file handle so the rest of the CLI can stream lines through one writer without
/// branching on every call.
///
/// Use ``stdout()`` for the default behaviour and ``file(at:noClobber:)`` for
/// the `-o, --output <path>` flag. Tilde paths are expanded (`~/foo` becomes
/// the absolute path under the user's home).
///
/// `--no-clobber` causes ``file(at:noClobber:)`` to throw if the destination
/// already exists. Otherwise existing files are overwritten.
final class OutputDestination: @unchecked Sendable {

    private let handle: FileHandle
    private let shouldClose: Bool

    nonisolated private init(handle: FileHandle, shouldClose: Bool) {
        self.handle = handle
        self.shouldClose = shouldClose
    }

    /// Wraps `FileHandle.standardOutput`.
    ///
    /// Never closes on `close()`.
    nonisolated static func stdout() -> OutputDestination {
        OutputDestination(handle: .standardOutput, shouldClose: false)
    }

    /// Opens (or truncates) the file at `path` for writing.
    ///
    /// Throws a `ValidationError` when `noClobber` is set and the file already exists,
    /// when tilde expansion produces an unreachable path, or when the
    /// underlying file system call fails (read-only mount, missing parent
    /// directory, permission denied).
    nonisolated static func file(
        at path: String,
        noClobber: Bool
    ) throws -> OutputDestination {
        let expanded = (path as NSString).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded)
        let manager = FileManager.default
        if manager.fileExists(atPath: url.path) {
            if noClobber {
                throw ValidationError(
                    "output: \(url.path) already exists and --no-clobber was set"
                )
            }
        } else {
            let created = manager.createFile(atPath: url.path, contents: nil)
            guard created else {
                throw ValidationError(
                    "output: failed to create \(url.path) (check parent directory exists and is writable)"
                )
            }
        }

        let handle: FileHandle
        do {
            handle = try FileHandle(forWritingTo: url)
        } catch {
            throw ValidationError(
                "output: cannot open \(url.path) for writing: \(error.localizedDescription)"
            )
        }
        do {
            try handle.truncate(atOffset: 0)
        } catch {
            try? handle.close()
            throw ValidationError(
                "output: cannot truncate \(url.path): \(error.localizedDescription)"
            )
        }
        return OutputDestination(handle: handle, shouldClose: true)
    }

    /// Writes `string` followed by a single `\n`.
    nonisolated func writeLine(_ string: String) {
        write(string)
        write("\n")
    }

    /// Writes `string` with no trailing newline.
    nonisolated func write(_ string: String) {
        guard let data = string.data(using: .utf8) else { return }
        handle.write(data)
    }

    /// Closes the underlying handle when this destination owns one.
    ///
    /// No-op for stdout.
    nonisolated func close() {
        guard shouldClose else { return }
        try? handle.close()
    }
}
