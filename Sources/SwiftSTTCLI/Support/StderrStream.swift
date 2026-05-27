import Foundation

/// Writes a UTF-8 string to stderr without a trailing newline.
///
/// Plain `print(_:to:)` requires an `inout TextOutputStream`, which is awkward
/// to share across actors. This direct write keeps the call sites short and
/// avoids the variable shuffling.
nonisolated func writeStderr(_ string: String) {
    guard let data = string.data(using: .utf8) else { return }
    FileHandle.standardError.write(data)
}
