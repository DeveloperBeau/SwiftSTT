import Darwin
import Foundation
import Synchronization

/// Single global SIGINT (Ctrl+C) latch shared by the long-running CLI
/// subcommands.
///
/// The mic transcription loop polls ``isStopRequested()`` between segment
/// yields and shuts the pipeline down cleanly when the flag flips. Tests
/// drive the latch directly via ``markStopRequested()`` and ``reset()``.
///
/// `Mutex<Bool>` makes the latch safe to set from a C signal handler context
/// (signal handlers are async-signal-safe-restricted; `Mutex.withLock` is a
/// short, allocation-free critical section).
enum SignalHandler {

    nonisolated private static let stopRequested: Mutex<Bool> = Mutex(false)

    /// Installs the SIGINT handler. Idempotent. The C handler defers to
    /// ``markStopRequested()``.
    nonisolated static func installSIGINT() {
        signal(SIGINT, swiftWhisperHandleSIGINT)
    }

    /// Restores the default SIGINT handler (process termination).
    nonisolated static func uninstallSIGINT() {
        signal(SIGINT, SIG_DFL)
    }

    nonisolated static func isStopRequested() -> Bool {
        stopRequested.withLock { $0 }
    }

    nonisolated static func markStopRequested() {
        stopRequested.withLock { $0 = true }
    }

    nonisolated static func reset() {
        stopRequested.withLock { $0 = false }
    }
}

/// C-callable SIGINT handler. Forwards to ``SignalHandler/markStopRequested()``.
///
/// Has to live as a free `@_cdecl` function because Swift closures cannot
/// safely be cast to a `sig_t` C function pointer under strict concurrency.
@_cdecl("swiftWhisperHandleSIGINT")
nonisolated func swiftWhisperHandleSIGINT(_ signal: Int32) {
    SignalHandler.markStopRequested()
}
