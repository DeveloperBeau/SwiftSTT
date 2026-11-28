import Foundation

/// A first-in, first-out lock for async code.
///
/// A caller that finds the lock held suspends until every earlier caller has released
/// it. No thread blocks while waiting. A waiting caller that is cancelled, for example
/// by a test time limit, stops waiting and throws `CancellationError`, so a lock that is
/// never released fails the test instead of hanging the run.
actor AsyncLock {

    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Void, any Error>
    }

    private var isHeld = false
    private var waiters: [Waiter] = []

    /// Takes the lock, waiting behind earlier callers when it is held.
    func acquire() async throws {
        if !isHeld {
            isHeld = true
            return
        }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation {
                (continuation: CheckedContinuation<Void, any Error>) in
                enqueue(Waiter(id: id, continuation: continuation))
            }
        } onCancel: {
            Task { await self.cancelWaiter(id) }
        }
    }

    private func enqueue(_ waiter: Waiter) {
        if Task.isCancelled {
            waiter.continuation.resume(throwing: CancellationError())
        } else {
            waiters.append(waiter)
        }
    }

    /// Releases the lock, handing it straight to the longest waiting caller.
    func release() {
        if waiters.isEmpty {
            isHeld = false
        } else {
            waiters.removeFirst().continuation.resume()
        }
    }

    private func cancelWaiter(_ id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        waiters.remove(at: index).continuation.resume(throwing: CancellationError())
    }
}
