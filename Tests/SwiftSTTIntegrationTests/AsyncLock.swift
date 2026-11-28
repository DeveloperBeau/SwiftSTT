/// A first-in, first-out lock for async code.
///
/// A caller that finds the lock held suspends until every earlier caller has released
/// it. No thread blocks while waiting.
actor AsyncLock {

    private var isHeld = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    /// Takes the lock, waiting behind earlier callers when it is held.
    func acquire() async {
        if !isHeld {
            isHeld = true
            return
        }
        await withCheckedContinuation { waiters.append($0) }
    }

    /// Releases the lock, handing it straight to the longest waiting caller.
    func release() {
        if waiters.isEmpty {
            isHeld = false
        } else {
            waiters.removeFirst().resume()
        }
    }
}
