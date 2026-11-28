import Synchronization
import Testing

/// Fails a test whose body returns without scoring the fixture.
///
/// The check runs outside the test body, so it also catches a return placed above
/// every check in the body. The body reports a finished score with ``markScored()``.
struct FixtureScoredCheck: TestTrait, TestScoping {

    /// Records whether the running test has scored the fixture.
    final class Record: Sendable {
        private let scored = Mutex(false)

        var wasScored: Bool { scored.withLock { $0 } }

        func markScored() { scored.withLock { $0 = true } }
    }

    @TaskLocal static var current: Record?

    /// Marks the fixture as scored for the running test.
    static func markScored() {
        current?.markScored()
    }

    func provideScope(
        for test: Test,
        testCase: Test.Case?,
        performing function: @concurrent @Sendable () async throws -> Void
    ) async throws {
        let record = Record()
        try await Self.$current.withValue(record) {
            try await function()
        }
        #expect(record.wasScored, "The test body returned before the fixture was scored")
    }
}

extension Trait where Self == FixtureScoredCheck {

    /// Fails the test when its body returns without scoring the fixture.
    static var requiresFixtureScored: Self { FixtureScoredCheck() }
}
