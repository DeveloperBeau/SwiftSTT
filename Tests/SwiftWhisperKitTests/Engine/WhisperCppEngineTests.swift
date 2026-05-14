import Foundation
import SwiftWhisperCore
import Testing
@testable import SwiftWhisperKit

@Suite("WhisperCppEngine lifecycle")
struct WhisperCppEngineTests {

    @Test("prepare emits .idle when no default model is set")
    func prepareIdle() async {
        let storage = DefaultModelStorage(
            defaults: UserDefaults(suiteName: UUID().uuidString)!
        )
        storage.model = nil
        let engine = WhisperCppEngine(storage: storage)
        await engine.prepare()

        let stream = engine.statusStream()
        var iterator = stream.makeAsyncIterator()
        let status = await iterator.next()
        #expect(status == .idle)
    }

    @Test("start throws when not prepared")
    func startWithoutPrepare() async {
        let storage = DefaultModelStorage(
            defaults: UserDefaults(suiteName: UUID().uuidString)!
        )
        storage.model = nil
        let engine = WhisperCppEngine(storage: storage)
        do {
            try await engine.start()
            Issue.record("expected throw")
        } catch let error as SwiftWhisperError {
            guard case .modelLoadFailed = error else {
                Issue.record("expected .modelLoadFailed, got \(error)")
                return
            }
        } catch {
            Issue.record("unexpected error type: \(error)")
        }
    }

    @Test("stop is idempotent when not started")
    func stopIdempotent() async {
        let storage = DefaultModelStorage(
            defaults: UserDefaults(suiteName: UUID().uuidString)!
        )
        let engine = WhisperCppEngine(storage: storage)
        await engine.stop()
    }
}
