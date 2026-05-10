@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Synchronization
import Testing

@testable import SwiftWhisperKit

// MARK: - Mock branchable runner

/// In-memory `BranchableStatefulRunner` mock that records reset, predict, and
/// branch traffic.
///
/// Acts as a stand-in for `MLStateModelRunner` in tests where constructing a real Core ML model is not practical.
private final class MockBranchableRunner: BranchableStatefulRunner, @unchecked Sendable {

    struct Counters: Sendable {
        var resetCount: Int = 0
        var predictCount: Int = 0
        var branchCount: Int = 0
    }

    private let logitsName: String
    private let counters: Mutex<Counters>
    /// Each branched runner gets its own dedicated counters; tests can read
    /// them independently from the parent.
    let parent: MockBranchableRunner?

    init(parent: MockBranchableRunner? = nil, logitsName: String = "logits") {
        self.logitsName = logitsName
        self.counters = Mutex<Counters>(Counters())
        self.parent = parent
    }

    var resetCount: Int { counters.withLock { $0.resetCount } }
    var predictCount: Int { counters.withLock { $0.predictCount } }
    var branchCount: Int { counters.withLock { $0.branchCount } }

    func resetState() async {
        counters.withLock { $0.resetCount += 1 }
    }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        counters.withLock { $0.predictCount += 1 }
        do {
            let array = try MLMultiArray(shape: [1, 1, 4], dataType: .float32)
            let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: 4)
            for i in 0..<4 {
                pointer[i] = Float(i)
            }
            return try MLDictionaryFeatureProvider(dictionary: [logitsName: array])
        } catch {
            throw .decoderFailure("mock provider: \(error.localizedDescription)")
        }
    }

    func branch() async throws(SwiftWhisperError) -> any BranchableStatefulRunner {
        counters.withLock { $0.branchCount += 1 }
        let child = MockBranchableRunner(parent: self, logitsName: logitsName)
        await child.resetState()
        return child
    }
}

@Suite("MLStateModelRunner branching")
struct MLStateModelRunnerBranchTests {

    @Test("branch returns a fresh runner instance")
    func branchReturnsNewInstance() async throws {
        let parent = MockBranchableRunner()
        await parent.resetState()
        let child = try await parent.branch() as? MockBranchableRunner
        #expect(child !== parent)
        #expect(child?.parent === parent)
    }

    @Test("branch increments the parent's branch counter")
    func branchIncrementsParentCounter() async throws {
        let parent = MockBranchableRunner()
        await parent.resetState()
        _ = try await parent.branch()
        _ = try await parent.branch()
        #expect(parent.branchCount == 2)
    }

    @Test("Mutating a branch leaves the parent's counters intact")
    func branchMutationDoesNotAffectParent() async throws {
        let parent = MockBranchableRunner()
        await parent.resetState()
        let predictCountBefore = parent.predictCount

        let child = try await parent.branch()
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "stub": MLFeatureValue(int64: 0)
        ])
        _ = try await child.predict(features: provider)
        _ = try await child.predict(features: provider)

        #expect(parent.predictCount == predictCountBefore)
    }

    @Test("Branched runner starts with a fresh state (resetCount > 0)")
    func branchedRunnerStateIsFresh() async throws {
        let parent = MockBranchableRunner()
        await parent.resetState()
        let child = try await parent.branch() as? MockBranchableRunner
        // The mock signals freshness by calling resetState() on the child.
        #expect((child?.resetCount ?? 0) >= 1)
    }

    @Test("Each branch is independent from siblings")
    func siblingsAreIndependent() async throws {
        let parent = MockBranchableRunner()
        await parent.resetState()
        let a = try await parent.branch() as? MockBranchableRunner
        let b = try await parent.branch() as? MockBranchableRunner

        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "stub": MLFeatureValue(int64: 0)
        ])
        if let a {
            _ = try await a.predict(features: provider)
        }
        #expect((a?.predictCount ?? 0) == 1)
        #expect((b?.predictCount ?? 0) == 0)
    }
}
