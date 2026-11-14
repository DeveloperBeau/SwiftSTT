import Testing

@testable import SwiftSTTKit

@Suite("OneShotFlag")
struct OneShotFlagTests {

    @Test("claim succeeds once, then fails")
    func claimsOnce() {
        let flag = OneShotFlag()
        #expect(flag.claim())
        #expect(!flag.claim())
        #expect(!flag.claim())
    }

    @Test("concurrent claims produce exactly one winner")
    func oneWinnerUnderContention() async {
        let flag = OneShotFlag()
        let winners = await withTaskGroup(of: Bool.self) { group in
            for _ in 0..<64 { group.addTask { flag.claim() } }
            return await group.reduce(0) { $0 + ($1 ? 1 : 0) }
        }
        #expect(winners == 1)
    }
}
