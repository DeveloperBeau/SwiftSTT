import Testing
@testable import SwiftWhisperCore

struct WhisperModelCanRunTests {

    private let oneGB: UInt64 = 1_073_741_824

    @Test
    func tinyRunsOnOneGigabyte() {
        #expect(WhisperModel.tiny.canRun(onPhysicalMemoryBytes: oneGB) == false)
        #expect(WhisperModel.tiny.canRun(onPhysicalMemoryBytes: 2 * oneGB) == true)
    }

    @Test
    func largeV3TurboRequiresAtLeastThreeGigabytes() {
        #expect(WhisperModel.largeV3Turbo.canRun(onPhysicalMemoryBytes: 2 * oneGB) == false)
        #expect(WhisperModel.largeV3Turbo.canRun(onPhysicalMemoryBytes: 4 * oneGB) == true)
    }

    @Test
    func smallSitsBetweenBaseAndTurbo() {
        // small.approximatePeakRuntimeBytes == 1 GB; needs 1 GB headroom → 2 GB total.
        // canRun(2 GB) is true because 1_000_000_000 + 1_073_741_824 <= 2_147_483_648.
        #expect(WhisperModel.small.canRun(onPhysicalMemoryBytes: 2 * oneGB) == true)
        #expect(WhisperModel.small.canRun(onPhysicalMemoryBytes: 3 * oneGB) == true)
        // Below the required 2 GB budget it cannot run.
        #expect(WhisperModel.small.canRun(onPhysicalMemoryBytes: oneGB) == false)
    }
}
