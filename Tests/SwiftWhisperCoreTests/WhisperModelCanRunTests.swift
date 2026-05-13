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
        #expect(WhisperModel.small.canRun(onPhysicalMemoryBytes: 2 * oneGB) == false)
        #expect(WhisperModel.small.canRun(onPhysicalMemoryBytes: 3 * oneGB) == true)
    }
}
