import Testing
@testable import SwiftWhisperCore

@Suite("DownloadProgress")
struct DownloadProgressTests {

    @Test("fractionComplete with positive totalBytes")
    func fractionComplete() {
        let p = DownloadProgress(
            totalFiles: 5, completedFiles: 2,
            totalBytes: 200, totalBytesDownloaded: 100,
            phase: .downloading
        )
        #expect(p.fractionComplete == 0.5)
    }

    @Test("fractionComplete returns 0 when totalBytes is 0")
    func fractionCompleteZero() {
        let p = DownloadProgress(
            totalFiles: 0, completedFiles: 0,
            totalBytes: 0, totalBytesDownloaded: 0,
            phase: .listing
        )
        #expect(p.fractionComplete == 0)
    }

    @Test("Phase raw values")
    func phaseRawValues() {
        #expect(DownloadProgress.Phase.listing.rawValue == "listing")
        #expect(DownloadProgress.Phase.complete.rawValue == "complete")
    }
}
