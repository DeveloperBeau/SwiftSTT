import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTKit

/// End-to-end suite that downloads a real Whisper model and runs it through
/// the whisper.cpp engine pipeline.
///
/// Disabled by default. Run with:
///
/// ```
/// SWIFTSTT_RUN_INTEGRATION=1 swift test --filter SwiftSTTIntegrationTests
/// ```
///
/// On a fresh machine the suite downloads the tiny model (~75 MB) into
/// `~/Library/Application Support/SwiftSTT/Models/`. Subsequent runs are
/// cached.
@Suite(
    "Integration",
    .disabled(if: ProcessInfo.processInfo.environment["SWIFTSTT_RUN_INTEGRATION"] == nil)
)
struct IntegrationTests {

    @Test("Tiny model downloads and bundle resolves ggml URL")
    func tinyModelDownloadAndBundle() async throws {
        let model = pickModel()

        // Phase 1: download (skipped if already cached).
        let downloader = ModelDownloader()
        if await !downloader.isDownloaded(model) {
            let stream = try await downloader.download(model)
            for try await progress in stream {
                if progress.phase == .complete { break }
            }
        }

        let bundle = try await downloader.bundle(for: model)
        #expect(FileManager.default.fileExists(atPath: bundle.ggmlModelURL.path))
    }

    // MARK: - Helpers

    /// Allows the `integration` workflow to override the model via env var.
    private func pickModel() -> WhisperModel {
        let raw = ProcessInfo.processInfo.environment["SWIFTSTT_INTEGRATION_MODEL"] ?? "tiny"
        return WhisperModel(rawValue: raw) ?? .tiny
    }
}
