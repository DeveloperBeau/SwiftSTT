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

    /// The model under test, cached or downloaded before the test body runs.
    ///
    /// Set up here so a missing model is downloaded or fails the test. The body has no
    /// way to see an uncached model.
    private let prepared: PreparedModel

    init() async throws {
        prepared = try await PreparedModel.prepare(Self.pickModel(), using: ModelDownloader())
    }

    @Test("Tiny model downloads and bundle resolves ggml URL")
    func tinyModelDownloadAndBundle() {
        #expect(FileManager.default.fileExists(atPath: prepared.bundle.ggmlModelURL.path))
    }

    // MARK: - Helpers

    /// Allows the `integration` workflow to override the model via env var.
    private static func pickModel() -> WhisperModel {
        let raw = ProcessInfo.processInfo.environment["SWIFTSTT_INTEGRATION_MODEL"] ?? "tiny"
        return WhisperModel(rawValue: raw) ?? .tiny
    }
}
