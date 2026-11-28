import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTKit

extension IntegrationRun {

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
        .disabled(if: ProcessInfo.processInfo.environment["SWIFTSTT_RUN_INTEGRATION"] == nil),
        .timeLimit(.minutes(10))
    )
    struct IntegrationTests {

        /// The model under test, cached or downloaded before the test body runs.
        ///
        /// Set up here so a missing model is downloaded or fails the test. The body has no
        /// way to see an uncached model.
        private let prepared: PreparedModel

        /// Whether the model was in the cache before ``prepared`` was made.
        private let modelExistedBeforeInit: Bool

        init() async throws {
            let model = Self.pickModel()
            modelExistedBeforeInit = IntegrationRun.modelIsInDefaultCache(model)
            prepared = try await PreparedModel.prepare(model, using: ModelDownloader())
            print("Integration: \(prepared.label)")
        }

        @Test("Tiny model downloads and bundle resolves ggml URL")
        func tinyModelDownloadAndBundle() {
            #expect(FileManager.default.fileExists(atPath: prepared.bundle.ggmlModelURL.path))
            #expect(prepared.wasDownloaded == !modelExistedBeforeInit)
            #expect(
                prepared.label
                    == "\(Self.pickModel().rawValue) model: "
                    + (modelExistedBeforeInit ? "cached" : "downloaded"))
        }

        // MARK: - Helpers

        /// Allows the `integration` workflow to override the model via env var.
        private static func pickModel() -> WhisperModel {
            let raw = ProcessInfo.processInfo.environment["SWIFTSTT_INTEGRATION_MODEL"] ?? "tiny"
            return WhisperModel(rawValue: raw) ?? .tiny
        }
    }
}
