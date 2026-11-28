import Foundation
import SwiftSTTCore
import SwiftSTTKit

/// A model made ready for an integration run.
///
/// ``prepare(_:using:)`` returns a bundle or throws. It has no path that skips, so a
/// missing model is downloaded or the test fails.
struct PreparedModel {

    /// The bundle for the cached model.
    let bundle: ModelBundle

    /// Whether this call downloaded the model, as opposed to finding it cached.
    let wasDownloaded: Bool

    /// Uses the cached model, or downloads it when it is not cached.
    ///
    /// A failed download throws its error. A download that ends without the model on
    /// disk throws `modelFileMissing` from the bundle lookup.
    static func prepare(
        _ model: WhisperModel,
        using downloader: ModelDownloader
    ) async throws -> PreparedModel {
        let wasCached = await downloader.isDownloaded(model)
        if !wasCached {
            let stream = try await downloader.download(model)
            for try await progress in stream {
                if progress.phase == .complete { break }
            }
        }
        let bundle = try await downloader.bundle(for: model)
        print("\(model.rawValue) model: \(wasCached ? "cached" : "downloaded")")
        return PreparedModel(bundle: bundle, wasDownloaded: !wasCached)
    }
}
