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

    /// The line printed for the run log, for example `tiny model: downloaded`.
    var label: String {
        "\(bundle.model.rawValue) model: \(wasDownloaded ? "downloaded" : "cached")"
    }

    /// Serialises every preparation in this process.
    ///
    /// Two callers with their own downloaders would otherwise both download into the
    /// same cache, and one could remove the file the other is about to use.
    private static let lock = AsyncLock()

    /// Uses the cached model, or downloads it when it is not cached.
    ///
    /// Callers run one at a time, so a caller that waited finds the model cached and
    /// makes no request. A failed download throws its error. A download that ends
    /// without the model on disk throws `modelFileMissing` from the bundle lookup.
    static func prepare(
        _ model: WhisperModel,
        using downloader: ModelDownloader
    ) async throws -> PreparedModel {
        await lock.acquire()
        do {
            let prepared = try await prepareAlone(model, using: downloader)
            await lock.release()
            print(prepared.label)
            return prepared
        } catch {
            await lock.release()
            throw error
        }
    }

    private static func prepareAlone(
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
        return PreparedModel(bundle: bundle, wasDownloaded: !wasCached)
    }
}
