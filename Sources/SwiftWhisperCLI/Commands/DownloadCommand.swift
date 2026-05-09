import ArgumentParser
import Foundation
import SwiftWhisperCore
import SwiftWhisperKit

/// Downloads a Whisper Core ML model into the default cache.
///
/// Progress lines stream to stderr so they don't pollute redirected stdout.
/// Each fractional update overwrites the previous line via `\r`. When the
/// download finishes, a final newline is written and the process exits 0.
struct DownloadCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "download",
        abstract: "Download a Whisper model into the local cache."
    )

    @Argument(help: "Model name to download (tiny, base, small, largeV3Turbo).")
    var model: WhisperModel

    func run() async throws {
        let downloader = ModelDownloader()

        if await downloader.isDownloaded(model) {
            print("\(model.rawValue) already downloaded.")
            return
        }

        let stream = try await downloader.download(model)
        var lastPercentReported: Double = -1

        do {
            for try await progress in stream {
                let percent = progress.fractionComplete * 100
                if percent - lastPercentReported >= 0.1 || progress.phase == .complete {
                    let line = String(
                        format: "\rDownloading \(model.rawValue): %.1f%%",
                        percent
                    )
                    writeStderr(line)
                    lastPercentReported = percent
                }
            }
            writeStderr("\n")
            print("\(model.rawValue) downloaded.")
        } catch {
            writeStderr("\n")
            throw error
        }
    }
}
