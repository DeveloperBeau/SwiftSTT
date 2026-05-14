import ArgumentParser
import Foundation
import SwiftWhisperCore
import SwiftWhisperKit

/// Prints the on-disk paths and download status for one model.
///
/// Useful when checking whether a previous `download` call landed everything
/// the loader expects (`AudioEncoder.mlmodelc`, `TextDecoder.mlmodelc`,
/// `tokenizer.json`).
///
/// `--cache-dir` overrides the default Application Support location.
struct InfoCommand: AsyncParsableCommand {

    static let configuration = CommandConfiguration(
        commandName: "info",
        abstract: "Show paths and download status for a single model."
    )

    @Argument(help: "Model name (tiny, base, small, largeV3Turbo).")
    var model: WhisperModel

    @Option(
        name: .long,
        help:
            "Override the model cache directory. Defaults to ~/Library/Application Support/SwiftWhisper/Models."
    )
    var cacheDir: String?

    func run() async throws {
        let downloader = ModelDownloader(cacheDirectory: CacheDirectoryOption.resolve(cacheDir))
        let dir = await downloader.cacheDirectory(for: model)
        let downloaded = await downloader.isDownloaded(model)

        print("name:        \(model.rawValue)")
        print("display:     \(model.displayName)")
        print("approx size: \(formatBytes(model.approximateSizeBytes))")
        print("cache dir:   \(dir.path)")
        print("status:      \(downloaded ? "downloaded" : "not downloaded")")

        if downloaded {
            do {
                let bundle = try await downloader.bundle(for: model)
                print("ggml model:  \(bundle.ggmlModelURL.path)")
                if let coreMLURL = bundle.coreMLEncoderURL {
                    print("coreml:      \(coreMLURL.path)")
                }
            } catch {
                print("warning:     downloaded marker present but bundle is incomplete (\(error))")
            }
        } else {
            print("hint:        run 'swiftwhisper download \(model.rawValue)' to fetch")
        }
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let mb = Double(bytes) / 1_000_000
        return String(format: "%.0f MB", mb)
    }
}
