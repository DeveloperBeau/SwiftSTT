import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

@Suite("ModelDownloader background mode")
struct ModelDownloaderBackgroundTests {

    @Test("foreground init keeps existing M4 behavior")
    func foregroundInit() async {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let downloader = ModelDownloader(cacheDirectory: tmp, mode: .foreground)
        let dir = await downloader.cacheDirectory(for: .tiny)
        #expect(dir.path.contains(tmp.path))
    }

    @Test("background init registers delegate under identifier")
    func backgroundInitRegistersDelegate() async {
        let identifier = "swiftwhisper.test.\(UUID().uuidString)"
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let downloader = ModelDownloader(
            cacheDirectory: tmp, mode: .background(identifier: identifier))
        defer { ModelDownloadDelegate.unregister(identifier: identifier) }

        #expect(ModelDownloadDelegate.delegate(for: identifier) != nil)
        _ = downloader
    }

    @Test("Custom urlSession overrides background config")
    func customSessionOverride() async {
        let identifier = "swiftwhisper.test.\(UUID().uuidString)"
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let injected = URLSession(configuration: .ephemeral)
        let downloader = ModelDownloader(
            cacheDirectory: tmp,
            mode: .background(identifier: identifier),
            urlSession: injected
        )
        defer { ModelDownloadDelegate.unregister(identifier: identifier) }
        _ = downloader
        // The delegate is still registered so handleBackgroundEvents can route.
        #expect(ModelDownloadDelegate.delegate(for: identifier) != nil)
    }

    @Test("handleBackgroundEvents stashes completion for known identifier")
    func handleBackgroundEventsStashes() async {
        let identifier = "swiftwhisper.test.\(UUID().uuidString)"
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let injected = URLSession(configuration: .ephemeral)
        _ = ModelDownloader(
            cacheDirectory: tmp,
            mode: .background(identifier: identifier),
            urlSession: injected
        )
        defer { ModelDownloadDelegate.unregister(identifier: identifier) }

        let counter = CounterBox()
        await ModelDownloader.handleBackgroundEvents(identifier: identifier) {
            counter.increment()
        }

        // Counter should still be zero: the completion is stashed, not fired.
        #expect(counter.value == 0)

        // Drain completions and assert one was queued.
        let stashed = ModelDownloadDelegate.delegate(for: identifier)?.drainCompletions() ?? []
        #expect(stashed.count == 1)
    }

    @Test("handleBackgroundEvents for unknown identifier fires completion immediately")
    func handleBackgroundEventsUnknownIdentifier() async {
        let counter = CounterBox()
        await ModelDownloader.handleBackgroundEvents(identifier: "nonexistent.\(UUID().uuidString)")
        {
            counter.increment()
        }
        #expect(counter.value == 1)
    }

    @Test("currentBackgroundDownloads returns empty in foreground mode")
    func currentBackgroundDownloadsForeground() async {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let downloader = ModelDownloader(cacheDirectory: tmp, mode: .foreground)
        let map = await downloader.currentBackgroundDownloads()
        #expect(map.isEmpty)
    }

    @Test("currentBackgroundDownloads returns empty when no tasks active")
    func currentBackgroundDownloadsNoTasks() async {
        let identifier = "swiftwhisper.test.\(UUID().uuidString)"
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let injected = URLSession(configuration: .ephemeral)
        let downloader = ModelDownloader(
            cacheDirectory: tmp,
            mode: .background(identifier: identifier),
            urlSession: injected
        )
        defer { ModelDownloadDelegate.unregister(identifier: identifier) }
        let map = await downloader.currentBackgroundDownloads()
        #expect(map.isEmpty)
    }

    @Test("ModelDownloadMode equality")
    func modeEquality() {
        #expect(ModelDownloadMode.foreground == ModelDownloadMode.foreground)
        #expect(ModelDownloadMode.background(identifier: "a") == .background(identifier: "a"))
        #expect(ModelDownloadMode.background(identifier: "a") != .background(identifier: "b"))
        #expect(ModelDownloadMode.foreground != .background(identifier: "a"))
    }
}

private final class CounterBox: @unchecked Sendable {
    private var _value: Int = 0
    private let lock = NSLock()
    var value: Int { lock.withLock { _value } }
    func increment() { lock.withLock { _value += 1 } }
}
