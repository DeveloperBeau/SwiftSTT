import Foundation
import SwiftWhisperCore
import Testing

@testable import SwiftWhisperKit

/// Tests for ModelDownloader background mode API surface.
///
/// Background download mode is deferred (Task 7 spec). These tests verify
/// the current deferred-mode contracts: the API compiles, modes are
/// distinguishable, and the public entry points return gracefully.
@Suite("ModelDownloader background mode")
struct ModelDownloaderBackgroundTests {

    @Test("foreground init keeps existing M4 behavior")
    func foregroundInit() async {
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let downloader = ModelDownloader(cacheDirectory: tmp, mode: .foreground)
        let dir = await downloader.cacheDirectory(for: .tiny)
        #expect(dir.path.contains(tmp.path))
    }

    @Test("background init does not crash (deferred mode)")
    func backgroundInitDoesNotCrash() async {
        let identifier = "swiftwhisper.test.\(UUID().uuidString)"
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let downloader = ModelDownloader(
            cacheDirectory: tmp, mode: .background(identifier: identifier))
        let dir = await downloader.cacheDirectory(for: .tiny)
        #expect(dir.path.contains(tmp.path))
    }

    @Test("Custom urlSession accepted in background mode (deferred)")
    func customSessionOverride() async {
        let identifier = "swiftwhisper.test.\(UUID().uuidString)"
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let injected = URLSession(configuration: .ephemeral)
        let downloader = ModelDownloader(
            cacheDirectory: tmp,
            mode: .background(identifier: identifier),
            urlSession: injected
        )
        let dir = await downloader.cacheDirectory(for: .tiny)
        #expect(dir.path.contains(tmp.path))
    }

    @Test("handleBackgroundEvents fires completion immediately (deferred mode)")
    func handleBackgroundEventsFiresImmediately() async {
        let identifier = "swiftwhisper.test.\(UUID().uuidString)"
        let counter = CounterBox()
        await ModelDownloader.handleBackgroundEvents(identifier: identifier) {
            counter.increment()
        }
        // In deferred mode the completion fires immediately (no stashing).
        #expect(counter.value == 1)
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

    @Test("currentBackgroundDownloads returns empty in deferred background mode")
    func currentBackgroundDownloadsDeferred() async {
        let identifier = "swiftwhisper.test.\(UUID().uuidString)"
        let tmp = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let injected = URLSession(configuration: .ephemeral)
        let downloader = ModelDownloader(
            cacheDirectory: tmp,
            mode: .background(identifier: identifier),
            urlSession: injected
        )
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
