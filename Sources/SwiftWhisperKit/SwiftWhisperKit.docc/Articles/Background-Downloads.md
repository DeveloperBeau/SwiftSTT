@Metadata {
    @TechnologyRoot
}

# Background Model Downloads

Continue downloading Whisper models after the host app suspends or exits.

## Overview

Whisper Core ML models are large. ``WhisperModel/largeV3Turbo`` is roughly
800 MB on disk; even ``WhisperModel/tiny`` is 150 MB. On a flaky cellular
connection, a foreground download can take long enough that the user
backgrounds the app or the phone locks. Foreground transfers stop the
moment the system suspends the process.

``ModelDownloader`` supports an opt-in background mode that hands the
transfer to `nsurlsessiond`. The download keeps progressing while the app
is suspended and resumes after relaunch. The library bridges
`URLSessionDownloadDelegate` callbacks back into the same
`AsyncThrowingStream<DownloadProgress, any Error>` API used in foreground
mode, so call-sites do not need to branch.

## When to use background mode

| Use foreground when... | Use background when... |
|---|---|
| The app stays in the foreground for the whole transfer | Downloads run for tens of seconds or longer |
| Downloads are small (a tokenizer JSON, a manifest) | Downloads are tens of MB or more |
| You need progress only while the user watches | The user can leave the app and come back later |

## iOS wiring

Background sessions deliver completion via the app delegate callback
`application(_:handleEventsForBackgroundURLSession:completionHandler:)`.
This is not delivered to a `SceneDelegate`, so apps using the scene-based
lifecycle still need a `UIApplicationDelegate` for this one method.

```swift
import SwiftUI
import SwiftWhisperKit
import UIKit

@main
struct WhisperApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var delegate
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        Task {
            await ModelDownloader.handleBackgroundEvents(
                identifier: identifier,
                completion: completionHandler
            )
        }
    }
}
```

The `completion` closure passed to
``ModelDownloader/handleBackgroundEvents(identifier:completion:)`` fires
once the URLSession reports it has delivered every queued event. Calling
the system handler before that point causes events to be dropped.

## Starting a background download

```swift
let downloader = ModelDownloader(
    cacheDirectory: nil,
    mode: .background(identifier: "com.example.WhisperApp.models")
)

let stream = try await downloader.download(.largeV3Turbo)
for try await progress in stream {
    updateProgress(progress.fractionComplete)
}
```

The progress stream behaves exactly the same as in foreground mode. While
the app is suspended the closure body does not run, but URLSession keeps
writing data in the background. On relaunch, the next iteration of the
stream picks up at the latest delegate-emitted progress.

## Recovering in-flight downloads on relaunch

After the system relaunches the app, query
``ModelDownloader/currentBackgroundDownloads()`` to see which downloads
are still in progress and surface them in the UI:

```swift
let inFlight = await downloader.currentBackgroundDownloads()
for (model, fraction) in inFlight {
    showBanner("Resuming \(model.displayName) at \(Int(fraction * 100))%")
}
```

If the app calls ``ModelDownloader/download(_:)`` for a model whose
transfer is already in flight, the existing stream is returned rather than
starting a new task.

## Identifier scoping

The session identifier passed to
``ModelDownloadMode/background(identifier:)`` must be unique per
`ModelDownloader` instance kept alive in the app. The system rejects
duplicate identifiers and the second downloader receives an empty
session. A stable per-app identifier (e.g. reverse-DNS bundle ID with a
descriptive suffix) is the right shape:

```
com.example.WhisperApp.modelDownloads
```

Do not reuse the identifier across products (a watch extension and the
iPhone app should pick distinct identifiers).

## macOS

macOS apps don't suspend the way iOS apps do, so background sessions are
mostly redundant. The same code still works, with one caveat: the system
does not call
`application(_:handleEventsForBackgroundURLSession:completionHandler:)` on
macOS, so the `handleBackgroundEvents` helper is a no-op. Use foreground
mode unless the app explicitly opts into being terminated mid-download.

## Troubleshooting

- **Stream never yields after relaunch.** The host app forgot to wire the
  delegate callback. Without ``ModelDownloader/handleBackgroundEvents(identifier:completion:)``,
  URLSession waits for the system handler that never fires.
- **`URLError(.backgroundSessionInUseByAnotherProcess)`.** Two
  `ModelDownloader` instances share the same identifier in the same
  process, or a stale session from a previous launch wasn't invalidated.
  Use a unique identifier per downloader and call
  `URLSession.invalidateAndCancel()` from `deinit` if you build short-lived
  downloaders.
- **`URLError(.notConnectedToInternet)` after backgrounding.** The system
  retries on its own; the stream stays alive and progresses once the
  network returns.
