import Foundation

/// Selects the URLSession transport backing a ``ModelDownloader``.
///
/// ``foreground`` uses the default URLSession (or the one injected at init)
/// and only runs while the host process is in memory. ``background(identifier:)``
/// configures `URLSessionConfiguration.background` so the system continues the
/// transfer after the app suspends or exits.
///
/// On iOS the host app must implement
/// `application(_:handleEventsForBackgroundURLSession:completionHandler:)` and
/// forward to ``ModelDownloader/handleBackgroundEvents(identifier:completion:)``
/// for the system to deliver completion events after relaunch.
public enum ModelDownloadMode: Sendable, Equatable {

    /// URLSession.shared (or the injected session). Existing M4 behaviour.
    case foreground

    /// `URLSessionConfiguration.background(withIdentifier:)`. Survives suspension.
    /// The identifier scopes the system-managed session and must be unique per
    /// downloader instance the app keeps alive.
    case background(identifier: String)
}
