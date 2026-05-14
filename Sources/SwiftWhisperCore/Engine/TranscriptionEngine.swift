import Foundation

/// Common interface for speech-to-text engines.
///
/// Implementers buffer audio between ``start()`` and ``stop()``, then
/// emit ``TranscriptionSegment`` values through ``segmentStream()``.
/// Lifecycle updates flow through ``statusStream()``.
///
/// All methods are async because most implementations are actor-backed.
/// Use ``prepare()`` after the persisted default model is known but
/// before the user can tap record, to warm the model.
public protocol TranscriptionEngine: Sendable {

    /// Stream of engine lifecycle status.
    ///
    /// Each call returns a fresh `AsyncStream`. Implementers should yield
    /// the current status to a new subscriber as soon as it registers, so
    /// late subscribers don't miss the initial state.
    func statusStream() -> AsyncStream<EngineStatus>

    /// Stream of transcribed segments. Emits during or after a recording
    /// session, depending on the implementation.
    func segmentStream() -> AsyncStream<TranscriptionSegment>

    /// Begins loading the persisted default model into memory.
    ///
    /// Returns as soon as loading starts, not when it completes. The
    /// actual outcome is delivered via ``statusStream()``: ``EngineStatus/preparing``
    /// while loading, ``EngineStatus/ready`` on success, ``EngineStatus/failed``
    /// on error, or ``EngineStatus/idle`` if no default is selected or the
    /// model is not yet downloaded. Callers must observe the stream and
    /// wait for ``EngineStatus/ready`` before calling ``start()``, otherwise
    /// ``start()`` will throw.
    func prepare() async

    /// Begins audio capture. Throws if no model is loaded yet.
    func start() async throws

    /// Ends audio capture and emits any pending segments. Idempotent.
    func stop() async
}
