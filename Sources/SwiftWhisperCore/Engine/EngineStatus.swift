import Foundation

/// Lifecycle status of a transcription engine.
///
/// Engines surface their current state through ``TranscriptionEngine/statusStream()``.
/// UI code observes the stream and renders splash/ready/listening affordances
/// accordingly.
public enum EngineStatus: Sendable, Equatable {

    /// No default model selected, or models not yet downloaded.
    case idle

    /// Loading a downloaded model into memory.
    case preparing

    /// A model download is in progress. `progress` is in [0, 1].
    case downloadingModel(progress: Float)

    /// Loaded and ready to start a recording.
    case ready

    /// Microphone is open and accumulating samples.
    case listening

    /// Engine entered an error state and won't transcribe until reset.
    case failed(String)
}
