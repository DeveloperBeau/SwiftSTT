import Foundation

/// When a ``WhisperTranscriptionEngine`` runs the model.
public enum TranscriptionTiming: Sendable, Equatable {

    /// Buffer the whole recording and transcribe once on stop. The default,
    /// and what every existing consumer gets.
    case onStop

    /// Transcribe incrementally during capture, cutting windows per the policy.
    case whileRecording(StreamingWindowPolicy)
}
