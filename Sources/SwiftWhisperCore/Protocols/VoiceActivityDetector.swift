import Foundation

/// Decides whether a given audio chunk contains speech.
///
/// Voice activity detection sits between capture and the heavy mel-plus-decoder
/// path. Filtering out silence has two practical wins:
///
/// 1. The decoder spends compute only on chunks that might contain words,
///    dropping average inference cost noticeably.
/// 2. Whisper has a known habit of hallucinating text on silent input. A VAD
///    pass is the cheapest defence against that.
///
/// The Kit module ships ``SwiftWhisperKit/EnergyVAD`` (RMS-based, no model needed).
/// A neural VAD is planned via ``SwiftWhisperKit/SileroVAD`` for harder noise
/// conditions where pure energy thresholds struggle.
public protocol VoiceActivityDetector: Actor {

    /// Returns `true` if the chunk should be treated as speech.
    /// Implementations may use internal state, so calls are not pure: the same
    /// chunk fed twice can produce different answers depending on hysteresis,
    /// adaptive threshold updates, etc.
    func isSpeech(chunk: AudioChunk) async -> Bool

    /// Clears any internal state (noise floor, hysteresis counters). Use when
    /// switching audio sources or after a long pause.
    func reset() async
}
