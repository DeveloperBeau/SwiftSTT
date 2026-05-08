import Foundation
import SwiftWhisperCore

/// Neural voice activity detector using a Silero VAD model.
///
/// Silero VAD is a small recurrent network trained on a wide range of
/// background noise. It outperforms ``EnergyVAD`` in conditions where energy
/// alone gives false positives: music, fan noise, traffic, and conversations
/// in the background.
///
/// > Important: Stub. Always returns `false`. Real implementation needs the
/// > Silero ONNX model converted to Core ML and bundled with the app, plus the
/// > 16 kHz mono float-input wrapper around it. Tracked in milestone M2.5
/// > (post-M2 polish).
public actor SileroVAD: VoiceActivityDetector {
    public init() {}

    public func isSpeech(chunk: AudioChunk) async -> Bool {
        false
    }

    public func reset() async {}
}
