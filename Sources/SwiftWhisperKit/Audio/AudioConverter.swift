import Foundation

/// Placeholder for a future shared audio-format conversion helper.
///
/// Today's conversion logic lives inline inside ``AVAudioCapture`` because it
/// only needs to handle one direction (microphone format to 16 kHz mono Float32).
/// When file-based input lands in M7 the same plumbing will also need to read
/// `.wav`, `.m4a`, and `.caf` inputs, at which point the shared parts move here.
///
/// > Note: Reserved for future use; carries no behaviour yet.
public struct AudioConverter: Sendable {
    public init() {}
}
