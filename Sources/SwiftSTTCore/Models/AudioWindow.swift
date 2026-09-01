import Foundation

/// Why a streaming window stopped accumulating audio.
public enum WindowCutCause: Sendable, Equatable {

    /// The voice activity detector confirmed a falling edge. The window ends
    /// on silence, so no audio is carried into the next window.
    case silence

    /// Speech ran past ``StreamingWindowPolicy/maximumWindowDuration``. The
    /// window was cut mid-speech and the next window re-covers its last
    /// ``StreamingWindowPolicy/overlapDuration`` seconds.
    case maximumDuration

    /// Capture ended. This is the last window of the session.
    case stop
}

/// A span of captured audio cut for one transcription pass.
public struct AudioWindow: Sendable, Equatable {

    /// 16 kHz mono Float32 PCM, in the range `-1.0 ... 1.0`.
    public let samples: [Float]

    /// Offset of the first sample from the start of capture, in seconds.
    public let startTime: TimeInterval

    /// Sample rate in Hz.
    public let sampleRate: Int

    /// Why the window was cut.
    public let cut: WindowCutCause

    /// Creates a new AudioWindow with the supplied values.
    public init(samples: [Float], startTime: TimeInterval, sampleRate: Int = 16_000, cut: WindowCutCause) {
        self.samples = samples
        self.startTime = startTime
        self.sampleRate = sampleRate
        self.cut = cut
    }

    /// Length of ``samples`` in seconds.
    public var duration: TimeInterval { Double(samples.count) / Double(sampleRate) }
}
