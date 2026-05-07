import Foundation
import Testing
@testable import SwiftWhisperKit

@Suite("AVAudioCapture")
struct AVAudioCaptureTests {

    @Test("Init does not crash and exposes a stream")
    func initExposesStream() async {
        let capture = AVAudioCapture()
        // Accessing audioStream should not block; no capture has been started.
        _ = capture.audioStream
    }

    @Test("Stop without start is a no-op")
    func stopWithoutStart() async {
        let capture = AVAudioCapture()
        await capture.stopCapture()
    }
}
