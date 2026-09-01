import Testing

@testable import SwiftSTTCore

@Suite("AudioWindow")
struct AudioWindowTests {

    @Test("W1: duration at the default sample rate")
    func durationAtDefaultRate() {
        let window = AudioWindow(
            samples: Array(repeating: 0, count: 8_000), startTime: 0, cut: .stop)
        #expect(window.duration == 0.5)
    }

    @Test("W2: duration at a non-default sample rate")
    func durationAtNonDefaultRate() {
        let window = AudioWindow(
            samples: Array(repeating: 0, count: 8_000), startTime: 0, sampleRate: 8_000, cut: .stop)
        #expect(window.duration == 1.0)
    }

    @Test("W3: duration of an empty window")
    func durationOfEmptyWindow() {
        let window = AudioWindow(samples: [], startTime: 0, cut: .stop)
        #expect(window.duration == 0)
    }

    @Test("W4: false-positive — the input that must not produce W1's answer")
    func durationDoesNotMatchWrongInput() {
        let window = AudioWindow(
            samples: Array(repeating: 0, count: 16_000), startTime: 0, cut: .stop)
        #expect(window.duration != 0.5)
        #expect(window.duration == 1.0)
    }
}
