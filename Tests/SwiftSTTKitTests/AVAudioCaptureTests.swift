import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTKit

// MARK: - Mock provider

/// In-memory `AudioInputProvider` for tests. Synchronously delivers a fixed
/// sequence of buffers when `start` is called, or throws a configured error.
private actor MockAudioInput: AudioInputProvider {

    enum Behaviour {
        case deliver([[Float]])
        case throwError(SwiftSTTError)
    }

    private let behaviour: Behaviour
    private(set) var startCount = 0
    private(set) var stopCount = 0
    private(set) var receivedRate: Double = 0
    private(set) var receivedBufferDuration: Double = 0

    init(_ behaviour: Behaviour) {
        self.behaviour = behaviour
    }

    func start(
        targetSampleRate: Double,
        bufferDurationSeconds: Double,
        onChunk: @Sendable @escaping ([Float]) -> Void
    ) async throws(SwiftSTTError) {
        startCount += 1
        receivedRate = targetSampleRate
        receivedBufferDuration = bufferDurationSeconds
        switch behaviour {
        case .throwError(let err):
            throw err
        case .deliver(let chunks):
            for chunk in chunks {
                onChunk(chunk)
            }
        }
    }

    func stop() async {
        stopCount += 1
    }
}

// MARK: - Tests

@Suite("AVAudioCapture")
struct AVAudioCaptureTests {

    /// Collect a fixed number of chunks from the stream, then stop the capturer.
    static func collect(
        _ capture: AVAudioCapture,
        count: Int
    ) async -> [AudioChunk] {
        var collected: [AudioChunk] = []
        for await chunk in capture.audioStream {
            collected.append(chunk)
            if collected.count >= count { break }
        }
        return collected
    }

    @Test("Init exposes a stream that the consumer can iterate")
    func initExposesStream() async {
        let mock = MockAudioInput(.deliver([[0.1, 0.2]]))
        let capture = AVAudioCapture(provider: mock)
        // Stream is non-nil and ready.
        let _ = capture.audioStream
        await capture.stopCapture()
    }

    @Test("Stop without start is a no-op")
    func stopWithoutStart() async {
        let mock = MockAudioInput(.deliver([]))
        let capture = AVAudioCapture(provider: mock)
        await capture.stopCapture()
        await #expect(mock.stopCount == 0)
    }

    @Test("Start delivers chunks from the provider into the stream")
    func startDeliversChunks() async throws {
        let mock = MockAudioInput(
            .deliver([
                [0.1, 0.2, 0.3],
                [0.4, 0.5],
                [0.6],
            ]))
        let capture = AVAudioCapture(provider: mock, targetSampleRate: 16_000)

        try await capture.startCapture()
        let chunks = await Self.collect(capture, count: 3)

        #expect(chunks.count == 3)
        #expect(chunks[0].samples == [0.1, 0.2, 0.3])
        #expect(chunks[1].samples == [0.4, 0.5])
        #expect(chunks[2].samples == [0.6])
        for chunk in chunks {
            #expect(chunk.sampleRate == 16_000)
            #expect(chunk.timestamp >= 0)
        }
    }

    @Test("Provider receives the configured rate and buffer duration")
    func providerReceivesConfig() async throws {
        let mock = MockAudioInput(.deliver([[0.0]]))
        let capture = AVAudioCapture(
            provider: mock,
            targetSampleRate: 22_050,
            bufferDurationSeconds: 0.128
        )
        try await capture.startCapture()
        await #expect(mock.receivedRate == 22_050)
        await #expect(mock.receivedBufferDuration == 0.128)
    }

    @Test("Double-start is a no-op")
    func doubleStart() async throws {
        let mock = MockAudioInput(.deliver([[0.0]]))
        let capture = AVAudioCapture(provider: mock)
        try await capture.startCapture()
        try await capture.startCapture()
        await #expect(mock.startCount == 1)
    }

    @Test("Provider error propagates")
    func providerError() async {
        let mock = MockAudioInput(.throwError(.audioCaptureFailed("boom")))
        let capture = AVAudioCapture(provider: mock)
        do {
            try await capture.startCapture()
            Issue.record("expected throw")
        } catch {
            #expect(error == .audioCaptureFailed("boom"))
        }
    }

    @Test("micPermissionDenied propagates")
    func permissionDenied() async {
        let mock = MockAudioInput(.throwError(.micPermissionDenied))
        let capture = AVAudioCapture(provider: mock)
        do {
            try await capture.startCapture()
            Issue.record("expected throw")
        } catch {
            #expect(error == .micPermissionDenied)
        }
    }

    @Test("Stop calls provider.stop and finishes the stream")
    func stopFinishesStream() async throws {
        let mock = MockAudioInput(.deliver([[0.1]]))
        let capture = AVAudioCapture(provider: mock)
        try await capture.startCapture()
        await capture.stopCapture()
        await #expect(mock.stopCount == 1)

        // Stream should be finished; iterating yields zero remaining elements
        // (the one we delivered before stop is still there if not consumed).
        var collected = 0
        for await _ in capture.audioStream {
            collected += 1
        }
        // Either 0 or 1 depending on iteration race; both prove stream finished.
        #expect(collected <= 1)
    }

    @Test("Double-stop is a no-op")
    func doubleStop() async throws {
        let mock = MockAudioInput(.deliver([[0.0]]))
        let capture = AVAudioCapture(provider: mock)
        try await capture.startCapture()
        await capture.stopCapture()
        await capture.stopCapture()
        await #expect(mock.stopCount == 1)
    }

    @Test("Default initializer wires up AVMicrophoneInput")
    func defaultProvider() async {
        // Just construct using defaults. Verifies default param resolves.
        // No startCapture - that requires real hardware.
        let capture = AVAudioCapture()
        let _ = capture.audioStream
        await capture.stopCapture()
    }

    @Test("Sample rate is reflected in emitted chunks")
    func sampleRateInChunks() async throws {
        let mock = MockAudioInput(.deliver([[1.0, 2.0]]))
        let capture = AVAudioCapture(provider: mock, targetSampleRate: 8_000)
        try await capture.startCapture()
        let chunks = await Self.collect(capture, count: 1)
        #expect(chunks.first?.sampleRate == 8_000)
    }

    @Test("Timestamps are monotonic across chunks")
    func monotonicTimestamps() async throws {
        let mock = MockAudioInput(
            .deliver([
                [0.0],
                [0.0],
                [0.0],
            ]))
        let capture = AVAudioCapture(provider: mock)
        try await capture.startCapture()
        let chunks = await Self.collect(capture, count: 3)
        #expect(chunks[0].timestamp <= chunks[1].timestamp)
        #expect(chunks[1].timestamp <= chunks[2].timestamp)
    }
}
