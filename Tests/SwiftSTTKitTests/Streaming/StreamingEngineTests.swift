import Foundation
import SwiftSTTCore
import Testing

@testable import SwiftSTTKit

/// In-memory `AudioInputProvider` for tests. Synchronously delivers a fixed
/// sequence of buffers when `start` is called. Copied from
/// `AVAudioCaptureTests.swift`'s private double rather than shared, per spec.
private actor MockAudioInput: AudioInputProvider {
    private let buffers: [[Float]]
    init(_ buffers: [[Float]]) { self.buffers = buffers }

    func start(
        targetSampleRate: Double,
        bufferDurationSeconds: Double,
        onChunk: @Sendable @escaping ([Float]) -> Void
    ) async throws(SwiftSTTError) {
        for chunk in buffers {
            onChunk(chunk)
        }
    }

    func stop() async {}
}

/// Returns a scripted result per call and records what it was handed.
private actor ScriptedWindowDecoder {
    enum Step {
        case segments([TranscriptionSegment])
        case failure(SwiftSTTError)
        case slow([TranscriptionSegment], nanoseconds: UInt64)
    }

    private let steps: [Step]
    private var index = 0
    private(set) var receivedSampleCounts: [Int] = []

    init(_ steps: [Step]) { self.steps = steps }

    func decode(_ samples: [Float]) async throws -> [TranscriptionSegment] {
        receivedSampleCounts.append(samples.count)
        defer { index += 1 }
        guard index < steps.count else { return [] }
        switch steps[index] {
        case .segments(let segments):
            return segments
        case .failure(let error):
            throw error
        case .slow(let segments, let nanoseconds):
            try await Task.sleep(nanoseconds: nanoseconds)
            return segments
        }
    }

    var callCount: Int { receivedSampleCounts.count }
}

private actor ScriptedVoiceActivityDetector: VoiceActivityDetector {
    private let verdicts: [Bool]
    private var index = 0
    init(_ verdicts: [Bool]) { self.verdicts = verdicts }
    func isSpeech(chunk: AudioChunk) async -> Bool {
        defer { index += 1 }
        return index < verdicts.count && verdicts[index]
    }
    func reset() async { index = 0 }
}

private func segment(_ text: String, _ start: TimeInterval, _ end: TimeInterval) -> TranscriptionSegment {
    TranscriptionSegment(text: text, start: start, end: end)
}

/// Races the stream against a timeout so "nothing arrived" is a value rather than a hang.
private func firstSegment(
    from stream: AsyncStream<TranscriptionSegment>,
    within seconds: Double
) async -> TranscriptionSegment? {
    await withTaskGroup(of: TranscriptionSegment?.self) { group in
        group.addTask {
            var iterator = stream.makeAsyncIterator()
            return await iterator.next()
        }
        group.addTask {
            try? await Task.sleep(nanoseconds: UInt64(seconds * 1_000_000_000))
            return nil
        }
        let result = await group.next() ?? nil
        group.cancelAll()
        return result
    }
}

private func makeEngine(
    provider: any AudioInputProvider,
    timing: TranscriptionTiming,
    decoder: ScriptedWindowDecoder,
    verdicts: [Bool] = [],
    policy: StreamingWindowPolicy = StreamingWindowPolicy(
        maximumWindowDuration: 1.0, minimumWindowDuration: 0.5, overlapDuration: 0.25)
) -> WhisperCppEngine {
    WhisperCppEngine(
        storage: WhisperModelStorage(defaults: UserDefaults(suiteName: UUID().uuidString)!),
        audioFactory: { provider },
        timing: timing,
        transcribeWindow: { samples in try await decoder.decode(samples) },
        makeCutter: { policy in
            AudioWindowCutter(
                policy: policy,
                detector: ScriptedVoiceActivityDetector(verdicts),
                refiner: VADBoundaryRefiner(startConsecutive: 1, endConsecutive: 1, sampleRate: 16_000)
            )
        }
    )
}

@Suite("WhisperCppEngine streaming")
struct StreamingEngineTests {

    @Test("E0: onStop hands the decoder samples in capture order")
    func onStopPreservesCaptureOrder() async throws {
        let buffers = (1...200).map { [Float($0)] }
        let provider = MockAudioInput(buffers)
        let decoder = ScriptedWindowDecoder([.segments([])])
        let engine = makeEngine(provider: provider, timing: .onStop, decoder: decoder)

        try await engine.start()
        await engine.stop()

        let received = await decoder.receivedSampleCounts
        #expect(received.count == 1)
        let expected = (1...200).map(Float.init)
        // The single decode call must have received the full, correctly
        // ordered buffer; compare its length here and verify ordering via a
        // decoder that records the actual samples below.
        #expect(received == [expected.count])
    }

    @Test("E0b: onStop hands the decoder the exact ordered sample array")
    func onStopOrdersExactSamples() async throws {
        let buffers = (1...200).map { [Float($0)] }
        let provider = MockAudioInput(buffers)

        actor RecordingDecoder {
            private(set) var received: [Float] = []
            func decode(_ samples: [Float]) async throws -> [TranscriptionSegment] {
                received = samples
                return []
            }
        }
        let recorder = RecordingDecoder()
        let engine = WhisperCppEngine(
            storage: WhisperModelStorage(defaults: UserDefaults(suiteName: UUID().uuidString)!),
            audioFactory: { provider },
            timing: .onStop,
            transcribeWindow: { samples in try await recorder.decode(samples) }
        )

        try await engine.start()
        await engine.stop()

        let received = await recorder.received
        let expected = (1...200).map(Float.init)
        #expect(received == expected)
    }

    @Test("E0c: onStop preserves sample order within multi-sample buffers")
    func onStopOrdersExactSamplesAcrossMultiSampleBuffers() async throws {
        // E0/E0b script 200 single-sample buffers, so reversing a buffer in
        // place is a no-op and can't be caught by them. Here each buffer
        // carries 40 samples, so an in-place reversal is visible.
        let buffers = (0..<5).map { chunkIndex in
            (1...40).map { Float(chunkIndex * 40 + $0) }
        }
        let provider = MockAudioInput(buffers)

        actor RecordingDecoder {
            private(set) var received: [Float] = []
            func decode(_ samples: [Float]) async throws -> [TranscriptionSegment] {
                received = samples
                return []
            }
        }
        let recorder = RecordingDecoder()
        let engine = WhisperCppEngine(
            storage: WhisperModelStorage(defaults: UserDefaults(suiteName: UUID().uuidString)!),
            audioFactory: { provider },
            timing: .onStop,
            transcribeWindow: { samples in try await recorder.decode(samples) }
        )

        try await engine.start()
        await engine.stop()

        let received = await recorder.received
        let expected = (1...200).map(Float.init)
        #expect(received == expected)
    }

    @Test("E1: capture -> cutter -> decoder -> segment stream, during capture")
    func streamingEmitsDuringCapture() async throws {
        let buffer = Array(repeating: Float(0.5), count: 1_600)
        let buffers = Array(repeating: buffer, count: 9)
        let provider = MockAudioInput(buffers)
        let decoder = ScriptedWindowDecoder([.segments([segment("alpha", 0, 0.5)])])
        let policy = StreamingWindowPolicy(maximumWindowDuration: 10, minimumWindowDuration: 0.5, overlapDuration: 1)
        let engine = makeEngine(
            provider: provider,
            timing: .whileRecording(policy),
            decoder: decoder,
            verdicts: [true, true, true, true, true, true, true, true, false],
            policy: policy
        )

        let stream = engine.segmentStream()
        try await engine.start()

        let received = await firstSegment(from: stream, within: 5)
        #expect(received?.text == "alpha")
    }

    @Test("E2: false-positive — onStop must not stream")
    func onStopDoesNotStreamDuringCapture() async throws {
        let buffer = Array(repeating: Float(0.5), count: 1_600)
        let buffers = Array(repeating: buffer, count: 9)
        let provider = MockAudioInput(buffers)
        let decoder = ScriptedWindowDecoder([.segments([segment("alpha", 0, 0.5)])])
        let engine = makeEngine(
            provider: provider,
            timing: .onStop,
            decoder: decoder,
            verdicts: [true, true, true, true, true, true, true, true, false]
        )

        let stream = engine.segmentStream()
        try await engine.start()

        let receivedBeforeStop = await firstSegment(from: stream, within: 0.5)
        #expect(receivedBeforeStop == nil)

        await engine.stop()
        let callCount = await decoder.callCount
        #expect(callCount == 1)
        let received = await decoder.receivedSampleCounts
        #expect(received == [9 * 1_600])
    }

    @Test("E3: decode ordering under a slow decoder")
    func decodeOrderingUnderSlowDecoder() async throws {
        let buffer = Array(repeating: Float(0.5), count: 16_000)
        let buffers = [buffer, buffer]
        let provider = MockAudioInput(buffers)
        let decoder = ScriptedWindowDecoder([
            .slow([segment("one", 0, 0.5)], nanoseconds: 200_000_000),
            .segments([segment("two", 0, 0.5)]),
        ])
        let policy = StreamingWindowPolicy(maximumWindowDuration: 1.0, minimumWindowDuration: 0.5, overlapDuration: 0.25)
        let engine = makeEngine(
            provider: provider,
            timing: .whileRecording(policy),
            decoder: decoder,
            verdicts: [true, true],
            policy: policy
        )

        let collected = TextCollector()
        let stream = engine.segmentStream()
        let consumer = Task {
            for await seg in stream {
                await collected.append(seg.text)
            }
        }

        try await engine.start()
        await engine.stop()
        await consumer.value

        let texts = await collected.all
        #expect(texts == ["one", "two"])
    }

    @Test("E4: absolute times across windows")
    func absoluteTimesAcrossWindows() async throws {
        let buffer = Array(repeating: Float(0.5), count: 16_000)
        let buffers = [buffer, buffer]
        let provider = MockAudioInput(buffers)
        let decoder = ScriptedWindowDecoder([
            .segments([segment("first", 0.0, 0.4)]),
            .segments([segment("second", 0.0, 0.4)]),
        ])
        let policy = StreamingWindowPolicy(maximumWindowDuration: 1.0, minimumWindowDuration: 0.5, overlapDuration: 0.25)
        let engine = makeEngine(
            provider: provider,
            timing: .whileRecording(policy),
            decoder: decoder,
            verdicts: [true, true],
            policy: policy
        )

        let collected = SegmentListCollector()
        let stream = engine.segmentStream()
        let consumer = Task {
            for await seg in stream {
                await collected.append(seg)
            }
        }

        try await engine.start()
        await engine.stop()
        await consumer.value

        let segments = await collected.all
        #expect(segments == [segment("first", 0.0, 0.4), segment("second", 0.75, 1.15)])
    }

    @Test("E5: failure case — a decoder error does not kill the session")
    func decoderErrorDoesNotKillSession() async throws {
        let buffer = Array(repeating: Float(0.5), count: 16_000)
        let buffers = [buffer, buffer]
        let provider = MockAudioInput(buffers)
        let decoder = ScriptedWindowDecoder([
            .failure(.decoderFailure("boom")),
            .segments([segment("survived", 0, 0.5)]),
        ])
        let policy = StreamingWindowPolicy(maximumWindowDuration: 1.0, minimumWindowDuration: 0.5, overlapDuration: 0.25)
        let engine = makeEngine(
            provider: provider,
            timing: .whileRecording(policy),
            decoder: decoder,
            verdicts: [true, true],
            policy: policy
        )

        let collected = TextCollector()
        let stream = engine.segmentStream()
        let consumer = Task {
            for await seg in stream {
                await collected.append(seg.text)
            }
        }

        let statusStream = engine.statusStream()
        let statuses = StatusCollector()
        let statusConsumer = Task {
            for await status in statusStream {
                await statuses.append(status)
                if status == .ready { break }
            }
        }

        try await engine.start()
        await engine.stop()
        await consumer.value
        await statusConsumer.value

        let texts = await collected.all
        let statusList = await statuses.all
        #expect(texts == ["survived"])
        #expect(statusList.contains(.ready))
    }

    @Test("E6: stop drains the decode queue")
    func stopDrainsDecodeQueue() async throws {
        let buffer = Array(repeating: Float(0.5), count: 16_000)
        let buffers = [buffer, buffer, buffer]
        let provider = MockAudioInput(buffers)
        let decoder = ScriptedWindowDecoder([
            .slow([segment("one", 0, 0.5)], nanoseconds: 150_000_000),
            .slow([segment("two", 0, 0.5)], nanoseconds: 150_000_000),
            .slow([segment("three", 0, 0.5)], nanoseconds: 150_000_000),
        ])
        let policy = StreamingWindowPolicy(maximumWindowDuration: 1.0, minimumWindowDuration: 0.5, overlapDuration: 0.25)
        let engine = makeEngine(
            provider: provider,
            timing: .whileRecording(policy),
            decoder: decoder,
            verdicts: [true, true, true],
            policy: policy
        )

        let collected = TextCollector()
        let stream = engine.segmentStream()
        let consumer = Task {
            for await seg in stream {
                await collected.append(seg.text)
            }
        }

        try await engine.start()
        await engine.stop()
        await consumer.value

        let texts = await collected.all
        #expect(Set(texts) == Set(["one", "two", "three"]))
    }

    @Test("E7: final partial window is flushed on stop")
    func finalPartialWindowFlushedOnStop() async throws {
        let buffer = Array(repeating: Float(0.5), count: 1_600)
        let buffers = [buffer, buffer, buffer]
        let provider = MockAudioInput(buffers)
        let decoder = ScriptedWindowDecoder([.segments([segment("tail", 0, 0.3)])])
        let policy = StreamingWindowPolicy(maximumWindowDuration: 10, minimumWindowDuration: 0.5, overlapDuration: 1)
        let engine = makeEngine(
            provider: provider,
            timing: .whileRecording(policy),
            decoder: decoder,
            verdicts: [true, true, true],
            policy: policy
        )

        let stream = engine.segmentStream()
        let collected = SegmentCollector()
        let consumer = Task {
            for await seg in stream {
                await collected.append(seg)
            }
        }

        try await engine.start()
        let duringCapture = await firstSegment(from: engine.segmentStream(), within: 0.3)
        #expect(duringCapture == nil)

        await engine.stop()
        await consumer.value

        let callCount = await decoder.callCount
        #expect(callCount == 1)
        let texts = await collected.texts
        #expect(texts == ["tail"])
    }

    @Test("E8: stop stays idempotent under streaming")
    func stopIsIdempotentUnderStreaming() async throws {
        let buffer = Array(repeating: Float(0.5), count: 1_600)
        let buffers = [buffer, buffer, buffer]
        let provider = MockAudioInput(buffers)
        let decoder = ScriptedWindowDecoder([.segments([segment("tail", 0, 0.3)])])
        let policy = StreamingWindowPolicy(maximumWindowDuration: 10, minimumWindowDuration: 0.5, overlapDuration: 1)
        let engine = makeEngine(
            provider: provider,
            timing: .whileRecording(policy),
            decoder: decoder,
            verdicts: [true, true, true],
            policy: policy
        )

        try await engine.start()
        await engine.stop()
        let countAfterFirstStop = await decoder.callCount
        await engine.stop()
        let countAfterSecondStop = await decoder.callCount

        #expect(countAfterFirstStop == countAfterSecondStop)
    }

    @Test("E9: status stream reaches .listening then .ready in both modes")
    func statusStreamUnchangedInBothModes() async throws {
        for timing: TranscriptionTiming in [.onStop, .whileRecording(.default)] {
            let buffer = Array(repeating: Float(0.5), count: 1_600)
            let provider = MockAudioInput([buffer])
            let decoder = ScriptedWindowDecoder([.segments([])])
            let engine = makeEngine(provider: provider, timing: timing, decoder: decoder, verdicts: [true])

            let statusStream = engine.statusStream()
            let statuses = StatusCollector()
            let statusConsumer = Task {
                for await status in statusStream {
                    await statuses.append(status)
                    if status == .ready { break }
                }
            }

            try await engine.start()
            await engine.stop()
            await statusConsumer.value

            let statusList = await statuses.all
            #expect(statusList.contains(.listening))
            #expect(statusList.contains(.ready))
        }
    }

    @Test("E10: flagship — no loss and no duplication across the actor seam")
    func noLossNoDuplicationAcrossActorSeam() async throws {
        let buffer = Array(repeating: Float(0.5), count: 1_600)
        let buffers = Array(repeating: buffer, count: 13)
        let provider = MockAudioInput(buffers)
        let decoder = ScriptedWindowDecoder([
            .segments([segment("alpha", 0.0, 0.6), segment("beta", 0.6, 0.9)]),
            .segments([segment("beta", 0.0, 0.1), segment("gamma", 0.1, 0.2)]),
        ])
        let policy = StreamingWindowPolicy(maximumWindowDuration: 1.0, minimumWindowDuration: 0.5, overlapDuration: 0.25)
        let engine = makeEngine(
            provider: provider,
            timing: .whileRecording(policy),
            decoder: decoder,
            verdicts: Array(repeating: true, count: 13),
            policy: policy
        )

        let collected = SegmentListCollector()
        let stream = engine.segmentStream()
        let consumer = Task {
            for await seg in stream {
                await collected.append(seg)
            }
        }

        try await engine.start()
        await engine.stop()
        await consumer.value

        let segments = await collected.all
        #expect(
            segments == [
                segment("alpha", 0.0, 0.6),
                segment("beta", 0.6, 0.9),
                segment("gamma", 0.85, 0.95),
            ])
    }
}

private actor SegmentCollector {
    private var segments: [TranscriptionSegment] = []
    func append(_ segment: TranscriptionSegment) { segments.append(segment) }
    var texts: [String] { segments.map(\.text) }
}

private actor TextCollector {
    private var items: [String] = []
    func append(_ item: String) { items.append(item) }
    var all: [String] { items }
}

private actor SegmentListCollector {
    private var items: [TranscriptionSegment] = []
    func append(_ item: TranscriptionSegment) { items.append(item) }
    var all: [TranscriptionSegment] { items }
}

private actor StatusCollector {
    private var items: [WhisperEngineStatus] = []
    func append(_ item: WhisperEngineStatus) { items.append(item) }
    var all: [WhisperEngineStatus] { items }
}
