@preconcurrency import CoreML
import Foundation
import SwiftSTTCore
import Synchronization
import Testing

@testable import SwiftSTTKit

// MARK: - Mock stateful runner that records inputs

private final class RecordingStatefulRunner: StatefulCoreMLModelRunner, @unchecked Sendable {

    struct Recording: Sendable {
        var resetCount: Int = 0
        var predictCount: Int = 0
        var lastStateFirstValue: Float = -1
        var lastAudioSampleCount: Int = 0
    }

    private let probabilities: Mutex<[Float]>
    private let recording: Mutex<Recording>
    private let stateOutputName: String
    private let probabilityOutputName: String
    private let audioInputName: String
    private let stateInputName: String
    /// When non-nil, predict throws this error.
    private let failure: SwiftSTTError?

    init(
        probabilities: [Float] = [0.9],
        featureNames: SileroVAD.FeatureNames = .default,
        failure: SwiftSTTError? = nil
    ) {
        self.probabilities = Mutex(probabilities)
        self.recording = Mutex(Recording())
        self.stateOutputName = featureNames.stateOutput
        self.probabilityOutputName = featureNames.probabilityOutput
        self.audioInputName = featureNames.audioInput
        self.stateInputName = featureNames.stateInput
        self.failure = failure
    }

    var resetCount: Int { recording.withLock { $0.resetCount } }
    var predictCount: Int { recording.withLock { $0.predictCount } }
    var lastStateFirstValue: Float { recording.withLock { $0.lastStateFirstValue } }
    var lastAudioSampleCount: Int { recording.withLock { $0.lastAudioSampleCount } }

    func resetState() async {
        recording.withLock { $0.resetCount += 1 }
    }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftSTTError) -> any MLFeatureProvider {
        recording.withLock { $0.predictCount += 1 }
        if let stateValue = features.featureValue(for: stateInputName),
            let stateArray = stateValue.multiArrayValue,
            stateArray.count > 0
        {
            let v = stateArray[0].floatValue
            recording.withLock { $0.lastStateFirstValue = v }
        }
        if let audioValue = features.featureValue(for: audioInputName),
            let audioArray = audioValue.multiArrayValue
        {
            let count = audioArray.count
            recording.withLock { $0.lastAudioSampleCount = count }
        }

        if let failure {
            throw failure
        }

        let prob: Float = probabilities.withLock { queue in
            if queue.isEmpty { return 0 }
            return queue.removeFirst()
        }

        let probArray: MLMultiArray
        let nextState: MLMultiArray
        do {
            probArray = try MLMultiArray(shape: [1], dataType: .float32)
            probArray.dataPointer
                .bindMemory(to: Float.self, capacity: 1)
                .pointee = prob

            nextState = try MLMultiArray(shape: SileroVAD.stateShape, dataType: .float32)
            let count = SileroVAD.stateShape.reduce(1) { $0 * $1.intValue }
            let pointer = nextState.dataPointer.bindMemory(to: Float.self, capacity: count)
            // Distinguish "next state" from "zeros" so tests can verify the
            // actor stored it for the next call.
            pointer[0] = 0.42
            for i in 1..<count { pointer[i] = 0 }
        } catch {
            throw .decoderFailure("mock array: \(error.localizedDescription)")
        }

        do {
            return try MLDictionaryFeatureProvider(dictionary: [
                probabilityOutputName: probArray,
                stateOutputName: nextState,
            ])
        } catch {
            throw .decoderFailure("mock provider: \(error.localizedDescription)")
        }
    }
}

@Suite("SileroVAD")
struct SileroVADTests {

    // MARK: - Helpers

    static func chunk(samples: [Float]) -> AudioChunk {
        AudioChunk(samples: samples, sampleRate: 16_000, timestamp: 0)
    }

    static func zeros(_ n: Int) -> [Float] {
        Array(repeating: 0, count: n)
    }

    // MARK: - Tests

    @Test("Below-threshold probability returns false")
    func belowThresholdReturnsFalse() async {
        let runner = RecordingStatefulRunner(probabilities: [0.2])
        let vad = SileroVAD(runner: runner, threshold: 0.5)
        let result = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(512)))
        #expect(result == false)
    }

    @Test("Above-threshold probability returns true")
    func aboveThresholdReturnsTrue() async {
        let runner = RecordingStatefulRunner(probabilities: [0.9])
        let vad = SileroVAD(runner: runner, threshold: 0.5)
        let result = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(512)))
        #expect(result == true)
    }

    @Test("State updates between calls (mock observes carried state)")
    func stateCarriesBetweenCalls() async {
        let runner = RecordingStatefulRunner(probabilities: [0.9, 0.9])
        let vad = SileroVAD(runner: runner, threshold: 0.5)

        _ = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(512)))
        // First call's state should be zeros.
        #expect(runner.lastStateFirstValue == 0)

        _ = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(512)))
        // Second call's state input should equal the first call's state output.
        #expect(runner.lastStateFirstValue == 0.42)
    }

    @Test("reset() clears carried state")
    func resetClearsState() async {
        let runner = RecordingStatefulRunner(probabilities: [0.9, 0.9])
        let vad = SileroVAD(runner: runner, threshold: 0.5)

        _ = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(512)))
        await vad.reset()

        _ = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(512)))
        #expect(runner.lastStateFirstValue == 0)
        #expect(runner.resetCount == 1)
    }

    @Test("Chunk smaller than window pads with zeros and runs once")
    func smallChunkPadsToWindow() async {
        let runner = RecordingStatefulRunner(probabilities: [0.9])
        let vad = SileroVAD(runner: runner, threshold: 0.5)

        let result = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(100)))
        #expect(result == true)
        #expect(runner.predictCount == 1)
        #expect(runner.lastAudioSampleCount == 512)
    }

    @Test("Chunk larger than window runs multiple sliding windows")
    func largeChunkRunsMultipleWindows() async {
        let runner = RecordingStatefulRunner(probabilities: [0.1, 0.1, 0.9])
        let vad = SileroVAD(runner: runner, threshold: 0.5)

        // 1500 samples = 3 windows of 512 (last padded).
        let result = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(1500)))
        #expect(result == true)
        #expect(runner.predictCount == 3)
    }

    @Test("Threshold is configurable")
    func thresholdConfigurable() async {
        let lowRunner = RecordingStatefulRunner(probabilities: [0.3])
        let lowVad = SileroVAD(runner: lowRunner, threshold: 0.2)
        #expect(await lowVad.isSpeech(chunk: Self.chunk(samples: Self.zeros(512))) == true)

        let highRunner = RecordingStatefulRunner(probabilities: [0.3])
        let highVad = SileroVAD(runner: highRunner, threshold: 0.8)
        #expect(await highVad.isSpeech(chunk: Self.chunk(samples: Self.zeros(512))) == false)
    }

    @Test("Custom feature names route correctly")
    func customFeatureNames() async {
        let names = SileroVAD.FeatureNames(
            audioInput: "audio_in",
            stateInput: "state_in",
            probabilityOutput: "prob_out",
            stateOutput: "state_out"
        )
        let runner = RecordingStatefulRunner(probabilities: [0.9], featureNames: names)
        let vad = SileroVAD(runner: runner, threshold: 0.5, featureNames: names)
        let result = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(512)))
        #expect(result == true)
        #expect(runner.predictCount == 1)
    }

    @Test("Runner failure produces false (fail-open)")
    func runnerFailureReturnsFalse() async {
        let runner = RecordingStatefulRunner(failure: .decoderFailure("oops"))
        let vad = SileroVAD(runner: runner, threshold: 0.5)
        let result = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(512)))
        #expect(result == false)
    }

    @Test("Empty chunk returns false")
    func emptyChunkReturnsFalse() async {
        let runner = RecordingStatefulRunner(probabilities: [0.9])
        let vad = SileroVAD(runner: runner, threshold: 0.5)
        let result = await vad.isSpeech(chunk: Self.chunk(samples: []))
        #expect(result == false)
        #expect(runner.predictCount == 0)
    }

    @Test("Large chunk takes max probability across windows (any speech triggers)")
    func largeChunkUsesMaxAcrossWindows() async {
        let runner = RecordingStatefulRunner(probabilities: [0.9, 0.1])
        let vad = SileroVAD(runner: runner, threshold: 0.5)
        let result = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(1024)))
        #expect(result == true)
    }

    @Test("All-silent windows produce false even on a long chunk")
    func longSilentChunkReturnsFalse() async {
        let runner = RecordingStatefulRunner(probabilities: [0.1, 0.1, 0.1])
        let vad = SileroVAD(runner: runner, threshold: 0.5)
        let result = await vad.isSpeech(chunk: Self.chunk(samples: Self.zeros(1500)))
        #expect(result == false)
    }

    @Test("splitIntoWindows: smaller than window returns one padded window")
    func splitSmallerWindow() {
        let windows = SileroVAD.splitIntoWindows(
            Array(repeating: Float(1), count: 100), windowSize: 512)
        #expect(windows.count == 1)
        #expect(windows[0].count == 512)
        // First 100 should be ones, the rest zeros.
        #expect(windows[0][0] == 1)
        #expect(windows[0][99] == 1)
        #expect(windows[0][100] == 0)
    }

    @Test("splitIntoWindows: exact multiple produces non-overlapping windows")
    func splitExactMultiple() {
        let windows = SileroVAD.splitIntoWindows(
            Array(repeating: Float(0), count: 1024), windowSize: 512)
        #expect(windows.count == 2)
        #expect(windows[0].count == 512)
        #expect(windows[1].count == 512)
    }

    @Test("splitIntoWindows: tail is padded")
    func splitTailPadded() {
        let windows = SileroVAD.splitIntoWindows(
            Array(repeating: Float(1), count: 700), windowSize: 512)
        #expect(windows.count == 2)
        #expect(windows[1][0] == 1)
        #expect(windows[1][187] == 1)
        #expect(windows[1][188] == 0)
    }

    @Test("splitIntoWindows: zero window size returns empty")
    func splitZeroWindow() {
        let windows = SileroVAD.splitIntoWindows([1, 2, 3], windowSize: 0)
        #expect(windows.isEmpty)
    }

    @Test("load(from:) throws when file does not exist")
    func loadMissingFile() async {
        let url = URL(fileURLWithPath: "/tmp/nonexistent_silero_\(UUID().uuidString).mlmodelc")
        do {
            _ = try await SileroVAD.load(from: url)
            Issue.record("Expected modelLoadFailed")
        } catch {
            switch error {
            case .modelLoadFailed:
                break
            default:
                Issue.record("Expected modelLoadFailed, got \(error)")
            }
        }
    }
}
