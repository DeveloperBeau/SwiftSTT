@preconcurrency import CoreML
import Foundation
import SwiftSTTCore

/// Neural voice activity detector backed by a Silero VAD Core ML model.
///
/// Silero VAD is a small recurrent network trained on a wide range of
/// background noise. It outperforms ``EnergyVAD`` in conditions where energy
/// alone gives false positives: music, fan noise, traffic, and conversations
/// in the background.
///
/// ## Model
///
/// The Silero VAD models live at <https://github.com/snakers4/silero-vad>.
/// The Core ML conversion is not maintained upstream; convert the ONNX model
/// using `coremltools` and provide the compiled `.mlmodelc` to
/// ``load(from:threshold:)``.
///
/// ## Window contract
///
/// - 16 kHz mono Float32 input
/// - 512-sample window (32 ms)
/// - Recurrent state held across windows. Cleared by ``reset()``
/// - Output: a single Float in `[0, 1]` interpreted as the probability of
///   speech in the window
///
/// Chunks shorter than 512 samples are zero-padded to a single window. Chunks
/// longer than 512 samples are split into non-overlapping 512-sample windows
/// and the maximum probability across the windows is compared against the
/// threshold so brief speech inside a longer chunk still triggers.
///
/// ## Failure mode
///
/// If the underlying Core ML runner throws, ``isSpeech(chunk:)`` returns
/// `false` rather than propagating. The contract on ``VoiceActivityDetector``
/// is non-throwing and a single transient model error should not poison the
/// rest of the audio stream. Persistent failures will manifest as the VAD
/// permanently rejecting speech, which is preferable to crashing the
/// pipeline.
public actor SileroVAD: VoiceActivityDetector {

    /// Names of the model's input and output features.
    ///
    /// Default values are a best guess for community Core ML conversions of Silero v4 / v5;
    /// override at construction time for a model that uses different names.
    public struct FeatureNames: Sendable, Equatable {
        /// Name of the audio input feature.
        public var audioInput: String
        /// Name of the recurrent-state input feature.
        public var stateInput: String
        /// Name of the probability output feature.
        public var probabilityOutput: String
        /// Name of the recurrent-state output feature.
        public var stateOutput: String

        /// Creates a new FeatureNames with the supplied values.
        public init(
            audioInput: String,
            stateInput: String,
            probabilityOutput: String,
            stateOutput: String
        ) {
            self.audioInput = audioInput
            self.stateInput = stateInput
            self.probabilityOutput = probabilityOutput
            self.stateOutput = stateOutput
        }

        /// Default feature names commonly used by community Silero v4/v5 Core ML conversions.
        public static let `default` = FeatureNames(
            audioInput: "input",
            stateInput: "state",
            probabilityOutput: "output",
            stateOutput: "stateN"
        )
    }

    /// Number of samples per Silero window at 16 kHz. 32 ms.
    public static let defaultWindowSamples: Int = 512

    /// Recurrent state size for Silero v4.
    ///
    /// The model expects a `[2, 1, 64]` tensor that the model both reads and writes. The actor holds zeros at
    /// startup and after ``reset()``.
    public static let stateShape: [NSNumber] = [2, 1, 64]

    private let runner: any StatefulCoreMLModelRunner
    private let threshold: Float
    private let windowSamples: Int
    private let sampleRate: Int
    private let featureNames: FeatureNames

    /// Carried recurrent state. `nil` means "use zeros on the next predict".
    private var state: MLMultiArray?

    /// Creates a new SileroVAD with the supplied values.
    public init(
        runner: any StatefulCoreMLModelRunner,
        threshold: Float = 0.5,
        windowSamples: Int = SileroVAD.defaultWindowSamples,
        sampleRate: Int = 16_000,
        featureNames: FeatureNames = .default
    ) {
        self.runner = runner
        self.threshold = threshold
        self.windowSamples = windowSamples
        self.sampleRate = sampleRate
        self.featureNames = featureNames
        self.state = nil
    }

    /// Returns whether the chunk contains speech.
    public func isSpeech(chunk: AudioChunk) async -> Bool {
        guard !chunk.samples.isEmpty else { return false }

        let windows = Self.splitIntoWindows(chunk.samples, windowSize: windowSamples)
        var maxProb: Float = 0
        for window in windows {
            let prob = await runWindow(window)
            if prob > maxProb { maxProb = prob }
        }
        return maxProb >= threshold
    }

    /// Resets the actor's state.
    public func reset() async {
        state = nil
        await runner.resetState()
    }

    // MARK: - Window evaluation

    /// Splits `samples` into non-overlapping windows of size `windowSize`.
    ///
    /// The final window is zero-padded to `windowSize`. A buffer shorter than
    /// `windowSize` becomes a single padded window so partial chunks still get
    /// classified.
    static func splitIntoWindows(_ samples: [Float], windowSize: Int) -> [[Float]] {
        guard windowSize > 0 else { return [] }
        if samples.count <= windowSize {
            var window = samples
            window.append(contentsOf: Array(repeating: 0, count: windowSize - samples.count))
            return [window]
        }
        var result: [[Float]] = []
        var pos = 0
        while pos < samples.count {
            let end = min(pos + windowSize, samples.count)
            var window = Array(samples[pos..<end])
            if window.count < windowSize {
                window.append(contentsOf: Array(repeating: 0, count: windowSize - window.count))
            }
            result.append(window)
            pos += windowSize
        }
        return result
    }

    private func runWindow(_ samples: [Float]) async -> Float {
        let inputArray: MLMultiArray
        do {
            inputArray = try Self.makeInputArray(samples: samples)
        } catch {
            return 0
        }

        let stateArray: MLMultiArray
        if let existing = state {
            stateArray = existing
        } else {
            do {
                stateArray = try Self.makeZeroState()
            } catch {
                return 0
            }
        }

        let provider: any MLFeatureProvider
        do {
            provider = try MLDictionaryFeatureProvider(dictionary: [
                featureNames.audioInput: inputArray,
                featureNames.stateInput: stateArray,
            ])
        } catch {
            return 0
        }

        let output: any MLFeatureProvider
        do {
            output = try await runner.predict(features: provider)
        } catch {
            return 0
        }

        if let nextStateValue = output.featureValue(for: featureNames.stateOutput),
            let nextState = nextStateValue.multiArrayValue
        {
            state = nextState
        }

        guard let probValue = output.featureValue(for: featureNames.probabilityOutput),
            let probArray = probValue.multiArrayValue,
            probArray.count > 0
        else {
            return 0
        }
        return probArray[0].floatValue
    }

    // MARK: - Tensor helpers

    static func makeInputArray(samples: [Float]) throws(SwiftSTTError) -> MLMultiArray {
        let array: MLMultiArray
        do {
            array = try MLMultiArray(
                shape: [1, NSNumber(value: samples.count)],
                dataType: .float32
            )
        } catch {
            throw .modelLoadFailed("Silero input MLMultiArray: \(error.localizedDescription)")
        }
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: samples.count)
        for i in 0..<samples.count {
            pointer[i] = samples[i]
        }
        return array
    }

    static func makeZeroState() throws(SwiftSTTError) -> MLMultiArray {
        let array: MLMultiArray
        do {
            array = try MLMultiArray(shape: stateShape, dataType: .float32)
        } catch {
            throw .modelLoadFailed("Silero state MLMultiArray: \(error.localizedDescription)")
        }
        let count = stateShape.reduce(1) { $0 * $1.intValue }
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            pointer[i] = 0
        }
        return array
    }
}

// MARK: - Disk loading

extension SileroVAD {

    /// Loads a compiled Silero VAD model from disk and wraps it in a
    /// ``SileroVAD`` actor.
    ///
    /// - Parameters:
    ///   - url: directory URL pointing at a compiled `.mlmodelc` package.
    ///   - threshold: probability threshold for speech.
    /// - Returns: a configured ``SileroVAD`` ready to score audio frames.
    /// - Throws: ``SwiftSTTError/modelLoadFailed(_:)`` if the file is
    ///   missing or Core ML rejects the model.
    public static func load(
        from url: URL,
        threshold: Float = 0.5
    ) async throws(SwiftSTTError) -> SileroVAD {
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw .modelLoadFailed("Silero VAD not found at \(url.path)")
        }
        let config = MLModelConfiguration()
        config.computeUnits = .all
        let model: MLModel
        do {
            model = try await MLModel.load(contentsOf: url, configuration: config)
        } catch {
            throw .modelLoadFailed("Silero VAD: \(error.localizedDescription)")
        }
        let runner = MLStateModelRunner(model: model)
        await runner.resetState()
        return SileroVAD(runner: runner, threshold: threshold)
    }
}
