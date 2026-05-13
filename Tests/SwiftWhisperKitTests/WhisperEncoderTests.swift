@preconcurrency import CoreML
import Foundation
import SwiftWhisperCore
import Synchronization
import Testing

@testable import SwiftWhisperKit

// MARK: - Mock runner

private final class MockRunner: CoreMLModelRunner, @unchecked Sendable {

    enum Behaviour {
        case returnArray(MLMultiArray)
        case throwError(SwiftWhisperError)
    }

    private struct State {
        var callCount: Int = 0
        var lastInputArray: MLMultiArray?
    }

    private let behaviour: Behaviour
    private let state = Mutex<State>(State())

    init(_ behaviour: Behaviour) {
        self.behaviour = behaviour
    }

    var callCount: Int { state.withLock { $0.callCount } }
    var lastInputArray: MLMultiArray? { state.withLock { $0.lastInputArray } }

    func predict(
        features: any MLFeatureProvider
    ) async throws(SwiftWhisperError) -> any MLFeatureProvider {
        let array = features.featureValue(for: WhisperEncoder.inputFeatureName)?.multiArrayValue
        state.withLock {
            $0.callCount += 1
            $0.lastInputArray = array
        }

        switch behaviour {
        case .throwError(let err):
            throw err
        case .returnArray(let array):
            do {
                return try MLDictionaryFeatureProvider(dictionary: [
                    WhisperEncoder.outputFeatureName: array
                ])
            } catch {
                throw .decoderFailure("mock provider: \(error.localizedDescription)")
            }
        }
    }
}

// MARK: - Helpers

private func makeMel(nMels: Int, nFrames: Int, fill: Float = 0.5) throws -> MelSpectrogramResult {
    try MelSpectrogramResult(
        frames: [Float](repeating: fill, count: nMels * nFrames),
        nMels: nMels,
        nFrames: nFrames
    )
}

private func makeOutputArray(shape: [Int], fill: Float = 1.0) throws -> MLMultiArray {
    let array = try MLMultiArray(
        shape: shape.map { NSNumber(value: $0) },
        dataType: .float32
    )
    let count = shape.reduce(1, *)
    let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
    for i in 0..<count {
        ptr[i] = fill
    }
    return array
}

// MARK: - Tests

@Suite("WhisperEncoder")
struct WhisperEncoderTests {

    // MARK: padOrTrim

    @Test("padOrTrim pads short input with zeros, preserves data")
    func padOrTrimPads() throws {
        let nMels = 4
        let inFrames = 3
        let outFrames = 5

        // Build mel where each cell has a unique value so we can verify placement.
        var raw = [Float](repeating: 0, count: nMels * inFrames)
        for m in 0..<nMels {
            for t in 0..<inFrames {
                raw[m * inFrames + t] = Float(m * 100 + t)
            }
        }
        let input = try MelSpectrogramResult(frames: raw, nMels: nMels, nFrames: inFrames)
        let out = try WhisperEncoder.padOrTrim(input, toFrames: outFrames)

        #expect(out.nFrames == outFrames)
        #expect(out.nMels == nMels)
        for m in 0..<nMels {
            for t in 0..<inFrames {
                #expect(out.frames[m * outFrames + t] == Float(m * 100 + t))
            }
            // Padding region should be zero.
            for t in inFrames..<outFrames {
                #expect(out.frames[m * outFrames + t] == 0)
            }
        }
    }

    @Test("padOrTrim trims long input")
    func padOrTrimTrims() throws {
        let nMels = 3
        let inFrames = 10
        let outFrames = 4

        var raw = [Float](repeating: 0, count: nMels * inFrames)
        for m in 0..<nMels {
            for t in 0..<inFrames {
                raw[m * inFrames + t] = Float(m * 1000 + t)
            }
        }
        let input = try MelSpectrogramResult(frames: raw, nMels: nMels, nFrames: inFrames)
        let out = try WhisperEncoder.padOrTrim(input, toFrames: outFrames)

        #expect(out.nFrames == outFrames)
        for m in 0..<nMels {
            for t in 0..<outFrames {
                #expect(out.frames[m * outFrames + t] == Float(m * 1000 + t))
            }
        }
    }

    @Test("padOrTrim with exact length returns same data")
    func padOrTrimExact() throws {
        let nMels = 2
        let frames = 5
        let raw = (0..<(nMels * frames)).map { Float($0) }
        let input = try MelSpectrogramResult(frames: raw, nMels: nMels, nFrames: frames)
        let out = try WhisperEncoder.padOrTrim(input, toFrames: frames)
        #expect(out.frames == raw)
        #expect(out.nFrames == frames)
        #expect(out.nMels == nMels)
    }

    // MARK: buildFeatureProvider

    @Test("buildFeatureProvider produces correct shape and values")
    func buildFeatureProviderShape() throws {
        let nMels = 4
        let nFrames = 6
        let raw = (0..<(nMels * nFrames)).map { Float($0) * 0.25 }
        let mel = try MelSpectrogramResult(frames: raw, nMels: nMels, nFrames: nFrames)

        let provider = try WhisperEncoder.buildFeatureProvider(mel: mel)
        let value = provider.featureValue(for: WhisperEncoder.inputFeatureName)
        #expect(value != nil)
        let array = try #require(value?.multiArrayValue)
        #expect(array.shape == [1, NSNumber(value: nMels), 1, NSNumber(value: nFrames)])
        #expect(array.dataType == .float32)

        let count = nMels * nFrames
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: count)
        for i in 0..<count {
            #expect(ptr[i] == raw[i])
        }
    }

    @Test("buildFeatureProvider exposes only the input feature")
    func buildFeatureProviderFeatureName() throws {
        let mel = try makeMel(nMels: 2, nFrames: 3)
        let provider = try WhisperEncoder.buildFeatureProvider(mel: mel)
        #expect(provider.featureNames.contains(WhisperEncoder.inputFeatureName))
    }

    // MARK: extractEmbeddings

    @Test("extractEmbeddings returns the output array when present")
    func extractEmbeddingsHappy() throws {
        let array = try makeOutputArray(shape: [1, 1500, 384])
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            WhisperEncoder.outputFeatureName: array
        ])
        let result = try WhisperEncoder.extractEmbeddings(from: provider)
        #expect(result.shape == [1, 1500, 384])
    }

    @Test("extractEmbeddings throws decoderFailure when feature missing")
    func extractEmbeddingsMissing() throws {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "wrong_name": try makeOutputArray(shape: [1, 1, 1])
        ])
        do {
            _ = try WhisperEncoder.extractEmbeddings(from: provider)
            Issue.record("expected throw")
        } catch {
            if case .decoderFailure = error {
            } else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    @Test("extractEmbeddings throws decoderFailure when feature is not multi-array")
    func extractEmbeddingsWrongType() throws {
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            WhisperEncoder.outputFeatureName: MLFeatureValue(string: "not an array")
        ])
        do {
            _ = try WhisperEncoder.extractEmbeddings(from: provider)
            Issue.record("expected throw")
        } catch {
            if case .decoderFailure = error {
            } else {
                Issue.record("wrong error: \(error)")
            }
        }
    }

    // MARK: encode end-to-end

    @Test("encode forwards prepared input to runner under correct feature name")
    func encodeForwardsToRunner() async throws {
        let cannedOutput = try makeOutputArray(shape: [1, 1500, 384], fill: 7.0)
        let runner = MockRunner(.returnArray(cannedOutput))
        let encoder = WhisperEncoder(runner: runner)

        let mel = try makeMel(nMels: 80, nFrames: 100, fill: 0.25)
        let result = try await encoder.encode(spectrogram: mel)

        #expect(runner.callCount == 1)

        let received = try #require(runner.lastInputArray)
        #expect(received.shape == [1, 80, 1, NSNumber(value: WhisperEncoder.expectedFrames)])

        // Values from the original 100 frames should be present at the start;
        // the rest should be zero-padded.
        let receivedPtr = received.dataPointer.bindMemory(
            to: Float.self,
            capacity: 80 * WhisperEncoder.expectedFrames
        )
        // Spot-check the first frame across all mel bands.
        for m in 0..<80 {
            #expect(receivedPtr[m * WhisperEncoder.expectedFrames] == 0.25)
        }
        // And confirm padding region is zero.
        for m in 0..<80 {
            #expect(
                receivedPtr[m * WhisperEncoder.expectedFrames + (WhisperEncoder.expectedFrames - 1)]
                    == 0)
        }

        // Returned array should be the canned output.
        #expect(result.shape == cannedOutput.shape)
    }

    @Test("encode propagates runner errors")
    func encodeRunnerError() async throws {
        let runner = MockRunner(.throwError(.decoderFailure("boom")))
        let encoder = WhisperEncoder(runner: runner)
        let mel = try makeMel(nMels: 80, nFrames: 100)

        do {
            _ = try await encoder.encode(spectrogram: mel)
            Issue.record("expected throw")
        } catch {
            #expect(error == .decoderFailure("boom"))
        }
    }
}
