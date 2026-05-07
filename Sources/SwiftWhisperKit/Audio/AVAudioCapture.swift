@preconcurrency import AVFoundation
import Foundation
import SwiftWhisperCore

/// Microphone capture via `AVAudioEngine` with on-the-fly downsampling to 16 kHz mono Float32.
///
/// Emits `AudioChunk` values on `audioStream`. Cleanup runs automatically when the stream
/// consumer cancels (via `onTermination`) or when `stopCapture()` is called explicitly.
///
/// Platform notes:
/// - iOS: configures `AVAudioSession` and observes interruption notifications.
/// - macOS: no audio session; the engine drives the input node directly. CLI/host app
///   must declare `NSMicrophoneUsageDescription` and (if sandboxed) the
///   `com.apple.security.device.audio-input` entitlement.
public actor AVAudioCapture: AudioCapturer {

    public nonisolated let audioStream: AsyncStream<AudioChunk>
    private nonisolated let continuation: AsyncStream<AudioChunk>.Continuation
    private let box: CaptureBox

    private let targetSampleRate: Double
    private let bufferDurationSeconds: Double
    private var isCapturing: Bool = false
    private var startTimestamp: TimeInterval = 0

    #if os(iOS)
    private var interruptionObserver: NSObjectProtocol?
    #endif

    public init(targetSampleRate: Double = 16_000, bufferDurationSeconds: Double = 0.064) {
        self.targetSampleRate = targetSampleRate
        self.bufferDurationSeconds = bufferDurationSeconds

        let (stream, continuation) = AsyncStream<AudioChunk>.makeStream()
        self.audioStream = stream
        self.continuation = continuation
        self.box = CaptureBox()

        // Capture cleanup state in the box; closure cannot reach actor-isolated state.
        let box = self.box
        continuation.onTermination = { @Sendable _ in
            box.teardown()
        }
    }

    public func startCapture() async throws(SwiftWhisperError) {
        guard !isCapturing else { return }

        let granted = await Self.requestPermission()
        guard granted else { throw .micPermissionDenied }

        #if os(iOS)
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: [])
            try session.setActive(true)
        } catch {
            throw .audioCaptureFailed("audio session: \(error.localizedDescription)")
        }
        installInterruptionObserver()
        #endif

        let inputFormat = box.engine.inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0 else {
            throw .audioCaptureFailed("input format has zero sample rate; no microphone?")
        }

        guard
            let outputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: targetSampleRate,
                channels: 1,
                interleaved: false
            ),
            let converter = AVAudioConverter(from: inputFormat, to: outputFormat)
        else {
            throw .audioConversionFailed
        }
        box.converter = converter
        box.outputFormat = outputFormat

        let bufferSize = AVAudioFrameCount(max(256, inputFormat.sampleRate * bufferDurationSeconds))
        let cont = continuation
        let target = targetSampleRate
        let inputRate = inputFormat.sampleRate
        let started = Date().timeIntervalSince1970
        let box = self.box
        startTimestamp = started

        box.engine.inputNode.installTap(
            onBus: 0,
            bufferSize: bufferSize,
            format: inputFormat
        ) { @Sendable buffer, _ in
            guard
                let convertedSamples = box.convert(buffer: buffer, sourceRate: inputRate, targetRate: target)
            else { return }
            let timestamp = Date().timeIntervalSince1970 - started
            cont.yield(AudioChunk(samples: convertedSamples, sampleRate: Int(target), timestamp: timestamp))
        }
        box.tapInstalled = true

        do {
            box.engine.prepare()
            try box.engine.start()
            isCapturing = true
        } catch {
            box.teardown()
            throw .audioCaptureFailed("engine start: \(error.localizedDescription)")
        }
    }

    public func stopCapture() async {
        guard isCapturing else { return }
        box.teardown()
        isCapturing = false

        #if os(iOS)
        if let observer = interruptionObserver {
            NotificationCenter.default.removeObserver(observer)
            interruptionObserver = nil
        }
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        #endif

        continuation.finish()
    }

    // MARK: - Permission

    private static func requestPermission() async -> Bool {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        switch status {
        case .authorized:
            return true
        case .notDetermined:
            return await AVCaptureDevice.requestAccess(for: .audio)
        case .denied, .restricted:
            return false
        @unknown default:
            return false
        }
    }

    // MARK: - Interruption (iOS only)

    #if os(iOS)
    private func installInterruptionObserver() {
        let box = self.box
        interruptionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: nil,
            queue: nil
        ) { notification in
            guard
                let info = notification.userInfo,
                let raw = info[AVAudioSessionInterruptionTypeKey] as? UInt,
                let type = AVAudioSession.InterruptionType(rawValue: raw)
            else { return }
            switch type {
            case .began:
                box.engine.pause()
            case .ended:
                try? box.engine.start()
            @unknown default:
                break
            }
        }
    }
    #endif
}

// MARK: - CaptureBox

/// Holds non-Sendable AVFoundation references behind a Sendable boundary.
/// Mutations are serialized: writes only happen from the actor before the tap is
/// installed; reads happen from the audio thread inside the tap callback or from
/// `teardown()` under the actor's serial isolation.
private final class CaptureBox: @unchecked Sendable {
    let engine = AVAudioEngine()
    var converter: AVAudioConverter?
    var outputFormat: AVAudioFormat?
    var tapInstalled = false

    func convert(
        buffer: AVAudioPCMBuffer,
        sourceRate: Double,
        targetRate: Double
    ) -> [Float]? {
        guard let converter, let outputFormat else { return nil }
        let ratio = targetRate / sourceRate
        let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio + 32)
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else {
            return nil
        }

        var error: NSError?
        let consumed = ConsumedFlag()
        let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
            if consumed.value {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed.value = true
            outStatus.pointee = .haveData
            return buffer
        }

        guard error == nil, status != .error, let data = outBuffer.floatChannelData?[0] else {
            return nil
        }
        let count = Int(outBuffer.frameLength)
        guard count > 0 else { return nil }
        return Array(UnsafeBufferPointer(start: data, count: count))
    }

    func teardown() {
        if tapInstalled {
            engine.inputNode.removeTap(onBus: 0)
            tapInstalled = false
        }
        if engine.isRunning {
            engine.stop()
        }
        converter = nil
    }
}

/// Reference-typed flag used by the synchronous converter callback. The callback
/// fires once per `convert(to:error:)` invocation, so single-threaded mutation
/// is safe even though the surrounding closure is `@Sendable`.
private final class ConsumedFlag: @unchecked Sendable {
    var value: Bool = false
}
