@preconcurrency import AVFoundation
import Foundation
import SwiftWhisperCore

/// Microphone capture backed by `AVAudioEngine`, with on-the-fly resampling to the
/// 16 kHz mono Float32 format Whisper expects.
///
/// This is the production ``AudioCapturer`` used by both the iOS host and the
/// macOS CLI. It hides three pieces of platform plumbing that would otherwise
/// leak into every caller:
///
/// 1. **Format conversion.** Hardware microphones report whatever rate they like
///    (44.1 kHz on most Macs, 48 kHz on iPhones, sometimes 24 kHz on AirPods).
///    `AVAudioConverter` handles the resampling so the rest of the pipeline can
///    assume 16 kHz mono Float32.
/// 2. **Stream cleanup.** When the consumer of ``audioStream`` cancels its task,
///    the engine and tap are torn down automatically through the continuation's
///    `onTermination` handler. No need to remember to call ``stopCapture()`` on
///    cancellation.
/// 3. **Interruptions.** On iOS, the audio session's interruption notifications
///    pause and resume the engine around phone calls and Siri activations.
///    On macOS this isn't needed because there's no audio session to lose.
///
/// ## Permission
///
/// Microphone access is requested on the first call to ``startCapture()``. The
/// host app must declare `NSMicrophoneUsageDescription` in `Info.plist`. If the
/// macOS host is sandboxed it must also enable the
/// `com.apple.security.device.audio-input` entitlement.
///
/// ## Example
///
/// ```swift
/// let capture = AVAudioCapture()
///
/// Task {
///     for await chunk in capture.audioStream {
///         print("got \(chunk.samples.count) samples at \(chunk.timestamp)s")
///     }
/// }
///
/// try await capture.startCapture()
/// // ... later
/// await capture.stopCapture()
/// ```
public actor AVAudioCapture: AudioCapturer {

    /// Async stream of resampled audio chunks. Stable for the lifetime of this
    /// actor; consumers can `for await` it before ``startCapture()`` returns.
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

    /// Creates a capture actor.
    ///
    /// - Parameters:
    ///   - targetSampleRate: sample rate to resample to. Defaults to 16 000 Hz to
    ///     match Whisper. Override only if you are wiring a non-Whisper consumer.
    ///   - bufferDurationSeconds: requested tap buffer length. The default of
    ///     64 ms balances callback frequency (avoiding overhead) against capture
    ///     latency (keeping the VAD responsive).
    public init(targetSampleRate: Double = 16_000, bufferDurationSeconds: Double = 0.064) {
        self.targetSampleRate = targetSampleRate
        self.bufferDurationSeconds = bufferDurationSeconds

        let (stream, continuation) = AsyncStream<AudioChunk>.makeStream()
        self.audioStream = stream
        self.continuation = continuation
        self.box = CaptureBox()

        let box = self.box
        continuation.onTermination = { @Sendable _ in
            box.teardown()
        }
    }

    /// Boots the audio engine, requests permission if needed, installs the input
    /// tap, and starts streaming chunks.
    ///
    /// Throws ``SwiftWhisperError/micPermissionDenied`` if the user has not
    /// granted access; ``SwiftWhisperError/audioConversionFailed`` if the
    /// resampler refuses to set up; or ``SwiftWhisperError/audioCaptureFailed(_:)``
    /// for any other engine or session error.
    ///
    /// Calling more than once without ``stopCapture()`` is a no-op.
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

    /// Stops the engine, removes the tap, deactivates the audio session (iOS),
    /// and finishes ``audioStream``. Idempotent.
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

/// Holds non-`Sendable` AVFoundation references behind a `Sendable` boundary.
///
/// `AVAudioEngine` and `AVAudioConverter` aren't `Sendable`, but the tap callback
/// runs on a real-time audio thread that sits outside actor isolation, so we
/// need a way to reach them from there without a hop. The box owns the engine,
/// the converter, and a flag tracking whether the tap is installed. Mutations
/// happen only inside `startCapture()` and `teardown()`, both of which run
/// under the actor's serial isolation, so the manual `@unchecked Sendable`
/// claim holds.
private final class CaptureBox: @unchecked Sendable {
    let engine = AVAudioEngine()
    var converter: AVAudioConverter?
    var outputFormat: AVAudioFormat?
    var tapInstalled = false

    /// Resamples a single capture buffer to the target rate as a contiguous
    /// `[Float]`. Returns `nil` on conversion failure rather than throwing,
    /// because the tap callback can't propagate Swift errors.
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

    /// Removes any tap and stops the engine. Safe to call when nothing is running.
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

/// Reference flag the converter callback uses to feed the buffer exactly once.
///
/// `AVAudioConverter.convert(to:error:withInputFrom:)` may invoke its callback
/// multiple times per call; we want to hand over the buffer the first time and
/// say `noDataNow` after that. A struct-typed `var` captured by the `@Sendable`
/// closure would trip Swift 6 capture rules, so the flag lives on a class.
private final class ConsumedFlag: @unchecked Sendable {
    var value: Bool = false
}
