@preconcurrency import AVFoundation
import Foundation
import Synchronization
import SwiftWhisperCore

/// Production ``AudioInputProvider`` backed by `AVAudioEngine`.
///
/// Hides three pieces of platform plumbing:
///
/// 1. **Format conversion.** Hardware microphones report whatever rate they
///    like. `AVAudioConverter` resamples to the target rate so callers see
///    consistent Float32 mono buffers.
/// 2. **Permission.** The first `start` call requests microphone access.
///    Throws ``SwiftWhisperCore/SwiftWhisperError/micPermissionDenied`` if
///    the user declines.
/// 3. **iOS interruption handling.** Phone calls and Siri pause the engine;
///    the observer resumes it when the interruption ends. macOS has no audio
///    session so this is a no-op.
///
/// The host app must declare `NSMicrophoneUsageDescription` in `Info.plist`.
/// Sandboxed macOS apps additionally need
/// `com.apple.security.device.audio-input`.
public actor AVMicrophoneInput: AudioInputProvider {

    private let box: CaptureBox
    private var isCapturing: Bool = false

    #if os(iOS)
    private var interruptionObserver: NSObjectProtocol?
    #endif

    public init() {
        self.box = CaptureBox()
    }

    public func start(
        targetSampleRate: Double,
        bufferDurationSeconds: Double,
        onChunk: @Sendable @escaping ([Float]) -> Void
    ) async throws(SwiftWhisperError) {
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
        let inputRate = inputFormat.sampleRate
        let target = targetSampleRate
        let box = self.box

        box.engine.inputNode.installTap(
            onBus: 0,
            bufferSize: bufferSize,
            format: inputFormat
        ) { @Sendable buffer, _ in
            guard let samples = box.convert(buffer: buffer, sourceRate: inputRate, targetRate: target) else { return }
            onChunk(samples)
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

    public func stop() async {
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

/// Wraps non-`Sendable` AVFoundation refs. Mutable fields are written once on
/// the actor before the tap installs, then read-only from the audio thread.
private final class CaptureBox: @unchecked Sendable {
    let engine = AVAudioEngine()
    var converter: AVAudioConverter?
    var outputFormat: AVAudioFormat?
    var tapInstalled = false

    /// Returns `nil` on failure because the tap callback can't propagate Swift errors.
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
        let consumed = Atomic<Bool>(false)
        let status = converter.convert(to: outBuffer, error: &error) { _, outStatus in
            if consumed.load(ordering: .relaxed) {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed.store(true, ordering: .relaxed)
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
