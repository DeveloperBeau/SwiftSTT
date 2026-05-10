@preconcurrency import AVFoundation
import Foundation
import SwiftWhisperCore
import Synchronization

/// File-backed ``AudioInputProvider`` for offline transcription.
///
/// Opens an audio file with `AVAudioFile`, resamples it to the requested rate
/// using `AVAudioConverter`, and emits Float32 mono buffers via the `onChunk`
/// callback supplied to ``start(targetSampleRate:bufferDurationSeconds:onChunk:)``.
///
/// Reading runs in an unstructured `Task` spawned from `start`; the call returns
/// as soon as the file is opened and conversion is configured. Use
/// ``waitUntilComplete()`` to await EOF (or a `stop()`-induced cancellation).
///
/// Format tolerance: anything `AVAudioFile` can open and `AVAudioConverter`
/// can resample to 16 kHz mono Float32. WAV, AIFF, and CAF are the well-tested
/// containers. M4A (AAC) typically works when the host platform has the codec
/// available; MP3 likewise depends on platform codec presence. When the
/// container or codec is unavailable, ``start(targetSampleRate:bufferDurationSeconds:onChunk:)``
/// throws ``SwiftWhisperCore/SwiftWhisperError/audioCaptureFailed(_:)`` with
/// the underlying reason.
public actor AudioFileInput: AudioInputProvider {

    private let fileURL: URL
    private var readTask: Task<Void, Never>?
    private var isRunning = false
    private let completionBox: CompletionBox

    public init(fileURL: URL) {
        self.fileURL = fileURL
        self.completionBox = CompletionBox()
    }

    public func start(
        targetSampleRate: Double,
        bufferDurationSeconds: Double,
        onChunk: @Sendable @escaping ([Float]) -> Void
    ) async throws(SwiftWhisperError) {
        guard !isRunning else { return }

        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: fileURL)
        } catch {
            throw .audioCaptureFailed("file: \(error.localizedDescription)")
        }

        guard
            let outputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: targetSampleRate,
                channels: 1,
                interleaved: false
            ),
            let converter = AVAudioConverter(from: file.processingFormat, to: outputFormat)
        else {
            throw .audioConversionFailed
        }

        let sourceRate = file.processingFormat.sampleRate
        let sourceFramesPerBuffer = AVAudioFrameCount(
            max(256, sourceRate * bufferDurationSeconds)
        )
        let outputFramesPerBuffer = AVAudioFrameCount(
            max(256, targetSampleRate * bufferDurationSeconds + 32)
        )

        completionBox.reset()
        isRunning = true

        let box = completionBox
        readTask = Task.detached {
            await Self.readLoop(
                file: file,
                converter: converter,
                outputFormat: outputFormat,
                sourceFramesPerBuffer: sourceFramesPerBuffer,
                outputFramesPerBuffer: outputFramesPerBuffer,
                onChunk: onChunk
            )
            box.complete()
        }
    }

    public func stop() async {
        guard isRunning else { return }
        isRunning = false
        readTask?.cancel()
        if let task = readTask {
            await task.value
        }
        readTask = nil
    }

    /// Awaits the read loop's completion. Returns once the file reaches EOF
    /// or ``stop()`` cancels the loop.
    public func waitUntilComplete() async {
        await completionBox.wait()
    }

    // MARK: - Read loop

    private static func readLoop(
        file: AVAudioFile,
        converter: AVAudioConverter,
        outputFormat: AVAudioFormat,
        sourceFramesPerBuffer: AVAudioFrameCount,
        outputFramesPerBuffer: AVAudioFrameCount,
        onChunk: @Sendable @escaping ([Float]) -> Void
    ) async {
        let sourceFormat = file.processingFormat

        while !Task.isCancelled {
            guard
                let inputBuffer = AVAudioPCMBuffer(
                    pcmFormat: sourceFormat,
                    frameCapacity: sourceFramesPerBuffer
                )
            else { return }

            do {
                try file.read(into: inputBuffer, frameCount: sourceFramesPerBuffer)
            } catch {
                return
            }

            if inputBuffer.frameLength == 0 {
                return
            }

            guard
                let outputBuffer = AVAudioPCMBuffer(
                    pcmFormat: outputFormat,
                    frameCapacity: outputFramesPerBuffer
                )
            else { return }

            var conversionError: NSError?
            let consumed = Atomic<Bool>(false)
            let status = converter.convert(to: outputBuffer, error: &conversionError) {
                _, outStatus in
                if consumed.load(ordering: .relaxed) {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                consumed.store(true, ordering: .relaxed)
                outStatus.pointee = .haveData
                return inputBuffer
            }

            if status == .error || conversionError != nil {
                return
            }

            if outputBuffer.frameLength > 0,
                let channelData = outputBuffer.floatChannelData?[0]
            {
                let count = Int(outputBuffer.frameLength)
                let samples = Array(UnsafeBufferPointer(start: channelData, count: count))
                onChunk(samples)
            }

            if inputBuffer.frameLength < sourceFramesPerBuffer {
                return
            }

            await Task.yield()
        }
    }
}

// MARK: - CompletionBox

/// Lets external callers `await` the read loop's end without holding actor
/// state hostage. The read loop runs as a detached `Task`, so we can't simply
/// `await readTask.value` from arbitrary contexts (that would block the actor).
private final class CompletionBox: @unchecked Sendable {

    private let mutex: Mutex<State>

    private struct State {
        var isComplete = false
        var waiters: [CheckedContinuation<Void, Never>] = []
    }

    init() {
        self.mutex = Mutex(State())
    }

    func reset() {
        mutex.withLock { state in
            state.isComplete = false
            state.waiters.removeAll()
        }
    }

    func complete() {
        let waiters: [CheckedContinuation<Void, Never>] = mutex.withLock { state in
            guard !state.isComplete else { return [] }
            state.isComplete = true
            let drained = state.waiters
            state.waiters.removeAll()
            return drained
        }
        for waiter in waiters {
            waiter.resume()
        }
    }

    func wait() async {
        await withCheckedContinuation { continuation in
            let alreadyDone: Bool = mutex.withLock { state in
                if state.isComplete { return true }
                state.waiters.append(continuation)
                return false
            }
            if alreadyDone {
                continuation.resume()
            }
        }
    }
}
