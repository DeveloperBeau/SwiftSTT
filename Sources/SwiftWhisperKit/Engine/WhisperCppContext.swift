@preconcurrency import Foundation
import SwiftWhisperCore
@preconcurrency import whisper

/// Loaded whisper.cpp model context.
///
/// Owns a `whisper_context*` and frees it on deinit. All methods serialise
/// through the actor so concurrent calls are safe.
///
/// Use ``transcribe(samples:options:)`` to convert a 16 kHz mono Float PCM
/// buffer into ``TranscriptionSegment`` values. The context is reused
/// across calls so the underlying state stays warm.
public actor WhisperCppContext {

    // Safety: whisper_context* is owned exclusively by this actor.
    // All access is serialised through actor isolation, so no data race is possible.
    private nonisolated(unsafe) let context: OpaquePointer

    /// Loads a ggml model file from disk.
    ///
    /// If a matching `<stem>-encoder.mlmodelc` sits next to the ggml file,
    /// whisper.cpp auto-loads it for ANE-accelerated encoding. The
    /// `coreMLEncoderURL` argument is currently informational; the file
    /// convention does the actual work.
    public init(
        ggmlModelURL: URL,
        coreMLEncoderURL: URL? = nil
    ) throws(SwiftWhisperError) {
        var params = whisper_context_default_params()
        // whisper.cpp's Metal backend crashes inside the iOS Simulator's
        // Metal driver (ggml_metal_buffer_set_tensor hits an XPC shared-
        // memory API misuse). Real devices handle Metal fine. Force the CPU
        // backend on the simulator; everywhere else uses the GPU.
        #if targetEnvironment(simulator)
        params.use_gpu = false
        #else
        params.use_gpu = true
        #endif

        let path = ggmlModelURL.path
        guard
            let ctx = path.withCString({ cPath in
                whisper_init_from_file_with_params(cPath, params)
            })
        else {
            throw .modelLoadFailed("whisper_init_from_file_with_params returned NULL")
        }
        self.context = ctx
        _ = coreMLEncoderURL
    }

    deinit {
        whisper_free(context)
    }

    /// Transcribes a 16 kHz mono Float32 PCM buffer in one shot.
    ///
    /// `whisper_full` is synchronous and CPU/GPU-bound. The call blocks
    /// the actor's executor thread for its full duration (typically 1-5
    /// seconds for tiny.en, longer for bigger models). Callers should
    /// not invoke this from a `@MainActor` context.
    ///
    /// - Parameters:
    ///   - samples: 16 kHz mono Float32 PCM audio.
    ///   - options: decoding options; language, task, temperature, etc.
    ///   - onProgress: optional callback invoked with a fractional progress
    ///     value (0.0...1.0) as whisper.cpp decodes. It fires synchronously
    ///     on the executor thread running `whisper_full`, so keep it cheap
    ///     (e.g. hop to another actor with a `Task`).
    /// - Returns: the transcribed segments, empty if `samples` is empty.
    /// - Throws: ``SwiftWhisperError/decoderFailure(_:)`` if `whisper_full`
    ///   returns a non-zero status.
    public func transcribe(
        samples: [Float],
        options: DecodingOptions = .default,
        onProgress: (@Sendable (Double) -> Void)? = nil
    ) throws(SwiftWhisperError) -> [TranscriptionSegment] {
        guard !samples.isEmpty else { return [] }

        var params = WhisperCppParams.fullParams(from: options)

        // Drive progress from *both* whisper.cpp callbacks. They measure the
        // same thing (position through the audio) so they can't conflict;
        // the consumer's monotonic guard simply takes whichever is further
        // along. `new_segment_callback` works for sub-30s clips (each
        // finished segment carries an absolute `t1` timestamp, so
        // `t1 / totalDuration` is real fractional progress), while
        // `progress_callback` ticks every 5% of audio position and fills in
        // the gaps on longer clips. The box is held across `whisper_full`
        // via `withExtendedLifetime` and read back through the user-data
        // pointer.
        let totalCentiseconds = Double(samples.count) / 16_000.0 * 100.0
        let progressBox = onProgress.map {
            ProgressBox(callback: $0, totalCentiseconds: totalCentiseconds)
        }
        if let progressBox {
            let boxPointer = Unmanaged.passUnretained(progressBox).toOpaque()

            params.new_segment_callback = { _, state, _, userData in
                guard let userData, let state else { return }
                let box = Unmanaged<ProgressBox>.fromOpaque(userData).takeUnretainedValue()
                let count = whisper_full_n_segments_from_state(state)
                guard count > 0 else { return }
                let t1 = whisper_full_get_segment_t1_from_state(state, count - 1)
                let fraction = min(max(Double(t1) / box.totalCentiseconds, 0), 1)
                box.callback(fraction)
            }
            params.new_segment_callback_user_data = boxPointer

            params.progress_callback = { _, _, progress, userData in
                guard let userData else { return }
                Unmanaged<ProgressBox>.fromOpaque(userData)
                    .takeUnretainedValue()
                    .callback(Double(progress) / 100.0)
            }
            params.progress_callback_user_data = boxPointer
        }

        // `language` needs a stable C string for the duration of the call.
        let languageCString: ContiguousArray<CChar>? = options.language.map { lang in
            ContiguousArray(lang.utf8CString)
        }

        let rc = withExtendedLifetime(progressBox) {
            samples.withUnsafeBufferPointer { audioPtr -> Int32 in
                languageCString.withCStringOrNil { langPtr in
                    params.language = langPtr
                    return whisper_full(
                        context,
                        params,
                        audioPtr.baseAddress,
                        Int32(audioPtr.count)
                    )
                }
            }
        }
        guard rc == 0 else {
            throw .decoderFailure("whisper_full returned \(rc)")
        }

        let n = Int(whisper_full_n_segments(context))
        var result: [TranscriptionSegment] = []
        result.reserveCapacity(n)
        for i in 0..<n {
            let idx = Int32(i)
            guard let cText = whisper_full_get_segment_text(context, idx) else { continue }
            let text = String(cString: cText)
            let t0 = whisper_full_get_segment_t0(context, idx)
            let t1 = whisper_full_get_segment_t1(context, idx)
            result.append(
                TranscriptionSegment(
                    text: text,
                    start: Double(t0) / 100.0,
                    end: Double(t1) / 100.0
                )
            )
        }
        return result
    }
}

extension Optional where Wrapped == ContiguousArray<CChar> {
    /// Borrow a C pointer to the stored string, or `nil`, for the body's lifetime.
    fileprivate func withCStringOrNil<R>(_ body: (UnsafePointer<CChar>?) -> R) -> R {
        switch self {
        case .none:
            return body(nil)
        case .some(let array):
            return array.withUnsafeBufferPointer { buf in
                body(buf.baseAddress)
            }
        }
    }
}

/// Reference box that carries a progress closure (and the total audio
/// duration needed to turn a segment timestamp into a fraction) across
/// whisper.cpp's C `new_segment_callback_user_data` pointer.
private final class ProgressBox: Sendable {
    let callback: @Sendable (Double) -> Void
    let totalCentiseconds: Double

    init(callback: @escaping @Sendable (Double) -> Void, totalCentiseconds: Double) {
        self.callback = callback
        self.totalCentiseconds = totalCentiseconds
    }
}
