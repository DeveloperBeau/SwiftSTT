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
        params.use_gpu = true

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
    public func transcribe(
        samples: [Float],
        options: DecodingOptions = .default
    ) throws(SwiftWhisperError) -> [TranscriptionSegment] {
        guard !samples.isEmpty else { return [] }

        var params = WhisperCppParams.fullParams(from: options)

        // `language` needs a stable C string for the duration of the call.
        let languageCString: ContiguousArray<CChar>? = options.language.map { lang in
            ContiguousArray(lang.utf8CString)
        }

        let rc = samples.withUnsafeBufferPointer { audioPtr -> Int32 in
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
