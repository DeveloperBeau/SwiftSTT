@preconcurrency import Foundation
import SwiftSTTCore
@preconcurrency import whisper

/// Maps SwiftSTT's ``DecodingOptions`` onto whisper.cpp's
/// `whisper_full_params` C struct.
public enum WhisperCppParams {

    /// Builds a `whisper_full_params` from `options`.
    ///
    /// Greedy sampling is used by default. Beam search and other
    /// strategies are not yet exposed.
    public static func fullParams(from options: DecodingOptions) -> whisper_full_params {
        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        params.translate = (options.task == .translate)
        params.no_context = true
        params.single_segment = false
        params.print_special = false
        params.print_progress = false
        params.print_realtime = false
        params.print_timestamps = false
        // `detect_language` means "detect the language and exit *without*
        // transcribing", so it must stay false. Auto-detection is driven by
        // `params.language` being nil/"auto" (set in `withCStrings`),
        // which detects the language *and* transcribes in it.
        params.detect_language = false
        params.temperature = options.temperature
        params.suppress_blank = options.suppressBlank
        params.n_threads = Int32(min(ProcessInfo.processInfo.activeProcessorCount, 4))
        params.max_tokens = 0
        return params
    }

    /// Calls `body` with `params` whose `language` and `initial_prompt` point at C copies of
    /// `options.language` and `options.initialPrompt`.
    ///
    /// The pointers are valid only while `body` runs, so `body` must make the whole synchronous
    /// `whisper_full` call. A `nil` option gives a `nil` pointer. The prompt bytes are passed as
    /// they are, and a C reader stops at the first NUL byte.
    static func withCStrings<R>(
        from options: DecodingOptions,
        appliedTo params: whisper_full_params,
        _ body: (whisper_full_params) -> R
    ) -> R {
        // Both strings need a stable C buffer for the duration of the call.
        let languageCString: ContiguousArray<CChar>? = options.language.map { lang in
            ContiguousArray(lang.utf8CString)
        }
        let promptCString: ContiguousArray<CChar>? = options.initialPrompt.map { prompt in
            ContiguousArray(prompt.utf8CString)
        }

        return languageCString.withCStringOrNil { langPtr in
            promptCString.withCStringOrNil { promptPtr in
                var decoderParams = params
                decoderParams.language = langPtr
                decoderParams.initial_prompt = promptPtr
                return body(decoderParams)
            }
        }
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
