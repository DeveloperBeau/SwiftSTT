@preconcurrency import Foundation
import SwiftWhisperCore
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
        // transcribing" — it must stay false. Auto-detection is driven by
        // `params.language` being nil/"auto" (set in WhisperCppContext),
        // which detects the language *and* transcribes in it.
        params.detect_language = false
        params.temperature = options.temperature
        params.suppress_blank = options.suppressBlank
        params.n_threads = Int32(min(ProcessInfo.processInfo.activeProcessorCount, 4))
        params.max_tokens = 0
        return params
    }
}
