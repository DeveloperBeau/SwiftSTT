import Foundation

/// The full streaming transcription pipeline as a single actor.
///
/// A `Transcriber` consumes an audio stream and emits ``TranscriptionResult``
/// updates as the model progresses. The result stream uses the local-agreement
/// pattern: each update carries both the confirmed text so far and a tentative
/// hypothesis for the most recent audio. Consumers typically render the two in
/// different styles so users can see the model thinking.
///
/// ## Example
///
/// ```swift
/// let transcriber: any Transcriber = TranscriptionPipeline(...)
///
/// Task {
///     for await update in await transcriber.resultStream {
///         print(update.text + " " + update.hypothesis)
///     }
/// }
///
/// try await transcriber.transcribe(audioStream: capture.audioStream)
/// ```
///
/// Stop the transcriber by cancelling the stream consumer or calling ``stop()``.
public protocol Transcriber: Actor {

    /// Stream of transcription updates. Emits a new value every time the
    /// confirmed or hypothesis text changes.
    var resultStream: AsyncStream<TranscriptionResult> { get }

    /// Begins consuming audio and producing results. Returns once the input
    /// stream finishes or an unrecoverable error is thrown.
    func transcribe(audioStream: AsyncStream<AudioChunk>) async throws(SwiftSTTError)

    /// Halts decoding, finalises any in-flight hypothesis, and finishes
    /// ``resultStream``. Safe to call more than once.
    func stop() async
}
