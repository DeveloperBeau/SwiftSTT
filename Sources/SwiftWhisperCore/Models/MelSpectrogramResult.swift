import Foundation

public struct MelSpectrogramResult: Sendable, Equatable {
    public let frames: [Float]
    public let nMels: Int
    public let nFrames: Int

    public init(frames: [Float], nMels: Int, nFrames: Int) {
        precondition(frames.count == nMels * nFrames, "frames length must equal nMels * nFrames")
        self.frames = frames
        self.nMels = nMels
        self.nFrames = nFrames
    }
}
