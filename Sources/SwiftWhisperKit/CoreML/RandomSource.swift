import Foundation

/// Pluggable source of random `UInt64` values consumed by the decoder's
/// temperature sampler.
///
/// Production code uses ``SystemRandom`` which delegates to
/// `SystemRandomNumberGenerator`. Tests inject ``SeededRandom`` so a sampling
/// run with a given temperature produces a deterministic token sequence.
public protocol RandomSource: Sendable {

    /// Returns the next random 64-bit integer.
    mutating func next() -> UInt64
}

/// Default `RandomSource` that defers to the system CSPRNG.
public struct SystemRandom: RandomSource {

    /// Creates a new SystemRandom with the supplied values.
    public init() {}

    /// Returns the next pseudo-random value.
    public mutating func next() -> UInt64 {
        var rng = SystemRandomNumberGenerator()
        return rng.next()
    }
}

/// Deterministic `RandomSource` for tests.
///
/// Uses SplitMix64, which is small, well-distributed for non-cryptographic use, and produces the same sequence
/// for the same seed.
public struct SeededRandom: RandomSource {

    private var state: UInt64

    /// Creates a new SeededRandom with the supplied values.
    public init(seed: UInt64) {
        self.state = seed
    }

    /// Returns the next pseudo-random value.
    public mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z &>> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z &>> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z &>> 31)
    }
}
