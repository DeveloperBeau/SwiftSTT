// swift-tools-version: 6.3
import PackageDescription

// Upcoming-feature flags activate Swift 6.1/6.2 "approachable concurrency"
// behaviour while staying on language mode v6:
//
//   NonisolatedNonsendingByDefault  - SE-0461 - async funcs run on caller's
//                                     executor instead of being implicitly
//                                     offloaded; eliminates many spurious
//                                     "sending risks data race" diagnostics.
//   InferIsolatedConformances       - SE-0466 - protocol conformances on
//                                     actor-isolated types infer the host
//                                     isolation by default.
//
// Library targets stay nonisolated by default. The CLI executable opts into
// MainActor isolation since it is a single-threaded program by nature.
let approachableConcurrency: [SwiftSetting] = [
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),
    .enableUpcomingFeature("InferIsolatedConformances"),
]

let package = Package(
    name: "SwiftWhisper",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "SwiftWhisperCore", targets: ["SwiftWhisperCore"]),
        .library(name: "SwiftWhisperKit", targets: ["SwiftWhisperKit"]),
        .executable(name: "swiftwhisper", targets: ["SwiftWhisperCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.4.0"),
    ],
    targets: [
        .target(
            name: "SwiftWhisperCore",
            swiftSettings: approachableConcurrency
        ),
        .target(
            name: "SwiftWhisperKit",
            dependencies: ["SwiftWhisperCore"],
            swiftSettings: approachableConcurrency
        ),
        .executableTarget(
            name: "SwiftWhisperCLI",
            dependencies: ["SwiftWhisperKit"],
            swiftSettings: approachableConcurrency + [
                .defaultIsolation(MainActor.self),
            ]
        ),
        .testTarget(
            name: "SwiftWhisperCoreTests",
            dependencies: ["SwiftWhisperCore"],
            swiftSettings: approachableConcurrency
        ),
        .testTarget(
            name: "SwiftWhisperKitTests",
            dependencies: ["SwiftWhisperKit"],
            swiftSettings: approachableConcurrency
        ),
    ],
    swiftLanguageModes: [.v6]
)
