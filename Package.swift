// swift-tools-version: 6.3
import PackageDescription

let upcoming: [SwiftSetting] = [
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),  // SE-0461
    .enableUpcomingFeature("InferIsolatedConformances"),       // SE-0466
    .enableUpcomingFeature("ExistentialAny"),                  // SE-0335
    .enableUpcomingFeature("MemberImportVisibility"),          // SE-0444
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
            swiftSettings: upcoming
        ),
        .target(
            name: "SwiftWhisperKit",
            dependencies: ["SwiftWhisperCore"],
            swiftSettings: upcoming
        ),
        .executableTarget(
            name: "SwiftWhisperCLI",
            dependencies: ["SwiftWhisperKit"],
            swiftSettings: upcoming + [
                .defaultIsolation(MainActor.self),
            ]
        ),
        .testTarget(
            name: "SwiftWhisperCoreTests",
            dependencies: ["SwiftWhisperCore"],
            swiftSettings: upcoming
        ),
        .testTarget(
            name: "SwiftWhisperKitTests",
            dependencies: ["SwiftWhisperKit"],
            swiftSettings: upcoming
        ),
    ],
    swiftLanguageModes: [.v6]
)
