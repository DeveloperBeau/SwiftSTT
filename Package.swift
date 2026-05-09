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
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
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
            dependencies: [
                "SwiftWhisperKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
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
        .testTarget(
            name: "SwiftWhisperCLITests",
            dependencies: [
                "SwiftWhisperCLI",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: upcoming + [
                .defaultIsolation(MainActor.self),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
