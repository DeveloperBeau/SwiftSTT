// swift-tools-version: 6.4
import Foundation
import PackageDescription

let upcoming: [SwiftSetting] = [
    .enableUpcomingFeature("NonisolatedNonsendingByDefault"),  // SE-0461
    .enableUpcomingFeature("InferIsolatedConformances"),  // SE-0466
    .enableUpcomingFeature("ExistentialAny"),  // SE-0335
    .enableUpcomingFeature("MemberImportVisibility"),  // SE-0444
]

// scripts/bootstrap-whisper.sh reads this version. Upstream publishes no xcframework for it.
let whisperCppVersion = "v1.9.5"
let whisperCppLocalPath = ".whisper/\(whisperCppVersion)/whisper.xcframework"

// A checkout with a local build uses it. A consumer that resolves SwiftSTT by URL never runs
// the script, so it falls back to the zip attached to the SwiftSTT release.
let whisperCppTarget: Target =
    FileManager.default.fileExists(atPath: Context.packageDirectory + "/" + whisperCppLocalPath)
    ? .binaryTarget(name: "WhisperCpp", path: whisperCppLocalPath)
    : .binaryTarget(
        name: "WhisperCpp",
        url: "https://github.com/DeveloperBeau/SwiftSTT/releases/download/"
            + "whisper-\(whisperCppVersion)/whisper-\(whisperCppVersion)-xcframework.zip",
        checksum: "08827b6ffcd8f7f6c26a77ef3595be66402d523a89fc9f26287ab8e280f86d1d"
    )

let package = Package(
    name: "SwiftSTT",
    platforms: [
        .iOS(.v18),
        .macOS(.v15),
    ],
    products: [
        .library(name: "SwiftSTTCore", targets: ["SwiftSTTCore"]),
        .library(name: "SwiftSTTKit", targets: ["SwiftSTTKit"]),
        .executable(name: "swiftstt", targets: ["SwiftSTTCLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-docc-plugin", from: "1.4.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/weichsel/ZIPFoundation", from: "0.9.0"),
    ],
    targets: [
        whisperCppTarget,
        .target(
            name: "SwiftSTTCore",
            swiftSettings: upcoming
        ),
        .target(
            name: "SwiftSTTKit",
            dependencies: [
                "SwiftSTTCore",
                "WhisperCpp",
                .product(name: "ZIPFoundation", package: "ZIPFoundation"),
            ],
            swiftSettings: upcoming
        ),
        .executableTarget(
            name: "SwiftSTTCLI",
            dependencies: [
                "SwiftSTTKit",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: upcoming + [
                .defaultIsolation(MainActor.self)
            ]
        ),
        .testTarget(
            name: "SwiftSTTCoreTests",
            dependencies: ["SwiftSTTCore"],
            swiftSettings: upcoming
        ),
        .testTarget(
            name: "SwiftSTTKitTests",
            dependencies: ["SwiftSTTKit"],
            swiftSettings: upcoming
        ),
        .testTarget(
            name: "SwiftSTTCLITests",
            dependencies: [
                "SwiftSTTCLI",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: upcoming + [
                .defaultIsolation(MainActor.self)
            ]
        ),
        .testTarget(
            name: "SwiftSTTIntegrationTests",
            dependencies: ["SwiftSTTKit"],
            swiftSettings: upcoming
        ),
    ],
    swiftLanguageModes: [.v6]
)
