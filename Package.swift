// swift-tools-version: 6.0
import PackageDescription

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
    targets: [
        .target(
            name: "SwiftWhisperCore"
        ),
        .target(
            name: "SwiftWhisperKit",
            dependencies: ["SwiftWhisperCore"]
        ),
        .executableTarget(
            name: "SwiftWhisperCLI",
            dependencies: ["SwiftWhisperKit"]
        ),
        .testTarget(
            name: "SwiftWhisperCoreTests",
            dependencies: ["SwiftWhisperCore"]
        ),
        .testTarget(
            name: "SwiftWhisperKitTests",
            dependencies: ["SwiftWhisperKit"]
        ),
    ],
    swiftLanguageModes: [.v6]
)
