import Foundation
import Testing
@testable import SwiftWhisperCLI

@Suite("CacheDirectoryOption environment resolution")
struct EnvironmentResolutionTests {

    @Test("Explicit flag overrides env var")
    func flagOverridesEnv() throws {
        let env = ["SWIFTWHISPER_CACHE_DIR": "/from/env"]
        let resolved = try #require(
            CacheDirectoryOption.resolve("/from/flag", environment: env)
        )
        #expect(resolved.path == "/from/flag")
    }

    @Test("Env var overrides default when flag is nil")
    func envOverridesDefault() throws {
        let env = ["SWIFTWHISPER_CACHE_DIR": "/from/env"]
        let resolved = try #require(CacheDirectoryOption.resolve(nil, environment: env))
        #expect(resolved.path == "/from/env")
    }

    @Test("Default applies (returns nil) when both are unset")
    func defaultWhenBothUnset() {
        #expect(CacheDirectoryOption.resolve(nil, environment: [:]) == nil)
    }

    @Test("Empty env var falls through to default")
    func emptyEnvFallsThrough() {
        let env = ["SWIFTWHISPER_CACHE_DIR": ""]
        #expect(CacheDirectoryOption.resolve(nil, environment: env) == nil)
    }

    @Test("Tilde in env var is expanded")
    func tildeInEnv() throws {
        let env = ["SWIFTWHISPER_CACHE_DIR": "~/swiftwhisper-env-cache"]
        let resolved = try #require(CacheDirectoryOption.resolve(nil, environment: env))
        let path = resolved.path
        #expect(!path.contains("~"))
        #expect(path.hasSuffix("/swiftwhisper-env-cache"))
    }

    @Test("Invalid path in env var is returned as-is; failure surfaces at use time")
    func invalidPathPassesThrough() throws {
        let env = ["SWIFTWHISPER_CACHE_DIR": "/definitely/does/not/exist/here"]
        let resolved = try #require(CacheDirectoryOption.resolve(nil, environment: env))
        #expect(resolved.path == "/definitely/does/not/exist/here")
    }

    @Test("Empty flag falls through to env var")
    func emptyFlagFallsToEnv() throws {
        let env = ["SWIFTWHISPER_CACHE_DIR": "/from/env"]
        let resolved = try #require(CacheDirectoryOption.resolve("", environment: env))
        #expect(resolved.path == "/from/env")
    }
}
