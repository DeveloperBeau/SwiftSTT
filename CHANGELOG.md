# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Background `URLSession` download mode for Whisper models, with delegate-bridged
  progress streams and relaunch recovery.
- `--background` flag on the `swiftstt download` CLI command.
- DocC article: `Background-Downloads` with iOS integration walkthrough.
- Integration test target `SwiftSTTIntegrationTests` (gated by
  `SWIFTSTT_RUN_INTEGRATION=1`).
- GitHub Actions workflows: `ci.yml`, `lint.yml`, `integration.yml`.

### Changed
- README expanded with badges, architecture diagram, and quick-start example.
- Documentation hygiene pass: every public type now has a doc comment.

### Fixed
- `LoadedModels` API usage corrected in README and DocC examples.

[Unreleased]: https://github.com/DeveloperBeau/SwiftSTT/compare/HEAD...HEAD
