# Contributing to SwiftWhisper

Thanks for considering a contribution. This document covers how to file issues,
propose changes, and get a pull request merged.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). By
participating, you agree to uphold its terms. See `CODE_OF_CONDUCT.md` for how
to report a concern.

## Ways to contribute

- **Bug reports** -- open an issue using the bug template. Include reproduction
  steps, expected vs actual output, OS/Xcode/Swift versions, and the model you
  were using.
- **Feature requests** -- open an issue using the feature template. Describe the
  use case before the proposed API.
- **Pull requests** -- fix bugs, add features, improve docs. Smaller PRs land
  faster than large ones.
- **Documentation** -- DocC articles in `Sources/SwiftWhisperKit/SwiftWhisperKit.docc/Articles/`
  are first-class contributions.

## Development setup

Requirements:

- macOS 15+
- Xcode 16.4+
- Swift 6.3 toolchain

Clone and build:

```bash
git clone https://github.com/DeveloperBeau/SwiftSTT.git
cd SwiftSTT
swift build
swift test
```

Run the CLI locally:

```bash
swift run swiftwhisper --help
swift run swiftwhisper list-models
swift run swiftwhisper download tiny
```

## Project layout

| Target | Purpose |
|--------|---------|
| `SwiftWhisperCore` | Pure-Swift models, errors, protocols. No CoreML, no AVFoundation. |
| `SwiftWhisperKit` | CoreML inference, audio capture, tokeniser, pipeline. |
| `SwiftWhisperCLI` | `swiftwhisper` executable. |

Tests mirror the source layout under `Tests/`.

## Branch and commit conventions

- Branch off `main`. Use a short prefix: `feat/`, `fix/`, `docs/`, `chore/`.
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit
  subjects: `feat(downloader): ...`, `fix(decoder): ...`, `docs: ...`.
- Subject line $\le$ 72 chars. No body required for trivial changes.

## Pull-request checklist

Before opening a PR:

- [ ] `swift build` succeeds with no warnings.
- [ ] `swift test` passes locally (438+ unit tests, ~10s).
- [ ] New code has tests. Behavioural change without a test is rare and needs
      justification in the PR description.
- [ ] Public API has DocC comments.
- [ ] `CHANGELOG.md` updated under `## [Unreleased]`.
- [ ] No em-dash characters (the lint workflow blocks them; use `--` or rewrite).
- [ ] No marketing-speak (`powerful`, `comprehensive`, `seamlessly`, `leverage`,
      `dive into`, `robust`) in production sources. The lint workflow blocks
      these.

CI runs build, tests, and prose lint on every PR. Integration tests run on
manual dispatch.

## Style

- Swift 6 strict concurrency. Annotate isolation explicitly.
- Use `@MainActor` only at the boundary; keep core logic isolation-agnostic.
- Prefer `actor` over `@unchecked Sendable`.
- Public API uses Swift naming conventions: omit needless words, fluent reads.

A `.swift-format` config lives at the repo root. CI runs
`swift format lint --strict` on every push.

Run `scripts/install-hooks.sh` once after cloning to enable a pre-commit hook
that lints staged Swift files locally before each commit.

### Linter choice

This project uses **`swift-format` only** (the official Apple linter, bundled
with the Swift toolchain). SwiftLint is **not** used.

The two tools have overlapping but non-identical rule sets, and running both
produces conflicting fixups (for example, brace placement and trailing-comma
style differ between defaults). Pick one. If you have SwiftLint installed
globally, configure your editor to disable it for this repo, or add a
`.swiftlint.yml` with `disabled_rules: [...]` shadowing every rule. Pull
requests that introduce a SwiftLint config will be closed.

## Releasing

Maintainers only. See `RELEASING.md` (TBD) for the tagging and DocC publish
flow. New contributors do not need to touch this.

## Questions

Open a GitHub Discussion or an issue with the `question` label.
