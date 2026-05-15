# Security Policy

## Supported versions

Until the project reaches `1.0.0`, only the latest tagged release on `main`
receives fixes. Older pre-1.0 tags are not patched.

| Version | Supported |
|---------|-----------|
| latest `0.x` tag | yes |
| older `0.x` tags | no |

## Reporting an issue

Please do **not** open a public GitHub issue for security-relevant reports.

Use GitHub's [private vulnerability reporting][report] for this repository.
That channel keeps the report visible only to the maintainer until a fix is
ready.

[report]: https://github.com/DeveloperBeau/SwiftSTT/security/advisories/new

Include in your report:

- A description of the issue and its impact.
- Reproduction steps or a minimal proof of concept.
- The commit hash or release tag you tested against.
- Your name and any disclosure preference (credit, anonymous, embargo window).

You can expect:

- An acknowledgement within 5 business days.
- A triage decision within 14 business days.
- A coordinated fix and disclosure timeline if the report is accepted.

## Scope

In scope:

- Code in `Sources/` (`SwiftSTTCore`, `SwiftSTTKit`, `SwiftSTTCLI`).
- The `swiftstt` CLI binary distributed via tagged releases.
- Build and CI configuration in this repository.

Out of scope:

- Third-party dependencies. Report those upstream
  (e.g. `swift-argument-parser`).
- Pre-trained Whisper model weights. Report concerns to OpenAI.
- Issues that require a compromised machine to exploit.

## Disclosure

Once a fix is released, the advisory is published in
[GitHub Security Advisories][gsa] for this repository, and credited reporters
are listed in the release notes unless they request otherwise.

[gsa]: https://github.com/DeveloperBeau/SwiftSTT/security/advisories
