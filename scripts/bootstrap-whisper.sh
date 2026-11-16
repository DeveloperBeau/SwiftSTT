#!/usr/bin/env bash
# Gets or builds whisper.xcframework at .whisper/<version>/ for local SwiftSTT builds.
# The version comes from whisperCppVersion in Package.swift, so it is pinned in one place.
#
# Usage: scripts/bootstrap-whisper.sh [--zip]
#   --zip  Also write the release zip and print its SwiftPM checksum.
#
# Environment (for tests):
#   WHISPER_TARBALL_URL  Override the source tarball URL.
#   WHISPER_TARBALL_SHA  Override the pinned sha256 of the tarball.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"

version="$(sed -n 's/^let whisperCppVersion = "\(v[0-9.]*\)"$/\1/p' "$root/Package.swift")"
if [[ -z "$version" ]]; then
    echo "error: cannot read whisperCppVersion from Package.swift" >&2
    exit 1
fi

# sha256 of the GitHub source tarball for each supported version.
case "$version" in
    v1.9.5) pinned_sha="ff1a9053feb509ff9d7729703355541ae9690073a6b1c40eb692c962e0dc1720" ;;
    *)
        echo "error: no pinned tarball sha256 for $version, add one to $0" >&2
        exit 1
        ;;
esac
pinned_sha="${WHISPER_TARBALL_SHA:-$pinned_sha}"
url="${WHISPER_TARBALL_URL:-https://github.com/ggml-org/whisper.cpp/archive/refs/tags/$version.tar.gz}"

dest="$root/.whisper/$version"
framework="$dest/whisper.xcframework"
zip="$dest/whisper-$version-xcframework.zip"

make_zip() {
    if [[ ! -f "$zip" ]]; then
        (cd "$dest" && ditto -c -k --keepParent whisper.xcframework "$zip.tmp" && mv "$zip.tmp" "$zip")
    fi
    echo "zip: $zip"
    echo "checksum: $(cd "$root" && swift package compute-checksum "$zip")"
}

want_zip=0
[[ "${1:-}" == "--zip" ]] && want_zip=1

if [[ -d "$framework" ]]; then
    echo "whisper.xcframework $version is present, nothing to build."
    [[ $want_zip == 1 ]] && make_zip
    exit 0
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "error: cmake is required to build whisper.cpp. Install it (for example brew install cmake) and run again." >&2
    exit 1
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/swiftstt-whisper.XXXXXX")"
trap 'rm -rf "$work"' EXIT

echo "Downloading whisper.cpp $version source"
curl -sSLf -o "$work/src.tar.gz" "$url"
actual_sha="$(shasum -a 256 "$work/src.tar.gz" | cut -d' ' -f1)"
if [[ "$actual_sha" != "$pinned_sha" ]]; then
    echo "error: sha256 mismatch for $url" >&2
    echo "  expected $pinned_sha" >&2
    echo "  actual   $actual_sha" >&2
    exit 1
fi

mkdir "$work/src"
tar -xzf "$work/src.tar.gz" -C "$work/src" --strip-components 1
echo "Building whisper.xcframework (this takes several minutes)"
(cd "$work/src" && ./build-xcframework.sh)

# Move into place last, so an interrupted build never looks complete.
mkdir -p "$dest"
mv "$work/src/build-apple/whisper.xcframework" "$framework"
echo "Built $framework"
# SwiftPM caches the manifest result, so a checkout that resolved before keeps the release zip.
echo "If SwiftPM resolved this checkout before, run 'swift package purge-cache' to use the local build."
[[ $want_zip == 1 ]] && make_zip
exit 0
