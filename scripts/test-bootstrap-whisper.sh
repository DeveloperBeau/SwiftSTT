#!/usr/bin/env bash
# Checks the offline paths of bootstrap-whisper.sh in a scratch copy of the repo layout.
# It never downloads or builds whisper.cpp.
set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
scratch="$(mktemp -d "${TMPDIR:-/tmp}/swiftstt-bootstrap-test.XXXXXX")"
trap 'rm -rf "$scratch"' EXIT
failures=0

make_repo() {
    local repo="$scratch/$1"
    mkdir -p "$repo/scripts"
    cp "$here/bootstrap-whisper.sh" "$repo/scripts/"
    echo "let whisperCppVersion = \"$2\"" > "$repo/Package.swift"
    echo "$repo"
}

check() {
    local name="$1" expected_status="$2" expected_text="$3" status="$4" output="$5"
    if [[ "$status" == "$expected_status" && "$output" == *"$expected_text"* ]]; then
        echo "pass: $name"
    else
        echo "FAIL: $name (status $status, output: $output)"
        failures=$((failures + 1))
    fi
}

# A present framework for the pinned version means no download and no build.
repo="$(make_repo present v1.9.5)"
mkdir -p "$repo/.whisper/v1.9.5/whisper.xcframework"
output="$(WHISPER_TARBALL_URL="file:///nonexistent" "$repo/scripts/bootstrap-whisper.sh" 2>&1)"
check "present framework is a no-op" 0 "nothing to build" $? "$output"

# A version with no pinned sha256 is rejected before any download.
repo="$(make_repo unpinned v0.0.1)"
output="$(WHISPER_TARBALL_URL="file:///nonexistent" "$repo/scripts/bootstrap-whisper.sh" 2>&1)"
check "unpinned version is rejected" 1 "no pinned tarball sha256 for v0.0.1" $? "$output"

# A tarball with the wrong sha256 is rejected and leaves no framework behind.
repo="$(make_repo tampered v1.9.5)"
echo "not the real source" > "$scratch/fake.tar.gz"
output="$(WHISPER_TARBALL_URL="file://$scratch/fake.tar.gz" "$repo/scripts/bootstrap-whisper.sh" 2>&1)"
check "sha256 mismatch is rejected" 1 "sha256 mismatch" $? "$output"
if [[ -e "$repo/.whisper/v1.9.5/whisper.xcframework" ]]; then
    echo "FAIL: sha256 mismatch left a framework behind"
    failures=$((failures + 1))
fi

# A matching sha256 gets past verification. The fake tarball then fails to unpack.
repo="$(make_repo override v1.9.5)"
sha="$(shasum -a 256 "$scratch/fake.tar.gz" | cut -d' ' -f1)"
output="$(WHISPER_TARBALL_URL="file://$scratch/fake.tar.gz" WHISPER_TARBALL_SHA="$sha" \
    "$repo/scripts/bootstrap-whisper.sh" 2>&1)"
status=$?
if [[ $status != 0 && "$output" != *"sha256 mismatch"* ]]; then
    echo "pass: matching sha256 passes verification"
else
    echo "FAIL: matching sha256 passes verification (status $status, output: $output)"
    failures=$((failures + 1))
fi

# Without cmake the script stops with a clear message.
repo="$(make_repo nocmake v1.9.5)"
output="$(PATH="/usr/bin:/bin" WHISPER_TARBALL_URL="file:///nonexistent" \
    "$repo/scripts/bootstrap-whisper.sh" 2>&1)"
check "missing cmake is reported" 1 "error: cmake is required to build whisper.cpp" $? "$output"

exit $((failures > 0))
