#!/usr/bin/env sh
# Installs the project's git hooks by pointing core.hooksPath at the tracked
# directory. Re-run this after cloning or after pulling a hook update.

set -e

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/.." && pwd)

git -C "$repo_root" config core.hooksPath "scripts/git-hooks"
chmod +x "$repo_root/scripts/git-hooks/pre-commit"

echo "Git hooks installed (core.hooksPath -> scripts/git-hooks)."
echo "Pre-commit will run 'swift format lint --strict' on staged Swift files."
