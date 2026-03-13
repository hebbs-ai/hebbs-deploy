#!/bin/sh
# Install git hooks for the hebbs workspace.
# Run once after cloning: ./scripts/install-hooks.sh

HOOK_DIR="$(git rev-parse --git-dir)/hooks"

cat > "$HOOK_DIR/pre-commit" << 'HOOK'
#!/bin/sh
set -e

echo "pre-commit: cargo fmt --check"
cargo fmt --all --check || {
    echo "ERROR: formatting check failed. Run 'cargo fmt --all' and re-commit."
    exit 1
}

echo "pre-commit: cargo clippy"
cargo clippy --workspace --all-targets -- -D warnings 2>&1 || {
    echo "ERROR: clippy found warnings. Fix them and re-commit."
    exit 1
}

echo "pre-commit: all checks passed."
HOOK

chmod +x "$HOOK_DIR/pre-commit"
echo "Git hooks installed."
