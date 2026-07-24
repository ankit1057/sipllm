#!/usr/bin/env bash
# Build and (optionally) upload a macOS (darwin) prebuilt sipllm bundle.
#
# macOS release bundles are NOT built on CI — they are built and uploaded
# manually from a local macOS machine with this script. It reproduces the
# "Stage bundle" layout from .github/workflows/release.yml byte-for-byte and
# emits an asset named exactly as install.sh expects:
#   sipllm-<VERSION>-darwin-<ARCH>.tar.gz   (ARCH = x86_64 | aarch64)
#
# Usage (from repo root):
#   scripts/release-macos.sh [VERSION] [--no-upload]
#
# VERSION defaults to the VERSION in install.sh. If a release for tag
# v<VERSION> exists, the tarball is uploaded via `gh release upload`; otherwise
# the tarball path and the exact upload command are printed.
set -euo pipefail

NO_UPLOAD=0
VERSION=""
for a in "$@"; do
    case "$a" in
        --no-upload) NO_UPLOAD=1 ;;
        -*) echo "unknown flag: $a" >&2; exit 2 ;;
        *)  [ -z "$VERSION" ] && VERSION="$a" || { echo "unexpected arg: $a" >&2; exit 2; } ;;
    esac
done

# Run from repo root regardless of invocation directory.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ---- resolve version ------------------------------------------------------
if [ -z "$VERSION" ]; then
    VERSION="$(sed -n 's/^VERSION="\(.*\)"/\1/p' install.sh | head -1)"
fi
[ -n "$VERSION" ] || { echo "could not resolve VERSION" >&2; exit 1; }

# ---- normalize arch (match install.sh's mapping) --------------------------
raw_arch="$(uname -m)"
case "$raw_arch" in
    aarch64|arm64) ARCH="aarch64" ;;
    x86_64|amd64)  ARCH="x86_64"  ;;
    *) echo "unsupported arch '$raw_arch'" >&2; exit 1 ;;
esac

echo ">> releasing sipllm $VERSION for darwin-$ARCH"

# ---- build portable binaries (no -march=native) ---------------------------
# ARCHFLAGS="" is REQUIRED: it prevents baking in -march=native, which would
# SIGILL when the bundle runs on a different Mac.
echo ">> building (ARCHFLAGS='')"
make -j ARCHFLAGS="" all
make ARCHFLAGS="" server

echo ">> testing"
make ARCHFLAGS="" test

# ---- sanity run -----------------------------------------------------------
echo ">> sanity run"
./build/make_toy_model /tmp/sipllm_rel_toy.gguf --gguf
./build/llm /tmp/sipllm_rel_toy.gguf -p hello -n 4 --greedy

# ---- stage bundle (mirrors release.yml "Stage bundle" step) ---------------
DIST="sipllm-${VERSION}-darwin-${ARCH}"
echo ">> staging $DIST"
rm -rf "$DIST" "$DIST.tar.gz"
mkdir "$DIST"
cp sipllm            "$DIST/sipllm"
cp build/llm         "$DIST/llm-engine"      # CLI looks for llm-engine
cp build/llm_server  "$DIST/llm_server"      2>/dev/null || true
for t in dump_logits bench inspect_gguf make_toy_model gguf_to_f16; do
    cp "build/$t" "$DIST/$t" 2>/dev/null || true
done
cp LICENSE README.md "$DIST/" 2>/dev/null || true
chmod +x "$DIST"/sipllm "$DIST"/llm-engine
tar -czf "$DIST.tar.gz" "$DIST"
ls -la "$DIST.tar.gz"

# ---- upload ---------------------------------------------------------------
if [ "$NO_UPLOAD" = "1" ]; then
    echo ">> --no-upload: stopping after producing $ROOT/$DIST.tar.gz"
    exit 0
fi

if command -v gh >/dev/null 2>&1 && gh release view "v$VERSION" >/dev/null 2>&1; then
    echo ">> uploading to release v$VERSION"
    gh release upload "v$VERSION" "$DIST.tar.gz" --clobber
    echo ">> uploaded $DIST.tar.gz to release v$VERSION"
else
    echo ">> no release v$VERSION found (or gh unavailable)."
    echo ">> tarball ready: $ROOT/$DIST.tar.gz"
    echo ">> to upload once the release exists, run:"
    echo "   gh release upload \"v$VERSION\" \"$DIST.tar.gz\" --clobber"
fi
