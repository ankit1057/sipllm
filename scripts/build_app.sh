#!/usr/bin/env bash
# build_app.sh — build SipLLM Studio app, FFI bindings, and package APK.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "== [1/3] Building native desktop FFI library =="
(cd bindings/flutter/sipllm_flutter/ffi && ./build_desktop.sh)

echo "== [2/3] Building SipLLM Studio Flutter APK (Release) =="
(cd bindings/flutter/sipllm_flutter/example && flutter build apk --release)

echo "== [3/3] Packaging root distribution APK =="
cp bindings/flutter/sipllm_flutter/example/build/app/outputs/flutter-apk/app-release.apk "$ROOT/sipllm-studio-universal-v1.0.apk"

echo ">> App build complete!"
echo ">> Root APK: $ROOT/sipllm-studio-universal-v1.0.apk"
