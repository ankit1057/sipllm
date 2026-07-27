#!/usr/bin/env bash
# deploy_android.sh — install and launch SipLLM Studio on connected Android device.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

APK="$ROOT/sipllm-studio-universal-v1.0.apk"
if [ ! -f "$APK" ]; then
    echo ">> Root APK not found ($APK). Building now..."
    "$SCRIPT_DIR/build_app.sh"
fi

DEVICE="$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')"
if [ -z "$DEVICE" ]; then
    echo "error: No connected Android device found via ADB." >&2
    exit 1
fi

echo ">> Deploying to connected Android device: $DEVICE"
adb -s "$DEVICE" install -r "$APK"

echo ">> Launching SipLLM Studio app on device..."
adb -s "$DEVICE" shell am start -n com.example.local_llm_chat/.MainActivity

echo ">> SipLLM Studio deployed and launched successfully on $DEVICE!"
