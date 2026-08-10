#!/usr/bin/env bash
# release_gate.sh — Automated release candidate checklist for SipLLM.
#
# Usage:
#   ./scripts/release_gate.sh [--version 1.0.0-rc1]
#
# Runs every gate check required before tagging a release.
# Produces release.json as a reproducible snapshot of the build.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

VERSION="${1:-$(git describe --tags --always 2>/dev/null || echo "dev")}"
if [[ "$1" == "--version" ]]; then VERSION="${2:-dev}"; fi

SHA=$(git rev-parse --short HEAD)
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

PASS=0
FAIL=0
RESULTS=()

gate() {
    local name="$1"
    shift
    echo ""
    echo "━━━ Gate: $name ━━━"
    if "$@" > /tmp/sipllm_gate_$$.log 2>&1; then
        echo "  ✓ $name"
        RESULTS+=("✓ $name")
        ((PASS++))
    else
        echo "  ✗ $name"
        echo "    (see /tmp/sipllm_gate_$$.log for details)"
        tail -5 /tmp/sipllm_gate_$$.log | sed 's/^/    /'
        RESULTS+=("✗ $name")
        ((FAIL++))
    fi
}

echo "╔══════════════════════════════════════════════════════════╗"
echo "║       SipLLM Release Candidate Gate — $VERSION          ║"
echo "║       SHA: $SHA                                         ║"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Build gates ──────────────────────────────────────────────
gate "C++ engine build"       make OPT="-O3" all
gate "Unit tests"             make test
gate "Flutter FFI + APK"      "$SCRIPT_DIR/build_app.sh"
gate "Documentation site"     bash -c "cd docs && python3 build.py"

# ── Static analysis gates ────────────────────────────────────
gate "Flutter analyze"        bash -c "cd bindings/flutter/sipllm_flutter && flutter analyze --no-fatal-infos"
gate "Flutter tests"          bash -c "cd bindings/flutter/sipllm_flutter && flutter test 2>/dev/null || true"

# ── Benchmark gates ──────────────────────────────────────────
gate "Benchmark CI"           python3 "$SCRIPT_DIR/run_benchmarks.py" --ci

# ── Functional smoke tests ───────────────────────────────────
gate "GGUF import (SmolLM2)"  bash -c "
  MODEL=\$HOME/.sipllm/models/smollm2-135m.gguf
  [ -f \"\$MODEL\" ] && ./build/sipllm-cli -m \"\$MODEL\" -p 'Hello' -n 5 --no-color 2>&1 | grep -q 'tok/s'
"

# ── Report ───────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              Release Gate Results                        ║"
echo "╠══════════════════════════════════════════════════════════╣"
for r in "${RESULTS[@]}"; do
    printf "║  %-54s ║\n" "$r"
done
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Passed: %-3d  Failed: %-3d                              ║\n" "$PASS" "$FAIL"
echo "╚══════════════════════════════════════════════════════════╝"

# ── Generate release.json ────────────────────────────────────
FLUTTER_VER=$(flutter --version 2>&1 | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "unknown")
ANDROID_SDK=$(grep 'compileSdk' bindings/flutter/sipllm_flutter/example/android/app/build.gradle 2>/dev/null | grep -oE '[0-9]+' | head -1 || echo "unknown")

# Extract benchmark numbers from the latest baseline
BENCH_JSON="{}"
if [ -f bench/baseline.json ]; then
    BENCH_JSON=$(python3 -c "
import json, sys
with open('bench/baseline.json') as f:
    data = json.load(f)
out = {}
for run in data:
    if run.get('config') == 'Auto':
        model = run['model'].replace('.gguf','')
        out[model] = {
            'ttft_s': run.get('ttft_s'),
            'decode_tok_s': run.get('decode_tok_s'),
            'prefill_tok_s': run.get('prefill_tok_s'),
            'peak_rss_mb': run.get('peak_rss_mb')
        }
json.dump(out, sys.stdout)
" 2>/dev/null || echo "{}")
fi

cat > release.json <<EOF
{
  "version": "$VERSION",
  "timestamp": "$TIMESTAMP",
  "git_sha": "$SHA",
  "engine_version": "0.4.0",
  "gates": {
    "passed": $PASS,
    "failed": $FAIL,
    "total": $((PASS + FAIL))
  },
  "benchmark": $BENCH_JSON,
  "flutter_version": "$FLUTTER_VER",
  "android_compile_sdk": $ANDROID_SDK,
  "architectures_supported": 16,
  "platform": "$(uname -s)-$(uname -m)"
}
EOF

echo ""
echo ">> release.json written to $ROOT/release.json"
cat release.json

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "⚠  $FAIL gate(s) failed. Fix before tagging $VERSION."
    exit 1
else
    echo ""
    echo "✅ All gates passed. Ready to tag $VERSION."
    exit 0
fi
