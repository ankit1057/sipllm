#!/usr/bin/env bash
# build_all.sh — master build script for SipLLM engine, tests, bindings, app & docs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

echo "=========================================================="
echo " SipLLM Unified Build Pipeline"
echo "=========================================================="

echo ">> [1/6] Building C++ engine binaries & tools (make all)..."
make OPT="-O3" all

echo ">> [2/6] Running full C++ unit test suite (make test)..."
make test

echo ">> [3/6] Building FFI bindings & Flutter Studio app..."
"$SCRIPT_DIR/build_app.sh"

echo ">> [4/6] Running benchmark suite..."
python3 "$SCRIPT_DIR/run_benchmarks.py" --ci

echo ">> [5/6] Generating updated benchmark report..."
python3 "$SCRIPT_DIR/bench_report.py"

echo ">> [6/6] Building static documentation portal site..."
(cd docs && python3 build.py)

echo "=========================================================="
echo " Unified Build Completed Successfully!"
echo "=========================================================="
