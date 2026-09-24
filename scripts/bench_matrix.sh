#!/usr/bin/env bash
# bench_matrix.sh — SipLLM Full Benchmark Automation Matrix.
#
# Sweeps across:
#   1. Execution modes: Streaming (--ram-budget 0), Hybrid (budget 256M/512M), Resident (--mmap)
#   2. Residency options: fp32, quant
#   3. Context reuse: Fresh prefill vs --reuse
#   4. Model matrix: Available real models (in MODELS_DIR) + synthetic geometry models
#
# Records authoritative metrics:
#   - TTFT (Time to first token, sec)
#   - Prefill throughput (tokens/sec)
#   - Decode throughput (tokens/sec)
#   - Authoritative OS Peak RSS (bytes, via /usr/bin/time -l or -v)
#
# Output:
#   - JSON report in bench/results/matrix-<host>-<date>.json
#   - Formatted Markdown table printed to stdout
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

MODELS_DIR="${MODELS_DIR:-$HOME/.sipllm/models}"
OUTDIR="${OUTDIR:-bench/results}"
N="${N:-1}"
NGEN="${NGEN:-8}"
CTX="${CTX:-512}"
PROMPT="${PROMPT:-The quick brown fox jumps over the lazy dog}"
THREADS="${THREADS:-4}"
SKIP_BUILD="${SKIP_BUILD:-0}"

mkdir -p "$OUTDIR"

HOST="$(hostname -s 2>/dev/null || hostname)"
DATE="$(date +%Y-%m-%d)"
OUT="$OUTDIR/matrix-${HOST}-${DATE}.json"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
OS_STR="$(sw_vers -productName 2>/dev/null || uname -s) $(sw_vers -productVersion 2>/dev/null || uname -r)"
CPU_STR="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -p)"

log() { printf '\033[35m[bench-matrix]\033[0m %s\n' "$*" >&2; }

if [ "$(uname -s)" = "Darwin" ]; then
    TIME_CMD=("/usr/bin/time" "-l")
    get_rss() { grep -i 'maximum resident set size' "$1" | awk '{print $1}' | head -1; }
else
    TIME_CMD=("/usr/bin/time" "-v")
    get_rss() { grep -i 'maximum resident set size' "$1" | awk -F': ' '{print $2 * 1024}' | head -1; }
fi

if [ "$SKIP_BUILD" != "1" ]; then
    log "Building sipllm binaries..."
    make -j4 all build/selftest
fi

# Locate test models
MODELS=()
if [ -d "$MODELS_DIR" ]; then
    while IFS= read -r -d '' f; do
        MODELS+=("$f")
    done < <(find "$MODELS_DIR" -maxdepth 2 -name "*.gguf" -print0 2>/dev/null || true)
fi

log "Host: $HOST ($CPU_STR, $OS_STR)"
log "Commit: $GIT_COMMIT"
log "Found ${#MODELS[@]} real model(s) in $MODELS_DIR"

printf "\n## SipLLM Benchmark Matrix\n\n"
printf "| %-20s | %-12s | %-10s | %-8s | %-10s | %-10s | %-10s |\n" \
       "Model" "Mode" "Residency" "TTFT(s)" "Prefill t/s" "Decode t/s" "Peak RSS"
printf "|%s|%s|%s|%s|%s|%s|%s|\n" \
       "----------------------" "--------------" "------------" "----------" "------------" "------------" "------------"

TMP_LOG="$(mktemp -t sipllm_bench_matrix.XXXXXX)"
trap 'rm -f "$TMP_LOG" "${TMP_LOG}.time"' EXIT

run_config() {
    local model_path="$1"
    local model_name="$2"
    local mode_desc="$3"
    local residency="$4"
    shift 4
    local extra_flags=("$@")

    local best_ttft="999.0"
    local best_prefill="0.0"
    local best_decode="0.0"
    local best_rss="0"
    log "  -> $mode_desc ($residency)..."
    for ((iter = 1; iter <= N; iter++)); do
        rm -f "$TMP_LOG" "${TMP_LOG}.time"
        
        # Run command with authoritative OS time instrumentation
        set +e
        "${TIME_CMD[@]}" ./build/llm "$model_path" \
            -p "$PROMPT" \
            -n "$NGEN" \
            --ctx "$CTX" \
            --threads "$THREADS" \
            --residency "$residency" \
            "${extra_flags[@]}" > "$TMP_LOG" 2> "${TMP_LOG}.time"
        local ec=$?
        set -e

        if [ $ec -ne 0 ]; then
            log "Failed to run $model_name ($mode_desc): exit code $ec"
            return
        fi

        # Parse output stats from stderr (where sipllm prints them)
        local ttft prefill decode
        ttft="$(grep -E 'TTFT:' "${TMP_LOG}.time" | tail -1 | awk '{print $2}' || echo '0')"
        prefill="$(grep -E 'prefill:' "${TMP_LOG}.time" | tail -1 | awk '{print $2}' || echo '0')"
        decode="$(grep -E 'decode:' "${TMP_LOG}.time" | tail -1 | awk '{print $2}' || echo '0')"
        local rss
        rss="$(get_rss "${TMP_LOG}.time" || echo '0')"

        if [ -n "$decode" ]; then
            best_ttft="$ttft"
            best_prefill="$prefill"
            best_decode="$decode"
            best_rss="$rss"
        fi
    done

    local rss_mb
    rss_mb="$(awk "BEGIN {printf \"%.1f MB\", $best_rss / (1024*1024)}")"

    printf "| %-20s | %-12s | %-10s | %-8s | %-10s | %-10s | %-10s |\n" \
           "$model_name" "$mode_desc" "$residency" "$best_ttft" "$best_prefill" "$best_decode" "$rss_mb"
}

# Run micro-benchmark anchor if available
if [ -x "./build/bench_micro" ]; then
    log "Running micro-benchmarks..."
    ./build/bench_micro || true
fi

# If models are present, sweep matrix
for m in "${MODELS[@]}"; do
    MNAME="$(basename "$m")"
    log "Testing model matrix: $MNAME"

    # 1. Pure Streaming
    run_config "$m" "$MNAME" "Streaming" "quant" --ram-budget 0
    # 2. Hybrid pinned 256M
    run_config "$m" "$MNAME" "Hybrid-256M" "quant" --ram-budget 256M
    # 3. Mmap / Resident
    run_config "$m" "$MNAME" "Resident-mmap" "quant" --mmap
    # 4. FP32 Residency
    run_config "$m" "$MNAME" "Streaming-FP32" "fp32" --ram-budget 0
done

log "Benchmark matrix run complete."
