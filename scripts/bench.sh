#!/usr/bin/env bash
# bench.sh — SipLLM reproducible benchmark harness (wave 2, v0.2 baseline).
#
# Runs median-of-N generation benchmarks for every cached model plus a set of
# toy models of IDENTICAL geometry at increasing depth (4/16/32 layers) to prove
# the core "bounded peak RSS" thesis: peak resident memory stays ~flat as depth
# (and therefore on-disk model size) grows, because layers are streamed.
#
# The authoritative peak-RSS number for EVERY runtime comes from the OS via
#   /usr/bin/time -l    ("maximum resident set size", in bytes on macOS)
# so it is directly comparable across SipLLM and llama.cpp (whose internal
# instrumentation differs). SipLLM's own internal peak_rss is 0 on macOS
# (current_rss_bytes() reads Linux /proc/self/statm), so we never rely on it.
#
# Output: one machine-readable JSON document at
#   bench/results/<host>-<YYYY-MM-DD>.json
# consumed by scripts/bench_report.py.
#
# ---------------------------------------------------------------------------
# Reproduce (single command, from repo root):
#   ./scripts/bench.sh
# Knobs (env vars, all optional):
#   N=5              median-of-N repetitions per data point
#   NGEN=16          tokens to generate per run
#   PROMPT="..."     prompt (default "Once upon a time")
#   THREADS=4        thread count, pinned identically for SipLLM and llama.cpp
#   MODELS_DIR=~/.sipllm/models
#   LLAMA_BIN=<dir>  directory containing llama-cli/llama-bench (cross-runtime
#                    baseline). Auto-detected at /tmp/llama.cpp/build/bin and on
#                    PATH; if absent, the llama.cpp section is skipped + noted.
#   PURGE=1          attempt a cold-cache run via `sudo purge` (needs passwordless
#                    sudo; if unavailable the harness reports WARM cache and says so).
#   SKIP_BUILD=1     reuse existing build/ without invoking make.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- locate repo root (script lives in <root>/scripts) ----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

# --- configuration ----------------------------------------------------------
N="${N:-5}"
NGEN="${NGEN:-16}"
PROMPT="${PROMPT:-Once upon a time}"
THREADS="${THREADS:-4}"
MODELS_DIR="${MODELS_DIR:-$HOME/.sipllm/models}"
OUTDIR="${OUTDIR:-bench/results}"
PURGE="${PURGE:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

HOST="$(hostname -s 2>/dev/null || hostname)"
DATE="$(date +%Y-%m-%d)"
OUT="$OUTDIR/${HOST}-${DATE}.json"

# Host facts (macOS-first, with graceful fallbacks).
OS_STR="$(sw_vers -productName 2>/dev/null || uname -s) $(sw_vers -productVersion 2>/dev/null || uname -r)"
CPU_STR="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -p)"
CORES="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo '?')"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"

# --- llama.cpp discovery ----------------------------------------------------
LLAMA_BIN="${LLAMA_BIN:-}"
if [ -z "$LLAMA_BIN" ]; then
  for cand in /tmp/llama.cpp/build/bin "$HOME/llama.cpp/build/bin" /opt/homebrew/bin; do
    if [ -x "$cand/llama-cli" ] || [ -x "$cand/llama-bench" ]; then LLAMA_BIN="$cand"; break; fi
  done
fi
LLAMA_CLI=""; LLAMA_BENCH=""
[ -n "$LLAMA_BIN" ] && [ -x "$LLAMA_BIN/llama-cli" ]   && LLAMA_CLI="$LLAMA_BIN/llama-cli"
[ -n "$LLAMA_BIN" ] && [ -x "$LLAMA_BIN/llama-bench" ] && LLAMA_BENCH="$LLAMA_BIN/llama-bench"

# --- working area -----------------------------------------------------------
WORK="$(mktemp -d "${TMPDIR:-/tmp}/sipllm_bench.XXXXXX")"
TOYDIR="$WORK/toy"
mkdir -p "$TOYDIR" "$OUTDIR"
MANIFEST="$WORK/manifest.tsv"
: > "$MANIFEST"
trap 'rm -rf "$WORK"' EXIT

log() { printf '\033[36m[bench]\033[0m %s\n' "$*" >&2; }

# --- build ------------------------------------------------------------------
if [ "$SKIP_BUILD" != "1" ]; then
  log "building (make OPT=\"-O3\" all)"
  make OPT="-O3" all >/dev/null
else
  log "SKIP_BUILD=1 — reusing existing build/"
fi
[ -x build/llm ] || { echo "FATAL: build/llm missing" >&2; exit 1; }
[ -x build/make_toy_model ] || { echo "FATAL: build/make_toy_model missing" >&2; exit 1; }

# --- cache control ----------------------------------------------------------
CACHE_STATE="warm"
CACHE_NOTE="warm page cache (models read at least once before timing)"
maybe_purge() {
  if [ "$PURGE" = "1" ]; then
    if sudo -n true 2>/dev/null; then
      sudo purge && CACHE_STATE="cold" \
        && CACHE_NOTE="cold cache: 'sudo purge' issued before each run"
    else
      CACHE_NOTE="cold cache requested but sudo unavailable (passwordless sudo required for 'purge'); reporting WARM only"
      log "$CACHE_NOTE"
    fi
  fi
}
maybe_purge

# --- power (needs sudo) -----------------------------------------------------
POWER_NOTE="N/A (sudo unavailable — 'sudo powermetrics' requires privileges; never fabricated)"
if sudo -n true 2>/dev/null; then
  POWER_NOTE="powermetrics available (sudo ok); per-run CPU power captured where sampled"
fi

# --- record one raw run into the manifest -----------------------------------
# args: section model quant backend layers runtime ngen  cmd...
record() {
  local section="$1" model="$2" quant="$3" backend="$4" layers="$5" runtime="$6" ngen="$7" nthreads="$8"
  shift 8
  local i raw
  for i in $(seq 1 "$N"); do
    raw="$WORK/${section}_${runtime}_${layers}_$(basename "$model")_b${backend}_t${nthreads}_r${i}.txt"
    # Merge stdout+stderr: /usr/bin/time -l and SipLLM stats print to stderr,
    # while llama-cli's "[ Prompt: .. | Generation: .. ]" summary and generated
    # tokens print to stdout. Capturing both keeps every runtime parseable.
    { /usr/bin/time -l "$@"; } >"$raw" 2>&1 || {
      log "WARN: run failed (see $raw)"; head -5 "$raw" >&2 || true; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$section" "$model" "$quant" "$backend" "$layers" "$runtime" "$ngen" "$nthreads" "$raw" >> "$MANIFEST"
    if [ "$CACHE_STATE" = "cold" ]; then maybe_purge; fi
  done
}

# --- SipLLM: real models ----------------------------------------------------
declare -a REAL_MODELS REAL_QUANT
[ -f "$MODELS_DIR/smollm2-135m.gguf" ]     && { REAL_MODELS+=("$MODELS_DIR/smollm2-135m.gguf");     REAL_QUANT+=("Q8_0"); }
[ -f "$MODELS_DIR/tinyllama-q4_k_m.gguf" ] && { REAL_MODELS+=("$MODELS_DIR/tinyllama-q4_k_m.gguf"); REAL_QUANT+=("Q4_K_M"); }

for idx in "${!REAL_MODELS[@]}"; do
  m="${REAL_MODELS[$idx]}"; q="${REAL_QUANT[$idx]}"
  log "SipLLM: $(basename "$m") x$N (pread)"
  record "real" "$m" "$q" "pread" "0" "sipllm" "$NGEN" "$THREADS" \
    ./build/llm "$m" -p "$PROMPT" -n "$NGEN" --threads "$THREADS" --greedy
done

# --- SipLLM: toy scaling (identical geometry, increasing depth) -------------
# The core "bounded RSS" experiment. Fixed layer geometry (dim/heads/kv/ffn/
# vocab); ONLY --layers varies, so on-disk size grows linearly with depth. We
# sweep depth at TWO context lengths to separate two effects:
#   * weights_resident MB — must stay FLAT vs depth (streaming keeps ~1 layer
#     resident regardless of how many layers the model has). This is the thesis.
#   * peak RSS — flat at small ctx; its only growth-vs-depth term is the dense
#     fp32 KV cache (O(layers x ctx)), a separately-tracked leak, NOT weights.
# Running ctx=64 (KV negligible) and ctx=512 (KV dominant) exposes both cleanly.
TOY_GEOM=(--dim 256 --heads 8 --kv 8 --ffn 512 --vocab 512)
for C in 64 512; do
  for L in 4 16 32; do
    toy="$TOYDIR/toy_${L}L_c${C}.gguf"
    ./build/make_toy_model "$toy" --gguf --q8 --layers "$L" --ctx "$C" "${TOY_GEOM[@]}" >/dev/null
    log "SipLLM toy: ${L} layers ctx=${C} x$N"
    record "toy" "$toy" "Q8_0" "pread" "$L" "sipllm" "$NGEN" "$THREADS" \
      ./build/llm "$toy" -p "a b c" -n "$NGEN" --threads "$THREADS" --greedy
  done
done

# --- SipLLM: thread sweep (over-subscription evidence) ----------------------
# Small batch=1 GEMVs over-subscribe on M3 (4 P + 4 E cores). Sweep threads on
# the fast model to expose the tuned thread count in the baseline.
SWEEP="${SWEEP:-1}"
SWEEP_MODEL="$MODELS_DIR/smollm2-135m.gguf"
if [ "$SWEEP" = "1" ] && [ -f "$SWEEP_MODEL" ]; then
  for T in 1 2 4 8; do
    log "SipLLM thread sweep: smollm2 t=$T x$N"
    record "sweep" "$SWEEP_MODEL" "Q8_0" "pread" "0" "sipllm" "$NGEN" "$T" \
      ./build/llm "$SWEEP_MODEL" -p "$PROMPT" -n "$NGEN" --threads "$T" --greedy
  done
fi

# --- llama.cpp cross-runtime baseline ---------------------------------------
LLAMA_NOTE="llama.cpp NOT found — cross-runtime section skipped. Build it with: \
git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp && \
cmake -B /tmp/llama.cpp/build -S /tmp/llama.cpp -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && \
cmake --build /tmp/llama.cpp/build -j --target llama-cli llama-bench, then re-run with LLAMA_BIN=/tmp/llama.cpp/build/bin"
if [ -n "$LLAMA_CLI" ] || [ -n "$LLAMA_BENCH" ]; then
  LLAMA_VER="$("${LLAMA_CLI:-$LLAMA_BENCH}" --version 2>&1 | head -1 || echo unknown)"
  LLAMA_NOTE="llama.cpp @ $LLAMA_BIN ($LLAMA_VER); CPU-only (-ngl 0) for a like-for-like comparison vs SipLLM (CPU streaming, no GPU)."
  for idx in "${!REAL_MODELS[@]}"; do
    m="${REAL_MODELS[$idx]}"; q="${REAL_QUANT[$idx]}"
    if [ -n "$LLAMA_CLI" ]; then
      log "llama-cli: $(basename "$m") x$N"
      record "real" "$m" "$q" "pread" "0" "llama_cli" "$NGEN" "$THREADS" \
        "$LLAMA_CLI" -m "$m" -p "$PROMPT" -n "$NGEN" -t "$THREADS" -ngl 0 \
        --no-warmup -st --seed 1
    fi
    if [ -n "$LLAMA_BENCH" ]; then
      log "llama-bench: $(basename "$m") (internal -r $N)"
      raw="$WORK/bench_llama_${idx}.txt"
      { /usr/bin/time -l "$LLAMA_BENCH" -m "$m" -t "$THREADS" -ngl 0 \
          -p 5 -n "$NGEN" -r "$N" >"$raw"; } 2>>"$raw" || true
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "real" "$m" "$q" "pread" "0" "llama_bench" "$NGEN" "$THREADS" "$raw" >> "$MANIFEST"
    fi
  done
else
  log "$LLAMA_NOTE"
fi

# --- assemble JSON (parse raw captures, compute medians) --------------------
log "assembling $OUT"
MANIFEST="$MANIFEST" OUT="$OUT" \
HOST="$HOST" DATE="$DATE" OS_STR="$OS_STR" CPU_STR="$CPU_STR" CORES="$CORES" \
THREADS="$THREADS" NGEN="$NGEN" PROMPT="$PROMPT" NREP="$N" \
CACHE_STATE="$CACHE_STATE" CACHE_NOTE="$CACHE_NOTE" POWER_NOTE="$POWER_NOTE" \
LLAMA_NOTE="$LLAMA_NOTE" GIT_COMMIT="$GIT_COMMIT" MODELS_DIR="$MODELS_DIR" \
python3 "$SCRIPT_DIR/_assemble.py"

log "done -> $OUT"
echo "$OUT"
