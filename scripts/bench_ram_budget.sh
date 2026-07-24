#!/usr/bin/env bash
# bench_ram_budget.sh — the #37 "RAM-speed dial" chart.
#
# Sweeps --ram-budget for each cached real model and records, at each budget,
# the decode throughput and the AUTHORITATIVE peak RSS (from the OS via
# `/usr/bin/time -l`, "maximum resident set size" in bytes on macOS), plus the
# number of layers the loader pinned resident. This is the headline
# "bounded peak RSS regardless of model size, with a speed dial" evidence:
# at budget 0 the model streams (smallest RSS, slowest decode); as the budget
# rises the loader pins more hot layers (larger RSS, faster decode) until the
# whole model is resident.
#
# Output: bench/results/ram-budget-<host>-<YYYY-MM-DD>.json + a markdown table
# on stdout. The cross-runtime llama.cpp anchor lives in the main harness
# (scripts/bench.sh); this script isolates the SipLLM dial.
#
# Reproduce (from repo root):
#   ./scripts/bench_ram_budget.sh
# Knobs (env, all optional):
#   N=3                 median-of-N repetitions per data point
#   NGEN=24             tokens generated per run
#   CTX=512             context window (kept small so KV does not dominate RSS)
#   PROMPT="..."        prompt (default "Once upon a time")
#   THREADS=4           thread count
#   BUDGETS="0 128M ..."space-separated --ram-budget values (0 = unlimited stream)
#   MODELS_DIR=~/.sipllm/models
#   SKIP_BUILD=1        reuse build/ without invoking make
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

N="${N:-3}"
NGEN="${NGEN:-24}"
CTX="${CTX:-512}"
PROMPT="${PROMPT:-Once upon a time}"
THREADS="${THREADS:-4}"
BUDGETS="${BUDGETS:-0 128M 256M 384M 512M 768M 1200M}"
MODELS_DIR="${MODELS_DIR:-$HOME/.sipllm/models}"
SKIP_BUILD="${SKIP_BUILD:-0}"
SETTLE="${SETTLE:-4}"   # seconds idle between data points (fanless thermal settle)
OUTDIR="bench/results"

HOST="$(hostname -s 2>/dev/null || hostname)"
DATE="$(date +%Y-%m-%d)"
OUT="$OUTDIR/ram-budget-${HOST}-${DATE}.json"
GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
OS_STR="$(sw_vers -productName 2>/dev/null || uname -s) $(sw_vers -productVersion 2>/dev/null || uname -r)"
CPU_STR="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || uname -p)"

log() { printf '\033[36m[rb-bench]\033[0m %s\n' "$*" >&2; }

if [ "$(uname -s)" != "Darwin" ]; then
    log "note: peak RSS parsing targets macOS '/usr/bin/time -l'. On Linux use GNU time -v and adjust RE_MAXRSS."
fi

[ "$SKIP_BUILD" = "1" ] || make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" build/llm >/dev/null
[ -x build/llm ] || { echo "FATAL: build/llm missing" >&2; exit 1; }
mkdir -p "$OUTDIR"

# Candidate models (name -> file), only those present are benched.
declare -a MODELS NAMES
add_model() { [ -f "$2" ] && { NAMES+=("$1"); MODELS+=("$2"); }; }
add_model smollm2-135m  "$MODELS_DIR/smollm2-135m.gguf"
add_model tinyllama     "$MODELS_DIR/tinyllama-q4_k_m.gguf"
[ "${#MODELS[@]}" -gt 0 ] || { echo "FATAL: no models in $MODELS_DIR" >&2; exit 1; }

median() { sort -n | awk '{a[NR]=$1} END{ if(NR==0){print 0} else if(NR%2){printf "%.2f",a[(NR+1)/2]} else {printf "%.2f",(a[NR/2]+a[NR/2+1])/2} }'; }

WORK="$(mktemp -d "${TMPDIR:-/tmp}/rb_bench.XXXXXX")"; trap 'rm -rf "$WORK"' EXIT
ROWS="$WORK/rows.tsv"; : > "$ROWS"

for mi in "${!MODELS[@]}"; do
    name="${NAMES[$mi]}"; path="${MODELS[$mi]}"
    log "warming $name"
    ./build/llm "$path" -p warm -n 1 --greedy --ctx "$CTX" >/dev/null 2>&1 || true
    for b in $BUDGETS; do
        budget_arg=(); [ "$b" != "0" ] && budget_arg=(--ram-budget "$b")
        log "$name  budget=$b  (x$N)"
        sleep "$SETTLE"
        decs="$WORK/dec"; : > "$decs"
        rss=""; pinned=""; wres=""; streamed=""
        for r in $(seq 1 "$N"); do
            cap="$WORK/cap"
            /usr/bin/time -l ./build/llm "$path" -p "$PROMPT" -n "$NGEN" --greedy \
                --ctx "$CTX" --threads "$THREADS" "${budget_arg[@]}" >/dev/null 2>"$cap" || true
            d="$(grep -E '^decode:' "$cap" | awk '{print $2}' | head -1)"
            [ -n "$d" ] && echo "$d" >> "$decs"
            # last run supplies the (deterministic) residency + RSS facts
            rss="$(grep 'maximum resident set size' "$cap" | awk '{print $1}' | head -1)"
            pinned="$(grep -E '^pinned layers:' "$cap" | awk '{print $3}' | head -1)"
            wres="$(grep -E '^weights resident:' "$cap" | sed -E 's/.*:([0-9.]+) MB.*/\1/' | head -1)"
            streamed="$(grep -E '^streamed:' "$cap" | sed -E 's/.*:[[:space:]]*([0-9.]+) MB.*/\1/' | head -1)"
        done
        dec_med="$(median < "$decs")"
        rss_mb="$(awk -v b="${rss:-0}" 'BEGIN{printf "%.1f", b/1e6}')"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$name" "$b" "${dec_med:-0}" "${rss_mb:-0}" "${pinned:-0}" "${wres:-0}" "${streamed:-0}" >> "$ROWS"
    done
done

# ---- emit JSON + markdown --------------------------------------------------
ROWS="$ROWS" OUT="$OUT" HOST="$HOST" DATE="$DATE" OS_STR="$OS_STR" CPU_STR="$CPU_STR" \
GIT_COMMIT="$GIT_COMMIT" NGEN="$NGEN" CTX="$CTX" PROMPT="$PROMPT" THREADS="$THREADS" NREP="$N" \
python3 - <<'PY'
import json, os
rows = []
with open(os.environ["ROWS"]) as f:
    for line in f:
        name, b, dec, rss, pinned, wres, streamed = line.rstrip("\n").split("\t")
        rows.append({
            "model": name,
            "ram_budget": b,                       # "0" = unlimited (pure stream)
            "decode_tok_s": float(dec),
            "peak_rss_mb": float(rss),
            "pinned_layers": int(pinned),
            "weights_resident_mb": float(wres),
            "streamed_mb": float(streamed),
        })
doc = {
    "meta": {
        "benchmark": "ram_budget_sweep (#37)",
        "host": os.environ["HOST"], "date": os.environ["DATE"],
        "os": os.environ["OS_STR"], "cpu": os.environ["CPU_STR"],
        "git_commit": os.environ["GIT_COMMIT"],
        "gen_tokens": int(os.environ["NGEN"]), "ctx": int(os.environ["CTX"]),
        "prompt": os.environ["PROMPT"], "threads": int(os.environ["THREADS"]),
        "median_of_n": int(os.environ["NREP"]), "cache_state": "warm",
        "peak_rss_source": "/usr/bin/time -l 'maximum resident set size' (bytes)",
        "reproduce": "./scripts/bench_ram_budget.sh",
    },
    "ram_budget_sweep": rows,
}
with open(os.environ["OUT"], "w") as f:
    json.dump(doc, f, indent=2); f.write("\n")

# markdown table to stdout
by = {}
for r in rows: by.setdefault(r["model"], []).append(r)
print()
for model, rs in by.items():
    print(f"### {model}")
    print("| --ram-budget | pinned | decode tok/s | weights MB | streamed MB | peak RSS MB |")
    print("|:-------------|-------:|-------------:|-----------:|------------:|------------:|")
    for r in rs:
        b = "0 (stream)" if r["ram_budget"] == "0" else r["ram_budget"]
        print(f"| {b} | {r['pinned_layers']} | {r['decode_tok_s']:.2f} | "
              f"{r['weights_resident_mb']:.1f} | {r['streamed_mb']:.1f} | {r['peak_rss_mb']:.1f} |")
    print()
print(f"wrote {os.environ['OUT']}")
PY
log "done -> $OUT"
