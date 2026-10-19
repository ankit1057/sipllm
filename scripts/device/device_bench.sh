#!/usr/bin/env bash
# device_bench.sh — host-side driver for the on-device matrix (adb shell only; no app).
# Prereq: adb push build-android/llm + the model to $DEV (see scripts/device/README.md).
# (forensic capture per run), pulls out/ into the current evidence dir, prints a TSV.
# Phone stays usable: each run is nice 10, nothing installed, no UI touched.
set -u
cd "$(dirname "$0")/../.."
# Evidence directory (gitignored by default under bench/device/). Every raw per-run
# capture is pulled here; only the aggregated JSON is meant to be committed.
D=${EVIDENCE_DIR:-bench/device/$(date +%Y-%m-%d_%H%M)}; mkdir -p "$D/runs"
echo "evidence → $D"
adb push scripts/device/device_run.sh "$DEV/device_run.sh" >/dev/null 2>&1 || true
DEV=${DEV:-/data/local/tmp/sipllm}
M=$DEV/tinyllama-q4_k_m.gguf
N=${N:-3}; NGEN=${NGEN:-32}; SETTLE=${SETTLE:-8}
P_PLAIN="Once upon a time"
P_REP="The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox"
OUT="$D/device-bench${PASS:+-$PASS}.tsv"
printf 'tag\tthreads\tbudget\tfast\tspec\tprompt\tdecode_med\tdecode_all\tprefill\tttft_s\tVmHWM_MB\twall_ms\tpinned\tspec_stats\n' > "$OUT"
med() { sort -n | awk '{a[NR]=$1} END{if(NR==0){print 0}else if(NR%2){print a[(NR+1)/2]}else{print (a[NR/2]+a[NR/2+1])/2}}'; }
run_cfg() {  # tag threads budget fast spec which
  local tag=$1 th=$2 budget=$3 fast=$4 spec=$5 which=$6
  local P; [ "$which" = plain ] && P="$P_PLAIN" || P="$P_REP"
  local args=(-p "$P" -n "$NGEN" --greedy --ctx 512 --threads "$th" --ram-budget "$budget")
  [ "$fast" = 1 ] && args+=(--fast); [ "$spec" != 0 ] && args+=(--spec "$spec")
  local decs=() line pf tt hwm wall pinned ss
  for i in $(seq 1 "$N"); do
    sleep "$SETTLE"
    line=$(adb shell "$DEV/device_run.sh ${tag}_r$i $(printf '%q ' "$M" "${args[@]}")" 2>&1 | tail -1)
    decs+=("$(echo "$line" | grep -o 'DECODE=[0-9.]*' | cut -d= -f2)")
    pf=$(echo "$line" | grep -o 'PREFILL=[0-9.]*' | cut -d= -f2)
    tt=$(echo "$line" | grep -o 'TTFT=[0-9.]*' | cut -d= -f2)
    hwm=$(echo "$line" | grep -o 'VmHWM_kB=[0-9]*' | cut -d= -f2)
    wall=$(echo "$line" | grep -o 'wall_ms=[0-9]*' | cut -d= -f2)
    pinned=$(echo "$line" | grep -o 'PINNED=[0-9/]*' | cut -d= -f2)
    ss=$(echo "$line" | grep -o 'SPEC=\[[^]]*\]' | sed 's/SPEC=\[//; s/\]//')
  done
  local m; m=$(printf '%s\n' "${decs[@]}" | med)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%.1f\t%s\t%s\t%s\n' "$tag" "$th" "$budget" "$fast" "$spec" "$which" "$m" "$(IFS=,; echo "${decs[*]}")" "${pf:-}" "${tt:-}" "$(awk -v k="${hwm:-0}" 'BEGIN{print k/1024}')" "${wall:-}" "${pinned:-}" "${ss:-}" | tee -a "$OUT"
}
SECTIONS=${SECTIONS:-threads budget spec}
BEST=${BEST_THREADS:-4}
case " $SECTIONS " in *" threads "*)
  for th in 2 4 8; do run_cfg "thr${th}_res_fast" "$th" 1G 1 0 plain; done;; esac
case " $SECTIONS " in *" budget "*)
  for budget in 1G 512M 256M 0; do for fast in 0 1; do run_cfg "b${budget}_f${fast}_t${BEST}" "$BEST" "$budget" "$fast" 0 plain; done; done;; esac
case " $SECTIONS " in *" spec "*)
  for budget in 0 1G; do for spec in 0 5; do run_cfg "spec${spec}_b${budget}_t${BEST}" "$BEST" "$budget" 1 "$spec" rep; done; done;; esac
case " $SECTIONS " in *" key8 "*)
  run_cfg "b1G_f1_t8" 8 1G 1 0 plain; run_cfg "b0_f1_t8" 8 0 1 0 plain; run_cfg "spec5_b0_t8" 8 0 1 5 rep;; esac
adb pull "$DEV/out" "$D/runs/" >/dev/null 2>&1 && echo "pulled raw runs → $D/runs/out"
echo "wrote $OUT"
