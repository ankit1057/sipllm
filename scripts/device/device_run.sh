#!/system/bin/sh
# device_run.sh — runs ONE sipllm invocation on the phone with forensic capture.
# usage: device_run.sh <tag> <llm args...>
# Writes out/<tag>.txt with: pre/post thermal+freq, VmHWM (kernel high-water RSS,
# sampled 5x/s until exit, monotonic so the last sample is the peak), wall time,
# and the engine's own stats. Runs at nice 10 so the UI stays responsive.
D=/data/local/tmp/sipllm
tag=$1; shift
out=$D/out/$tag.txt
{
  echo "### tag=$tag"; echo "### date=$(date '+%Y-%m-%d %H:%M:%S')"; echo "### args: $*"
  echo "### pre: battery_temp_dC=$(dumpsys battery | grep temperature | awk '{print $2}') level=$(dumpsys battery | grep level | awk '{print $2}')"
  echo "### pre: cpufreq cur (policy0 little / policy6 big): $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq) $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq)"
  echo "### pre: MemAvailable=$(grep MemAvailable /proc/meminfo | awk '{print $2}') kB"
  echo "### run"
  t0=$(cut -d' ' -f1 /proc/uptime | tr -d .)
  nice -n 10 $D/llm "$@" > $D/out/$tag.stdout 2> $D/out/$tag.stderr &
  pid=$!
  hwm=0
  while kill -0 $pid 2>/dev/null; do
    h=$(grep VmHWM /proc/$pid/status 2>/dev/null | awk '{print $2}')
    [ -n "$h" ] && hwm=$h
    sleep 0.2
  done
  wait $pid; rc=$?
  t1=$(cut -d' ' -f1 /proc/uptime | tr -d .)
  echo "### exit=$rc wall_ms=$(( (t1 - t0) * 10 )) VmHWM_kB=$hwm"
  echo "### post: battery_temp_dC=$(dumpsys battery | grep temperature | awk '{print $2}')"
  echo "### post: cpufreq cur (little/big): $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq) $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq)"
  echo "### engine stderr"; cat $D/out/$tag.stderr
  echo "### stdout (first 400 chars)"; head -c 400 $D/out/$tag.stdout; echo
} > $out 2>&1
awk '/^### exit/{printf "%s ", $0} /^decode:/{printf "DECODE=%s ", $2} /^prefill:/{printf "PREFILL=%s ", $2} /^TTFT:/{printf "TTFT=%s ", $2} /^generated:/{printf "GEN=%s ", $2} /^pinned layers:/{printf "PINNED=%s/%s ", $3, $5} /^spec decode:/{sub(/^spec decode: */,""); printf "SPEC=[%s] ", $0}' $out; echo
