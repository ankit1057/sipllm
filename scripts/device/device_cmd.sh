#!/system/bin/sh
# device_cmd.sh — forensic wrapper for an ARBITRARY command on the phone (used for llama.cpp).
# usage: device_cmd.sh <tag> <cmd...>   → /data/local/tmp/sipllm/out/<tag>.txt
D=/data/local/tmp/sipllm
tag=$1; shift
out=$D/out/$tag.txt
{
  echo "### tag=$tag"; echo "### date=$(date '+%Y-%m-%d %H:%M:%S')"; echo "### cmd: $*"
  echo "### pre: battery_temp_dC=$(dumpsys battery | grep temperature | awk '{print $2}') level=$(dumpsys battery | grep level | awk '{print $2}')"
  echo "### pre: cpufreq cur (little/big): $(cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq) $(cat /sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq)"
  echo "### pre: MemAvailable=$(grep MemAvailable /proc/meminfo | awk '{print $2}') kB"
  echo "### run"
  t0=$(cut -d' ' -f1 /proc/uptime | tr -d .)
  nice -n 10 "$@" > $D/out/$tag.stdout 2> $D/out/$tag.stderr &
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
  echo "### stdout"; cat $D/out/$tag.stdout
  echo "### stderr (tail)"; tail -20 $D/out/$tag.stderr
} > $out 2>&1
grep -E "^### exit" $out; grep -E "tg32|tg16|pp16|pp5|t/s" $D/out/$tag.stdout | tail -4
