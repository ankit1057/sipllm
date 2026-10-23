# On-device benchmarking (adb shell, no app)

Runs the engine on a connected Android phone from `/data/local/tmp` with forensic
capture per run: battery temperature, big/little `scaling_cur_freq`, MemAvailable,
the kernel's `VmHWM` (high-water RSS, sampled 5×/s), wall time, and the engine's own
stats. Nothing is installed and the UI is never touched; every run is `nice 10`.

```sh
# 1. cross-build (NDK; libc++ linked statically so the binaries run from /data/local/tmp)
make -j4 BUILD=build-android CXX=$NDK/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android29-clang++ \
     ARCHFLAGS="-march=armv8-a" build-android/llm build-android/test_neon build-android/test_spec_equiv
#    (baseline ISA: runs on any arm64 phone; the SDOT kernels in src/neon_sdot.cpp are built with
#     dotprod automatically and chosen at runtime — check `test_neon` prints dotprod=1 on SDOT phones)
# 2. push
adb shell mkdir -p /data/local/tmp/sipllm/out
adb push build-android/llm build-android/test_neon build-android/test_spec_equiv scripts/device/device_run.sh scripts/device/device_cmd.sh /data/local/tmp/sipllm/
adb push ~/.sipllm/models/tinyllama-q4_k_m.gguf /data/local/tmp/sipllm/
adb shell "chmod 755 /data/local/tmp/sipllm/*"
# 3. correctness on the SoC, then the matrix
adb shell /data/local/tmp/sipllm/test_neon && adb shell /data/local/tmp/sipllm/test_spec_equiv
SECTIONS="threads budget spec" N=3 NGEN=32 ./scripts/device/device_bench.sh
# any other binary through the same capture (e.g. llama.cpp):
adb shell /data/local/tmp/sipllm/device_cmd.sh llamabench /data/local/tmp/llamacpp/llama-bench -m ... -t 4 -p 16 -n 32 -r 3
```

Before benchmarking, quiet the phone: `pm list packages -3` ∩ live processes →
`am force-stop <pkg>` each (record what was stopped). `am kill-all` needs a permission
the shell user lacks. Thermal zones and the cpufreq governor are often unreadable by
the shell user; the wrapper records what is readable and says so.

Evidence stays local under `bench/device/<timestamp>/` (gitignored); commit only the
aggregated JSON produced by `scripts/device/assemble_device_json.py`.
