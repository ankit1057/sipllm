# Auto-tuning

> [!NOTE] Architecture book
> **Problem → Design → Alternatives → Tradeoffs → Source → Future.** The auto-tuner picks two runtime knobs per machine — thread count and scheduler policy — from a one-time matmul micro-benchmark, then caches the result. It does **not** choose the RAM budget; that is the [memory planner](memory-planner.html)'s job.

## Problem

The right thread count for a memory-bound matmul is not "all cores" — past a point, more threads just contend for memory bandwidth and the barrier cost dominates. The right value depends on the machine and cannot be guessed from core count alone. Likewise the [thread pool](thread-pool.html) has several work-splitting policies whose best choice varies by hardware. Rather than ship a fixed default or make the user tune by hand, SipLLM measures the machine once and remembers the answer.

## Design

Two pieces: hardware detection (`include/llm/device_profile.h`, `src/device_profile.cpp`) and the tuner (`include/llm/auto_tuner.h`, `src/auto_tuner.cpp`).

### Hardware detection

`get_hardware_info()` reads static traits from the OS — CPU model, logical/physical cores, RAM bytes, page size, architecture:

- **macOS** via `sysctlbyname` (`machdep.cpu.brand_string`, `hw.logicalcpu`, `hw.physicalcpu`, `hw.memsize`, `hw.pagesize`).
- **Linux** via `/proc/cpuinfo`, `sysconf(_SC_NPROCESSORS_ONLN)`, `lscpu`, and `_SC_PHYS_PAGES × _SC_PAGE_SIZE`.
- **Otherwise** `std::thread::hardware_concurrency()` as a floor.

`HardwareInfo::hardware_id()` collapses those into a deterministic slug — architecture, cleaned CPU model, logical core count, and RAM in GB (e.g. `arm64_apple_m3_8c_16gh`) — which keys the cache so a profile from one machine never loads on another.

### The tuner

`tune_if_needed(hw, opt)` is the entry point:

```text
tune_if_needed(hw, opt):
    if opt.disable_autotune:                 return defaults (threads=4, policy=0)
    if !force_recalibrate and cached profile exists for hw_id:
                                             return cached profile
    profile = run_micro_benchmarks(hw)
    save to ~/.sipllm/runtime/<hw_id>.json
    return profile
```

`run_micro_benchmarks` runs a **4096×4096 fp32 matmul** — sized to mimic an 8B-Llama3 projection — under different configurations, timing iterations into a pseudo-tokens/sec score:

1. **Thread count.** A narrowing search over candidates drawn from `logical_cores`, **capped at 16** (diminishing returns past that for a memory-bound workload). For ≤4 cores it tries every count; for more it probes `{1, 2, max/2, max}` then optionally refines around the winner. Measured under a fixed baseline policy.
2. **Scheduler policy.** With the winning thread count fixed, it benchmarks a set of policies and keeps the fastest.

The chosen `threads` and `schedule_policy` are written to `~/.sipllm/runtime/<hw_id>.json` (under `get_sipllm_home()`, default `$HOME/.sipllm`), alongside the raw per-config measurements. The next run loads that file and skips the benchmark entirely.

### CLI surface

- `--no-autotune` — skip tuning, use defaults (`threads=4`, `policy=0`).
- `--recalibrate` — force a fresh benchmark even if a cached profile exists (also the only flag `bench_micro` accepts).
- `--profile-info` — print the detected `hardware_id` and the `HardwareInfo` JSON, then exit.

## Caveats

> [!WARNING] What the auto-tuner does and does not do
> - **It does not pick the RAM budget.** Peak-RSS budgeting is the separate `mem_plan.cpp` contract planner behind `--ram-budget`; the tuner only touches thread count and scheduler.
> - **Only 4 of the 7 schedule policies are benchmarked** — `Static`, `Fixed16`, `Proportional2`, `Adaptive`. The other policies exist in the pool but are never auto-selected.
> - **No big.LITTLE / core-cluster detection.** It reads a single logical/physical core count and treats all cores as equivalent; it does not distinguish performance from efficiency cores.
> - **`load_hardware_profile` is a no-op.** It intentionally returns `false` without parsing — hardware re-detection is <5 ms, so the `hardware.json` cache is never actually read back. Only the *runtime* profile (`runtime/<hw_id>.json`) is loaded.

## Alternatives considered

| Approach | Why not |
|:---------|:--------|
| Fixed default thread count | Wrong on most machines — either leaves cores idle or oversubscribes a bandwidth-bound kernel. |
| `threads = logical_cores` | Past ~16 threads a memory-bound matmul regresses; the cap and search find the real peak. |
| Tune on every launch | The 4096² benchmark costs real time; caching per `hardware_id` pays it once. |
| Benchmark all 7 policies | Diminishing returns for extra launch cost; the 4 measured cover the useful spectrum. |
| Parse `hardware.json` on load | Detection is faster than parsing, so the read-back is deliberately skipped (`load_hardware_profile` returns `false`). |

## Tradeoffs

The tuner optimizes a **proxy** — a synthetic 4096² fp32 matmul — not the exact model you will run, so the picked thread count is a good default, not a per-model optimum. It also assumes homogeneous cores, which is imperfect on big.LITTLE ARM. In exchange it is cheap (one benchmark, then a cached JSON read), fully offline, and machine-scoped. Because it only sets thread count and scheduler, it can never affect correctness or memory footprint — those are owned by the loader and memory planner.

## Source files

| File | Role |
|:-----|:-----|
| `include/llm/auto_tuner.h` | `tune_if_needed` / `run_micro_benchmarks` entry points, `AutoTunerOptions` |
| `src/auto_tuner.cpp` | 4096² matmul micro-benchmark, thread-count search (cap 16), 4-policy scheduler selection |
| `include/llm/device_profile.h` | `HardwareInfo` / `RuntimeProfile` structs, profile path helpers |
| `src/device_profile.cpp` | `get_hardware_info()` (sysctl / `/proc`), `hardware_id()`, profile load/save, no-op `load_hardware_profile` |
| `tools/bench_micro.cpp` | standalone tuner runner (`--recalibrate`) |

## Future work

- **big.LITTLE awareness** — pin to performance cores or weight the thread search by cluster.
- **Benchmark the remaining schedule policies** and select from the full set.
- **Model-aware tuning** — benchmark at the actual model's projection sizes instead of a fixed 4096².
- **Wire up `load_hardware_profile`** or remove it — today it is a deliberate stub.
