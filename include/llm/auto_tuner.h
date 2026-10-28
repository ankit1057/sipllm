#pragma once

#include "llm/device_profile.h"
#include <string>

namespace llm {

struct AutoTunerOptions {
    bool force_recalibrate = false;
    bool disable_autotune = false;
};

// Top-level entrypoint: loads the profile if it exists, otherwise runs
// the micro-benchmarks to generate it.
RuntimeProfile tune_if_needed(const HardwareInfo& hw, const AutoTunerOptions& opt);

// Issue #77: the profile the `llm` CLI applies. The tuner runs only when the
// user asked for it (--autotune / --recalibrate) and --no-autotune was not
// given; otherwise the profile is empty (threads 0, schedule -1), so the CLI
// keeps its measured defaults: every core and the fixed16 schedule.
RuntimeProfile cli_runtime_profile(const HardwareInfo& hw, bool autotune_requested,
                                   const AutoTunerOptions& opt);

// Manual runner for the bench_micro CLI tool
RuntimeProfile run_micro_benchmarks(const HardwareInfo& hw);

} // namespace llm
