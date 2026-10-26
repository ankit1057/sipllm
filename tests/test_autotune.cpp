// test_autotune.cpp — issue #77: the CLI's tuning decision.
//
// The auto-tuner's micro-benchmark is not representative on big.LITTLE phones
// (it picked 2 threads + Proportional2, the schedule wave 16 measured at ~40% of
// fixed16), so the CLI runs it only on request (--autotune / --recalibrate).
// Without a request the profile must leave every default alone: threads 0 (the
// ThreadPool uses all cores) and schedule -1 (keep fixed16). No benchmarks run.
#include "llm/auto_tuner.h"
#include "tests/test_util.h"

using namespace llm;

TEST(cli_profile_without_request_keeps_defaults) {
    const RuntimeProfile p = cli_runtime_profile(get_hardware_info(), /*autotune_requested=*/false,
                                                 AutoTunerOptions{});
    CHECK_MSG(p.threads == 0, "threads left to the ThreadPool (all cores), got " + std::to_string(p.threads));
    CHECK_MSG(p.schedule_policy == -1, "schedule left at the default (fixed16), got " +
                                       std::to_string(p.schedule_policy));
}

TEST(cli_profile_no_autotune_wins_over_request) {
    AutoTunerOptions opt;
    opt.disable_autotune = true;
    const RuntimeProfile p = cli_runtime_profile(get_hardware_info(), /*autotune_requested=*/true, opt);
    CHECK_MSG(p.threads == 0 && p.schedule_policy == -1, "--no-autotune keeps every default");
}

int main() {
    printf("== test_autotune ==\n");
    return llmtest::run_all();
}
