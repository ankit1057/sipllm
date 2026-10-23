#include "llm/auto_tuner.h"
#include "llm/threadpool.h"
#include "llm/ops.h"
#include "llm/common.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>

namespace llm {

// Micro-benchmark for a single configuration (threads + policy)
static double measure_matmul_throughput(int threads, ThreadPool::SchedulePolicy policy, int iters = 20) {
    ThreadPool pool(threads);
    pool.set_policy(policy);

    // Simulate an 8B model Llama3 size matmul (e.g. 4096x4096)
    const int N = 4096;
    std::vector<float> W(N * N, 0.01f);
    std::vector<float> x(N, 0.01f);
    std::vector<float> y(N, 0.0f);

    // Warmup
    for (int i = 0; i < 2; ++i) {
        matmul(y.data(), W.data(), x.data(), N, N, &pool);
    }

    double t0 = now_sec();
    for (int i = 0; i < iters; ++i) {
        matmul(y.data(), W.data(), x.data(), N, N, &pool);
    }
    double elapsed = now_sec() - t0;
    
    // return pseudo-tokens/sec for comparison
    return (double)iters / elapsed;
}

RuntimeProfile run_micro_benchmarks(const HardwareInfo& hw) {
    LOG_INFO("Running auto-tuner micro-benchmarks on hardware: %s", hw.hardware_id().c_str());
    
    RuntimeProfile profile;
    std::stringstream ms_json;
    ms_json << "{\n    \"threads\": {\n";
    
    // 1. Thread count narrowing search (1..hw.logical_cores)
    int max_threads = hw.logical_cores;
    if (max_threads > 16) max_threads = 16; // Typically diminishing returns past 16 for memory bound workloads

    std::vector<int> candidates;
    if (max_threads == 1) candidates = {1};
    else if (max_threads <= 4) {
        for (int i = 1; i <= max_threads; ++i) candidates.push_back(i);
    } else {
        candidates = {1, 2, max_threads / 2, max_threads};
    }

    int best_threads = 1;
    double best_thread_score = 0;
    bool first_ms = true;

    // Use a fast baseline policy for finding the thread peak
    ThreadPool::SchedulePolicy baseline_policy = ThreadPool::SchedulePolicy::Fixed16;

    for (int t : candidates) {
        if (t == 0) continue;
        double score = measure_matmul_throughput(t, baseline_policy, 10);
        LOG_INFO("  Threads %2d -> throughput: %.1f", t, score);
        if (!first_ms) ms_json << ",\n";
        ms_json << "      \"" << t << "\": " << score;
        first_ms = false;
        
        if (score > best_thread_score) {
            best_thread_score = score;
            best_threads = t;
        }
    }

    // Optional: refine around the best thread count
    if (max_threads > 4 && best_threads > 1 && best_threads < max_threads) {
        int t_down = best_threads - 1;
        int t_up = best_threads + 1;
        
        if (std::find(candidates.begin(), candidates.end(), t_down) == candidates.end() && t_down > 0) {
            double score = measure_matmul_throughput(t_down, baseline_policy, 10);
            LOG_INFO("  Threads %2d -> throughput: %.1f", t_down, score);
            ms_json << ",\n      \"" << t_down << "\": " << score;
            if (score > best_thread_score) { best_thread_score = score; best_threads = t_down; }
        }
        if (std::find(candidates.begin(), candidates.end(), t_up) == candidates.end() && t_up <= max_threads) {
            double score = measure_matmul_throughput(t_up, baseline_policy, 10);
            LOG_INFO("  Threads %2d -> throughput: %.1f", t_up, score);
            ms_json << ",\n      \"" << t_up << "\": " << score;
            if (score > best_thread_score) { best_thread_score = score; best_threads = t_up; }
        }
    }
    ms_json << "\n    },\n    \"schedule_policies\": {\n";

    // 2. Scheduler selection (using best_threads)
    std::vector<ThreadPool::SchedulePolicy> policies = {
        ThreadPool::SchedulePolicy::Static,
        ThreadPool::SchedulePolicy::Fixed16,
        ThreadPool::SchedulePolicy::Proportional2,
        ThreadPool::SchedulePolicy::Adaptive
    };

    ThreadPool::SchedulePolicy best_policy = baseline_policy;
    double best_policy_score = 0;
    first_ms = true;

    LOG_INFO("Tuning scheduler with %d threads...", best_threads);
    for (auto p : policies) {
        double score = measure_matmul_throughput(best_threads, p, 15);
        LOG_INFO("  Policy %d -> throughput: %.1f", (int)p, score);
        if (!first_ms) ms_json << ",\n";
        ms_json << "      \"" << (int)p << "\": " << score;
        first_ms = false;
        
        if (score > best_policy_score) {
            best_policy_score = score;
            best_policy = p;
        }
    }
    ms_json << "\n    }\n  }";

    profile.threads = best_threads;
    profile.schedule_policy = (int)best_policy;
    profile.thread_throughput_relative = best_policy_score;
    profile.measurements_json = ms_json.str();
    
    LOG_INFO("Auto-tuner complete: selected %d threads, policy %d", profile.threads, profile.schedule_policy);
    return profile;
}

RuntimeProfile tune_if_needed(const HardwareInfo& hw, const AutoTunerOptions& opt) {
    RuntimeProfile profile;
    std::string hw_id = hw.hardware_id();

    if (opt.disable_autotune) {
        LOG_INFO("Auto-tuner disabled, using defaults");
        // No tuned policy: -1 keeps the caller's default schedule. The struct
        // default (0) is SchedulePolicy::Static, which callers would apply as
        // if it had been measured (it is the slowest policy on big.LITTLE).
        profile.schedule_policy = -1;
        return profile;
    }

    if (!opt.force_recalibrate && load_runtime_profile(hw_id, profile)) {
        LOG_INFO("Loaded runtime profile for %s (threads=%d, policy=%d)", hw_id.c_str(), profile.threads, profile.schedule_policy);
        return profile;
    }

    profile = run_micro_benchmarks(hw);
    save_runtime_profile(hw_id, profile);
    return profile;
}

} // namespace llm
