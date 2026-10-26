// threadpool.h — a minimal persistent worker pool for data-parallel loops.
//
// The hot path is matmul: split the output rows across N workers. Spawning
// threads per matmul would dominate runtime, so we keep persistent workers and
// hand them a [begin,end) range via a parallel_for barrier.
//
// Handoff (#80): a decode token runs ~200 parallel regions, so a mutex +
// condition-variable round trip per region dominated small models on phones.
// Workers now spin briefly on an atomic generation counter (and the caller on
// an atomic completion count) before falling back to sleeping on the condition
// variables. The work split is unchanged, so results are unchanged.
#pragma once

#include "llm/common.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace llm {

inline void cpu_relax() {
#if defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__("yield");
#elif defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#endif
}

class ThreadPool {
public:
    // n_threads <= 0 → use hardware_concurrency. We leave headroom so the OS /
    // prefetch thread stay responsive on an 8-core phone.
    explicit ThreadPool(int n_threads = 0) {
        int hw = static_cast<int>(std::thread::hardware_concurrency());
        if (hw <= 0) hw = 4;
        n_ = n_threads > 0 ? n_threads : hw;
        if (n_ < 1) n_ = 1;
        // One less worker thread than n_ because the calling thread also works.
        for (int i = 1; i < n_; ++i)
            workers_.emplace_back([this, i] { worker_loop(i); });
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lk(m_);
            stop_.store(true);
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    int size() const { return n_; }

    enum class SchedulePolicy {
        Static,
        Fixed8,
        Fixed16,
        Fixed32,
        Proportional2,
        Proportional4,
        Adaptive
    };

    struct Stats {
        std::atomic<uint64_t> chunks_processed{0};
        std::atomic<uint64_t> steals{0};
        std::atomic<uint64_t> idle_time_us{0};
        std::atomic<uint64_t> barrier_time_us{0};
        void reset() {
            chunks_processed = 0;
            steals = 0;
            idle_time_us = 0;
            barrier_time_us = 0;
        }
    };

    Stats stats;
    SchedulePolicy policy = SchedulePolicy::Fixed16;   // see main.cpp (wave 16)

    void set_policy(SchedulePolicy p) { policy = p; }

    // chunk_override > 0 fixes the chunk size for this call (attention uses 1:
    // few, heavy items). 0 keeps the schedule policy's chunking.
    void parallel_for(int64_t total,
                      const std::function<void(int, int64_t, int64_t)>& fn,
                      int64_t chunk_override = 0) {
        if (total <= 0) return;
        if (n_ == 1) { fn(0, 0, total); return; }

        {
            std::unique_lock<std::mutex> lk(m_);
            fn_ = &fn;
            total_ = total;
            next_row_.store(0, std::memory_order_relaxed);
            
            // Calculate chunk size based on policy
            if (policy == SchedulePolicy::Static) chunk_ = (total + n_ - 1) / n_;
            else if (policy == SchedulePolicy::Fixed8) chunk_ = 8;
            else if (policy == SchedulePolicy::Fixed16) chunk_ = 16;
            else if (policy == SchedulePolicy::Fixed32) chunk_ = 32;
            else if (policy == SchedulePolicy::Proportional2) chunk_ = std::max<int64_t>(1, total / (n_ * 2));
            else if (policy == SchedulePolicy::Proportional4) chunk_ = std::max<int64_t>(1, total / (n_ * 4));
            else if (policy == SchedulePolicy::Adaptive) chunk_ = std::max<int64_t>(16, total / (n_ * 4));
            if (chunk_override > 0) chunk_ = chunk_override;

            remaining_.store(n_ - 1);   // workers we wait on (calling thread does its own)
            generation_.fetch_add(1);   // publish under m_: a worker about to sleep re-checks it
        }
        if (sleepers_.load() > 0) cv_.notify_all();

        run_chunk(0);
        double end_time = now_sec();

        // Wait for workers: spin first, sleep only if they are slow.
        for (int spin = 0; remaining_.load() != 0 && spin < kSpinIters; ++spin) cpu_relax();
        if (remaining_.load() != 0) {
            std::unique_lock<std::mutex> lk(m_);
            main_waiting_.store(true);
            done_cv_.wait(lk, [this] { return remaining_.load() == 0; });
            main_waiting_.store(false);
        }
        stats.barrier_time_us.fetch_add((uint64_t)((now_sec() - end_time) * 1e6), std::memory_order_relaxed);
        fn_ = nullptr;
    }

private:
    void run_chunk(int tid) {
        if (policy == SchedulePolicy::Static) {
            int64_t begin = static_cast<int64_t>(tid) * chunk_;
            if (begin >= total_) return;
            int64_t end = std::min(begin + chunk_, total_);
            stats.chunks_processed.fetch_add(1, std::memory_order_relaxed);
            (*fn_)(tid, begin, end);
            return;
        }

        // Dynamic work-stealing loop
        while (true) {
            int64_t begin = next_row_.fetch_add(chunk_, std::memory_order_relaxed);
            if (begin >= total_) break;
            int64_t end = std::min(begin + chunk_, total_);
            stats.chunks_processed.fetch_add(1, std::memory_order_relaxed);
            if (begin > 0) stats.steals.fetch_add(1, std::memory_order_relaxed); // Proxy for steals
            (*fn_)(tid, begin, end);
        }
    }

    void worker_loop(int tid) {
        // Platform specific affinity / QoS
#if defined(__APPLE__)
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
        uint64_t seen = 0;
        for (;;) {
            double wait_start = now_sec();
            uint64_t g = generation_.load();
            for (int spin = 0; g == seen && !stop_.load() && spin < kSpinIters; ++spin) {
                cpu_relax();
                g = generation_.load();
            }
            if (g == seen && !stop_.load()) {
                std::unique_lock<std::mutex> lk(m_);
                sleepers_.fetch_add(1);
                cv_.wait(lk, [this, &seen] { return stop_.load() || generation_.load() != seen; });
                sleepers_.fetch_sub(1);
                g = generation_.load();
            }
            stats.idle_time_us.fetch_add((uint64_t)((now_sec() - wait_start) * 1e6), std::memory_order_relaxed);
            if (stop_.load()) return;
            seen = g;

            run_chunk(tid);

            // Last worker out wakes the caller only if it went to sleep.
            if (remaining_.fetch_sub(1) == 1 && main_waiting_.load()) {
                std::lock_guard<std::mutex> lk(m_);
                done_cv_.notify_one();
            }
        }
    }

    int n_ = 1;
    std::vector<std::thread> workers_;

    // ~50-150 us of spinning on a phone core before a thread sleeps.
    static constexpr int kSpinIters = 20000;

    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> generation_{0};
    std::atomic<int> remaining_{0};
    std::atomic<int> sleepers_{0};        // workers blocked in cv_
    std::atomic<bool> main_waiting_{false};

    const std::function<void(int, int64_t, int64_t)>* fn_ = nullptr;
    int64_t total_ = 0, chunk_ = 0;
    std::atomic<int64_t> next_row_{0};
};

// Process-wide default pool, lazily created.
inline ThreadPool& default_pool() {
    static ThreadPool pool;
    return pool;
}

} // namespace llm
