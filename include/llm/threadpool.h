// threadpool.h — a minimal persistent worker pool for data-parallel loops.
//
// The hot path is matmul: split the output rows across N workers. Spawning
// threads per matmul would dominate runtime, so we keep persistent workers and
// hand them a [begin,end) range via a parallel_for barrier.
#pragma once

#include "llm/common.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace llm {

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
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    int size() const { return n_; }

    // Run fn(thread_id, begin, end) over [0, total) split into n_ contiguous
    // chunks. Blocks until every chunk completes. Safe to nest? No — do not
    // call parallel_for from within a worker body.
    void parallel_for(int64_t total,
                      const std::function<void(int, int64_t, int64_t)>& fn) {
        if (total <= 0) return;
        if (n_ == 1) { fn(0, 0, total); return; }

        // Balanced contiguous partition.
        const int64_t chunk = (total + n_ - 1) / n_;
        {
            std::unique_lock<std::mutex> lk(m_);
            fn_ = &fn;
            total_ = total;
            chunk_ = chunk;
            remaining_ = n_ - 1;   // workers we wait on (calling thread does its own)
            ++generation_;
        }
        cv_.notify_all();

        // Calling thread handles chunk 0.
        run_chunk(0);

        // Wait for workers.
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [this] { return remaining_ == 0; });
        fn_ = nullptr;
    }

private:
    void run_chunk(int tid) {
        int64_t begin = static_cast<int64_t>(tid) * chunk_;
        if (begin >= total_) return;
        int64_t end = std::min(begin + chunk_, total_);
        (*fn_)(tid, begin, end);
    }

    void worker_loop(int tid) {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [this, &seen] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            lk.unlock();

            run_chunk(tid);

            lk.lock();
            if (--remaining_ == 0) done_cv_.notify_one();
        }
    }

    int n_ = 1;
    std::vector<std::thread> workers_;

    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    bool stop_ = false;
    uint64_t generation_ = 0;
    int remaining_ = 0;

    const std::function<void(int, int64_t, int64_t)>* fn_ = nullptr;
    int64_t total_ = 0, chunk_ = 0;
};

// Process-wide default pool, lazily created.
inline ThreadPool& default_pool() {
    static ThreadPool pool;
    return pool;
}

} // namespace llm
