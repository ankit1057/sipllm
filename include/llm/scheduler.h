// scheduler.h — FIFO Request Scheduler with worker thread and future.
//
// Serializes multiple concurrent inference requests into a dedicated worker
// thread, eliminating 429 busy errors and coordinating access to the Runtime.
#pragma once

#include "llm/runtime.h"

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

namespace llm {

struct ScheduledTask {
    std::string prompt;
    int max_new = 64;
    SamplerConfig scfg;
    Runtime::TokenCallback on_token = nullptr;
    std::shared_ptr<std::promise<std::pair<std::string, GenStats>>> promise;

    std::function<void(Runtime&)> custom_fn;
    std::shared_ptr<std::promise<void>> promise_void;
};

class Scheduler {
public:
    explicit Scheduler(Runtime* rt, size_t max_queue = 64);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // Start the worker thread.
    void start();

    // Stop the worker thread and drain remaining or reject new tasks.
    void stop();

    // Submit an inference task, returning a future with (generated_text, stats).
    // Returns unfulfilled future with exception if queue is full or scheduler stopped.
    std::future<std::pair<std::string, GenStats>> submit(
        std::string prompt, int max_new = 64,
        SamplerConfig scfg = {}, Runtime::TokenCallback on_token = nullptr);

    // Submit a custom task to run against the Runtime on the scheduler thread.
    std::future<void> submit_fn(std::function<void(Runtime&)> fn);

    // Queue inspections
    size_t queue_size() const;
    bool is_running() const { return running_.load(); }

private:
    void worker_loop();

    Runtime* rt_;
    size_t max_queue_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_requested_{false};

    std::queue<ScheduledTask> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
};

} // namespace llm
