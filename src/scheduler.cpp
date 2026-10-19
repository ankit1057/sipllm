#include "llm/scheduler.h"

namespace llm {

Scheduler::Scheduler(Runtime* rt, size_t max_queue)
    : rt_(rt), max_queue_(max_queue) {
    start();
}

Scheduler::~Scheduler() {
    stop();
}

void Scheduler::start() {
    bool expected = false;
    if (running_.compare_exchange_strong(expected, true)) {
        stop_requested_.store(false);
        worker_ = std::thread(&Scheduler::worker_loop, this);
    }
}

void Scheduler::stop() {
    if (running_.load()) {
        stop_requested_.store(true);
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
        running_.store(false);
    }
}

std::future<std::pair<std::string, GenStats>> Scheduler::submit(
    std::string prompt, int max_new, SamplerConfig scfg, Runtime::TokenCallback on_token) {
    auto prom = std::make_shared<std::promise<std::pair<std::string, GenStats>>>();
    auto fut = prom->get_future();

    if (stop_requested_.load() || !running_.load()) {
        prom->set_exception(std::make_exception_ptr(std::runtime_error("Scheduler is stopped")));
        return fut;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= max_queue_) {
            prom->set_exception(std::make_exception_ptr(std::runtime_error("Scheduler queue full")));
            return fut;
        }

        ScheduledTask task;
        task.prompt = std::move(prompt);
        task.max_new = max_new;
        task.scfg = scfg;
        task.on_token = std::move(on_token);
        task.promise = prom;

        queue_.push(std::move(task));
    }
    cv_.notify_one();
    return fut;
}

std::future<void> Scheduler::submit_fn(std::function<void(Runtime&)> fn) {
    auto prom = std::make_shared<std::promise<void>>();
    auto fut = prom->get_future();

    if (stop_requested_.load() || !running_.load()) {
        prom->set_exception(std::make_exception_ptr(std::runtime_error("Scheduler is stopped")));
        return fut;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= max_queue_) {
            prom->set_exception(std::make_exception_ptr(std::runtime_error("Scheduler queue full")));
            return fut;
        }

        ScheduledTask task;
        task.custom_fn = std::move(fn);
        task.promise_void = prom;

        queue_.push(std::move(task));
    }
    cv_.notify_one();
    return fut;
}

size_t Scheduler::queue_size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void Scheduler::worker_loop() {
    while (!stop_requested_.load()) {
        ScheduledTask task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [&]() {
                return stop_requested_.load() || !queue_.empty();
            });

            if (stop_requested_.load() && queue_.empty()) {
                break;
            }

            if (!queue_.empty()) {
                task = std::move(queue_.front());
                queue_.pop();
            } else {
                continue;
            }
        }

        if (task.custom_fn) {
            try {
                if (rt_) task.custom_fn(*rt_);
                task.promise_void->set_value();
            } catch (...) {
                task.promise_void->set_exception(std::current_exception());
            }
        } else {
            try {
                GenStats st;
                std::string out = "";
                if (rt_) {
                    out = rt_->generate(task.prompt, task.max_new, task.scfg, task.on_token, &st);
                }
                task.promise->set_value({out, st});
            } catch (...) {
                task.promise->set_exception(std::current_exception());
            }
        }
    }

    // Drain remaining tasks with exception
    std::lock_guard<std::mutex> lock(mutex_);
    while (!queue_.empty()) {
        auto& t = queue_.front();
        if (t.promise) {
            t.promise->set_exception(std::make_exception_ptr(std::runtime_error("Scheduler stopped during queue drain")));
        }
        if (t.promise_void) {
            t.promise_void->set_exception(std::make_exception_ptr(std::runtime_error("Scheduler stopped during queue drain")));
        }
        queue_.pop();
    }
}

} // namespace llm
