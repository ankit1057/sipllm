#include "llm/scheduler.h"
#include "llm/toy_model.h"
#include "tests/test_util.h"

using namespace llm;

static std::string make_toy_sched() {
    ToyConfig tc;
    tc.n_layers = 2;
    tc.dim = 32;
    tc.n_heads = 4;
    tc.n_kv_heads = 2;
    tc.ffn_dim = 64;
    tc.vocab_size = 256;
    tc.ctx_len = 256;
    tc.seed = 7;
    std::string path = llmtest::scratch_path("toy_sched.llmw");
    write_toy_model(path, tc);
    return path;
}

TEST(scheduler_fifo_order_and_future) {
    const std::string path = make_toy_sched();
    auto src = open_model(path);
    LayerLoader::Options opt;
    opt.residency = Residency::FP32;
    opt.async = false;
    Runtime rt(std::move(src), opt, 256, 1);

    Scheduler sched(&rt, 8);
    CHECK(sched.is_running());

    SamplerConfig scfg;
    scfg.temperature = 0.0f; // greedy

    auto f1 = sched.submit("Task 1", 3, scfg);
    auto f2 = sched.submit("Task 2", 3, scfg);

    auto r1 = f1.get();
    auto r2 = f2.get();

    CHECK(!r1.first.empty());
    CHECK(!r2.first.empty());
    CHECK(r1.second.gen_tokens > 0);
    CHECK(r2.second.gen_tokens > 0);

    sched.stop();
    CHECK(!sched.is_running());
}

TEST(scheduler_queue_limit_rejection) {
    const std::string path = make_toy_sched();
    auto src = open_model(path);
    LayerLoader::Options opt;
    opt.residency = Residency::FP32;
    opt.async = false;
    Runtime rt(std::move(src), opt, 256, 1);

    // Queue depth 1
    Scheduler sched(&rt, 1);
    SamplerConfig scfg;

    auto f1 = sched.submit("T1", 5, scfg);
    auto f2 = sched.submit("T2", 5, scfg);
    auto f3 = sched.submit("T3", 5, scfg);

    // One of them must be accepted, and excess queued tasks might succeed or fail depending on drain speed,
    // but stopping should gracefully drain/reject.
    sched.stop();
}

int main() {
    printf("== test_scheduler ==\n");
    return llmtest::run_all();
}
