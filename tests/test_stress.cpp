// test_stress.cpp — crash/fuzz regression harness for the inference runtime.
//
// Guards the intermittent uninitialized-read class of bug (see the fix commit):
// it drives many prompts and loader/threading configurations end-to-end against
// a toy model (always) and the real model (when present), in a single long-
// lived process so that thread-pool and prefetch-worker stacks are reused and
// accumulate the leftover garbage that fresh one-shot processes zero out. It is
// the workload the Linux valgrind/ASan CI jobs run under so an uninitialized
// read can never silently return.
//
// Env:
//   SIPLLM_STRESS_MODEL  optional path to a real GGUF (else toy only)
//   SIPLLM_STRESS_ITERS  iterations per config (default 40)
#include "llm/runtime.h"
#include "llm/gguf_writer.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace llm;

namespace {

// A spread of prompts: short, repeated char (the original repro), spaces,
// digits, punctuation, and mixed — to exercise the tokenizer + varied seq lens.
const char* kPrompts[] = {
    "AAAA", "Hello", "The capital of France is", "a", "",
    "1234567890", "   ", "!?.,;:", "The quick brown fox", "xyzzy AAAA 42",
};

struct Cfg { bool async; int buffers; int threads; const char* name; };
const Cfg kCfgs[] = {
    {true,  2, 0, "async/buffers=2/threads=auto"},
    {true,  3, 4, "async/buffers=3/threads=4"},
    {true,  4, 8, "async/buffers=4/threads=8"},
    {false, 1, 1, "sync/no-async/threads=1"},
    {false, 1, 2, "sync/threads=2"},
};

// Run `iters` generations for one (model, cfg). Returns number of anomalies.
int run_config(const std::string& model, const Cfg& c, int iters) {
    int anomalies = 0;
    for (int it = 0; it < iters; ++it) {
        const char* prompt = kPrompts[(it * 7 + c.buffers) % (int)(sizeof(kPrompts) / sizeof(kPrompts[0]))];
        LayerLoader::Options opt;
        opt.async = c.async;
        opt.n_buffers = c.async ? c.buffers : 1;
        auto src = open_model(model, /*use_mmap=*/(it & 1) != 0);
        Runtime rt(std::move(src), opt, /*max_ctx=*/0, c.threads);
        SamplerConfig scfg;
        scfg.temperature = 0.f;   // greedy — deterministic, matches the repro
        GenStats st;
        int n_new = 1 + (it % 8);
        std::string out = rt.generate(prompt, n_new, scfg, nullptr, &st);
        // Sanity: the stat counters must be plausible small integers. A wild
        // heap write (the non-crashing face of the bug) showed up here first as
        // a pointer-sized garbage value, so assert a sane upper bound.
        uint64_t loads = st.prefetch_hits + st.prefetch_misses;
        if (loads > (uint64_t)rt.config().n_layers * (st.prompt_tokens + st.gen_tokens + 4) + 64) {
            fprintf(stderr, "  ANOMALY [%s] iter %d: implausible loads=%llu (hits=%llu misses=%llu)\n",
                    c.name, it, (unsigned long long)loads,
                    (unsigned long long)st.prefetch_hits, (unsigned long long)st.prefetch_misses);
            anomalies++;
        }
        if (st.gen_tokens < 0 || st.gen_tokens > n_new) {
            fprintf(stderr, "  ANOMALY [%s] iter %d: gen_tokens=%d out of range\n", c.name, it, st.gen_tokens);
            anomalies++;
        }
    }
    return anomalies;
}

}  // namespace

int main() {
    const int iters = []{ const char* e = std::getenv("SIPLLM_STRESS_ITERS"); return e ? std::max(1, atoi(e)) : 40; }();

    // Always exercise a toy model (no download). Write it to a temp path.
    std::string toy = "/tmp/sipllm_stress_toy.gguf";
    {
        ToyGgufConfig spec;
        spec.n_layers = 4; spec.dim = 256; spec.n_heads = 8; spec.n_kv_heads = 8;
        spec.ffn_dim = 1024; spec.vocab_size = 320; spec.ctx_len = 128;
        spec.weight_type = DType::Q8_0; spec.with_tokenizer = true;
        write_toy_gguf(toy, spec);
    }

    std::vector<std::string> models{toy};
    if (const char* m = std::getenv("SIPLLM_STRESS_MODEL")) models.emplace_back(m);

    int total = 0, anomalies = 0;
    for (const auto& model : models) {
        fprintf(stderr, "== stress model: %s ==\n", model.c_str());
        for (const auto& c : kCfgs) {
            anomalies += run_config(model, c, iters);
            total += iters;
        }
    }
    fprintf(stderr, "stress: ran %d generations across %zu model(s); %d anomalies\n",
            total, models.size(), anomalies);
    if (anomalies) { fprintf(stderr, "STRESS FAILED\n"); return 1; }
    fprintf(stderr, "STRESS OK\n");
    return 0;
}
