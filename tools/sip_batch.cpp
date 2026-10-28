// sip_batch.cpp — batched multi-sequence generation over a simple pipe protocol.
//
//   sip_batch <model> [--batch N] [--ctx N] [--threads N] [--gpu-layers N|max]
//             [--ram-budget BYTES|N{K,M,G}] [--fast] [--mmap] [--max-active-tokens N]
//
// --max-active-tokens caps memory: a job is admitted only while the active
// jobs' (prompt + max_new) tokens, plus its own, stay within N (each cached
// token costs one KV row per layer). A job is always admitted into an empty
// batch, so oversized jobs still run, alone.
//
// Loads the model once, prefills an optional shared PREFIX once, then runs up
// to N jobs at a time: each job's prompt is prefilled into its own KV cache
// (starting from a copy of the prefix KV when the prompt begins with the
// prefix's tokens) and all active jobs decode in lockstep through
// Transformer::forward_multi, so every layer's weights are read once per step
// for all of them. A finished job's slot is refilled with the next job
// immediately (continuous batching). Greedy decoding; per job the stop rules
// are Runtime::generate's: end-of-generation token, max_new tokens, or a full
// context window.
//
// stdin (length-prefixed, so prompts may contain anything):
//   PREFIX <nbytes>\n<bytes>                 (optional, must come first)
//   JOB <id> <max_new> <nbytes>\n<bytes>     (repeated)
//   END\n                                    (or EOF)
// stdout, one record per finished job, in completion order:
//   RESULT <id> <stop:eog|max|ctx> <prompt_tokens> <reused_tokens> <gen_tokens>
//          <prefill_ms> <decode_ms> <nbytes>\n<bytes>\n
// Logs go to stderr.
#include "llm/runtime.h"
#include "llm/auto_tuner.h"
#include "llm/cuda_backend.h"
#include "llm/device_profile.h"
#include "llm/sampler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

using namespace llm;

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t parse_bytes(const std::string& s) {
    if (s.empty()) return 0;
    double v = std::atof(s.c_str());
    switch (s.back()) {
        case 'K': case 'k': v *= 1024.0; break;
        case 'M': case 'm': v *= 1024.0 * 1024; break;
        case 'G': case 'g': v *= 1024.0 * 1024 * 1024; break;
        default: break;
    }
    return (size_t)v;
}

bool read_line(std::string& line) {
    line.clear();
    int c;
    while ((c = std::fgetc(stdin)) != EOF) {
        if (c == '\n') return true;
        line.push_back((char)c);
    }
    return !line.empty();
}

bool read_bytes(size_t n, std::string& out) {
    out.resize(n);
    return n == 0 || std::fread(&out[0], 1, n, stdin) == n;
}

struct Job {
    std::string id;
    int max_new = 0;
    std::string prompt;
    std::vector<int64_t> ids;   // filled at admission
};

struct Slot {
    bool active = false;
    Job job;
    std::unique_ptr<KVCache> kv;
    std::unique_ptr<Sampler> sampler;
    int64_t pos = 0;
    int64_t next = -1;
    int gen = 0;
    int prompt_tokens = 0, reused = 0;
    double prefill_ms = 0, decode_t0 = 0;
    std::string out;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model> [--batch N] [--ctx N] [--threads N] "
                             "[--gpu-layers N|max] [--ram-budget B] [--fast] [--mmap]\n", argv[0]);
        return 2;
    }
    const std::string model = argv[1];
    int batch = 4, ctx = 0, threads = 0;
    long long max_active_tokens = 0;   // 0 = no cap
    bool threads_overridden = false;
    size_t ram_budget = 0;
    LayerLoader::Options opt;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* def) -> std::string { return (i + 1 < argc) ? argv[++i] : def; };
        if (a == "--batch") batch = std::max(1, std::atoi(next("4").c_str()));
        else if (a == "--ctx") ctx = std::atoi(next("0").c_str());
        else if (a == "--threads") { threads = std::atoi(next("0").c_str()); threads_overridden = threads > 0; }
        else if (a == "--gpu-layers") opt.gpu_layers = cuda::parse_gpu_layers(next("max"));
        else if (a == "--ram-budget") ram_budget = parse_bytes(next("0"));
        else if (a == "--fast") opt.fast_quant = true;
        else if (a == "--mmap") opt.use_mmap = true;
        else if (a == "--max-active-tokens") max_active_tokens = std::atoll(next("0").c_str());
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }

    // ---- read the whole job list ------------------------------------------
    std::string prefix;
    std::deque<Job> pending;
    std::string line;
    while (read_line(line)) {
        if (line == "END") break;
        char kind[16] = {0};
        if (std::sscanf(line.c_str(), "%15s", kind) != 1) continue;
        if (!std::strcmp(kind, "PREFIX")) {
            size_t n = 0;
            std::sscanf(line.c_str(), "PREFIX %zu", &n);
            if (!read_bytes(n, prefix)) { std::fprintf(stderr, "short PREFIX\n"); return 2; }
        } else if (!std::strcmp(kind, "JOB")) {
            char id[256] = {0};
            int max_new = 0;
            size_t n = 0;
            if (std::sscanf(line.c_str(), "JOB %255s %d %zu", id, &max_new, &n) != 3) {
                std::fprintf(stderr, "bad JOB header: %s\n", line.c_str());
                return 2;
            }
            Job j;
            j.id = id;
            j.max_new = max_new;
            if (!read_bytes(n, j.prompt)) { std::fprintf(stderr, "short JOB %s\n", id); return 2; }
            pending.push_back(std::move(j));
        } else {
            std::fprintf(stderr, "unknown record: %s\n", line.c_str());
            return 2;
        }
    }
    const size_t n_jobs = pending.size();

    try {
        HardwareInfo hw = get_hardware_info();
        AutoTunerOptions tuner_opt;
        RuntimeProfile rp = tune_if_needed(hw, tuner_opt);
        if (!threads_overridden && rp.threads > 0) threads = rp.threads;
        Runtime rt(open_model(model, opt.use_mmap), opt, ctx, threads, ram_budget, false, KVPrecision::FP32);
        if (rt.thread_pool() && rp.schedule_policy >= 0)
            rt.thread_pool()->set_policy((ThreadPool::SchedulePolicy)rp.schedule_policy);
        const Tokenizer& tok = rt.tokenizer();
        Transformer* tf = rt.transformer();
        KVCache* base = rt.kv_cache();
        const int64_t vocab = rt.config().vocab_size;
        const int64_t max_ctx = base->max_ctx();

        // ---- shared prefix, prefilled once -------------------------------
        std::vector<int64_t> prefix_ids;
        if (!prefix.empty()) {
            prefix_ids = tok.encode(prefix, /*add_bos=*/true);
            base->clear();
            tf->set_kv(base);
            const double t0 = now_ms();
            tf->prefill(prefix_ids.data(), (int64_t)prefix_ids.size(), 0);
            std::fprintf(stderr, "[batch] prefix: %zu tokens in %.1f s\n", prefix_ids.size(),
                         (now_ms() - t0) / 1000.0);
        }
        const KVCache prefix_kv = *base;   // template every slot starts from

        std::vector<Slot> slots((size_t)batch);
        for (Slot& s : slots) s.kv = std::make_unique<KVCache>(prefix_kv);
        std::fprintf(stderr, "[batch] %zu jobs, batch %d, ctx %lld\n", n_jobs, batch, (long long)max_ctx);

        SamplerConfig scfg;
        scfg.temperature = 0.f;   // greedy, as `llm --greedy`
        double fwd_ms = 0, prefill_ms_total = 0;   // time inside forward_multi / prefill

        auto emit = [&](Slot& s, const char* stop) {
            const double dec_ms = s.decode_t0 > 0 ? now_ms() - s.decode_t0 : 0;
            std::printf("RESULT %s %s %d %d %d %.0f %.0f %zu\n", s.job.id.c_str(), stop, s.prompt_tokens,
                        s.reused, s.gen, s.prefill_ms, dec_ms, s.out.size());
            std::fwrite(s.out.data(), 1, s.out.size(), stdout);
            std::fputc('\n', stdout);
            std::fflush(stdout);
            s.active = false;
        };
        // Per-job stop rules, checked before a token is emitted (Runtime::generate).
        auto should_stop = [&](Slot& s) -> const char* {
            if (s.next < 0 || tok.is_eog(s.next)) return "eog";
            if (s.gen >= s.job.max_new) return "max";
            if (s.pos >= max_ctx) return "ctx";
            return nullptr;
        };

        auto start_job = [&](Slot& s, Job&& j) {
            s.job = std::move(j);
            s.out.clear();
            s.gen = 0;
            s.decode_t0 = 0;
            std::vector<int64_t> ids = std::move(s.job.ids);
            s.prompt_tokens = (int)ids.size();
            const int64_t P = (int64_t)prefix_ids.size();
            int64_t start = 0;
            if (P > 0 && (int64_t)ids.size() > P && std::equal(prefix_ids.begin(), prefix_ids.end(), ids.begin())) {
                *s.kv = prefix_kv;
                start = P;
            } else {
                s.kv->clear();
            }
            s.reused = (int)start;
            s.sampler = std::make_unique<Sampler>(scfg);
            if ((int64_t)ids.size() >= max_ctx) {   // cannot even prefill
                s.pos = (int64_t)ids.size();
                s.prefill_ms = 0;
                s.active = true;
                emit(s, "ctx");
                return;
            }
            tf->set_kv(s.kv.get());
            const double t0 = now_ms();
            const float* lg = tf->prefill(ids.data() + start, (int64_t)ids.size() - start, start);
            s.prefill_ms = now_ms() - t0;
            prefill_ms_total += s.prefill_ms;
            for (int64_t id : ids) s.sampler->accept(id);
            s.pos = (int64_t)ids.size();
            s.next = s.sampler->sample(lg, vocab);
            s.active = true;
        };

        size_t done = 0;
        const double t_all = now_ms();
        long long steps = 0, rows = 0;
        // Every 50 decode steps: ms/step, rows/step and mean position (context
        // length) over that window, so a running batch can be compared live.
        double win_ms = 0, win_pos = 0;
        long long win_steps = 0, win_rows = 0;
        while (true) {
            // Refill free slots, and retire jobs that stop before decoding.
            for (Slot& s : slots) {
                while (!s.active && !pending.empty()) {
                    Job& front = pending.front();
                    if (front.ids.empty()) front.ids = tok.encode(front.prompt, /*add_bos=*/true);
                    if (max_active_tokens > 0) {
                        long long used = 0;
                        bool any = false;
                        for (const Slot& o : slots)
                            if (o.active) { used += o.prompt_tokens + o.job.max_new; any = true; }
                        if (any && used + (long long)front.ids.size() + front.max_new > max_active_tokens) break;
                    }
                    Job j = std::move(pending.front());
                    pending.pop_front();
                    start_job(s, std::move(j));
                    if (!s.active) { ++done; continue; }        // emitted already ("ctx")
                    if (const char* why = should_stop(s)) { emit(s, why); ++done; }
                }
            }
            // Gather active rows: emit each slot's pending token, then decode it.
            std::vector<Slot*> act;
            for (Slot& s : slots)
                if (s.active) act.push_back(&s);
            if (act.empty()) break;
            std::vector<int64_t> toks, poss;
            std::vector<KVCache*> kvs;
            std::vector<Slot*> stepping;
            for (Slot* s : act) {
                if (s->decode_t0 == 0) s->decode_t0 = now_ms();
                s->out += tok.decode_token(s->next);
                ++s->gen;
                // The forward after the last allowed token would only feed a
                // sample that is discarded; finish without it.
                if (s->gen >= s->job.max_new) { emit(*s, "max"); ++done; continue; }
                toks.push_back(s->next);
                poss.push_back(s->pos);
                kvs.push_back(s->kv.get());
                stepping.push_back(s);
            }
            if (stepping.empty()) continue;
            const double tf0 = now_ms();
            const float* lg = tf->forward_multi(toks.data(), poss.data(), kvs.data(), (int64_t)stepping.size());
            const double step_ms = now_ms() - tf0;
            fwd_ms += step_ms;
            ++steps;
            rows += (long long)stepping.size();
            win_ms += step_ms; ++win_steps; win_rows += (long long)stepping.size();
            for (int64_t p : poss) win_pos += (double)p;
            if (win_steps == 50) {
                std::fprintf(stderr, "[batch] step %lld: %.0f ms/step, %.1f rows/step, mean ctx %.0f\n", steps,
                             win_ms / win_steps, (double)win_rows / win_steps, win_pos / win_rows);
                win_ms = win_pos = 0; win_steps = win_rows = 0;
            }
            for (size_t r = 0; r < stepping.size(); ++r) {
                Slot* s = stepping[r];
                ++s->pos;
                s->next = s->sampler->sample(lg + r * vocab, vocab);
                if (const char* why = should_stop(*s)) { emit(*s, why); ++done; }
            }
        }
        tf->set_kv(base);   // slots (and their caches) die before the runtime
        const double secs = (now_ms() - t_all) / 1000.0;
        std::fprintf(stderr, "[batch] done: %zu/%zu jobs in %.1f s, %lld decode steps, %.2f rows/step\n",
                     done, n_jobs, secs, steps, steps ? (double)rows / steps : 0.0);
        std::fprintf(stderr, "[batch] prefill %.1f s total; decode %lld tokens in %.1f s = %.2f tok/s (%.0f ms/step)\n",
                     prefill_ms_total / 1000.0, rows, fwd_ms / 1000.0, fwd_ms > 0 ? rows / (fwd_ms / 1000.0) : 0.0,
                     steps ? fwd_ms / steps : 0.0);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
