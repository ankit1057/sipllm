// main.cpp — command-line driver for the streaming inference engine.
//
//   llm <model.gguf|model.llmw> [-p "prompt"] [-n tokens] [-t temp]
//       [--residency fp32|quant] [--mmap] [--no-async] [--buffers N]
//       [--ctx N] [--threads N] [--seed S] [--greedy]
//
// Streams tokens to stdout as they are produced and prints a stats block.
#include "llm/runtime.h"
#include "llm/device_profile.h"
#include "llm/auto_tuner.h"
#include "llm/plugin.h"
#include "llm/kosh.h"
#include "llm/semantic_cache.h"
#include "llm/rtk.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <cctype>
#include <string>

using namespace llm;

// Parse a byte count with an optional K/M/G (or KB/MB/GB) suffix: "512M", "1.5G",
// "268435456", "0". Used by --ram-budget (#37).
static size_t parse_bytes(const std::string& in) {
    if (in.empty()) return 0;
    std::string t = in;
    if (t.size() >= 2 && (t.back() == 'B' || t.back() == 'b')) t.pop_back();
    size_t mult = 1;
    if (!t.empty()) {
        switch (std::toupper((unsigned char)t.back())) {
            case 'K': mult = 1024ull; t.pop_back(); break;
            case 'M': mult = 1024ull * 1024; t.pop_back(); break;
            case 'G': mult = 1024ull * 1024 * 1024; t.pop_back(); break;
            default: break;
        }
    }
    if (t.empty()) return 0;
    return (size_t)(std::stod(t) * (double)mult);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <model> [-p prompt] [-n tokens] [-t temp]\n"
            "          [--top-k K] [--top-p P] [--repeat-penalty R] [--repeat-last-n N]\n"
            "          [--residency fp32|quant] [--mmap] [--no-async] [--stream-lm-head]\n"
            "          [--buffers N] [--ctx N] [--threads N] [--seed S] [--greedy] [--schedule P]\n"
            "          [--ram-budget BYTES|N{K,M,G}] [--fast]\n"
            "          [--kosh] [--kosh-max-run N] [--rtk] [--reuse]\n"
            "          [--save-session F] [--load-session F]\n",
            argv[0]);
        return 2;
    }
    std::string model = argv[1];
    std::string prompt = "Hello";
    int max_new = 64, threads = 0, buffers = 2, ctx = 0;
    size_t ram_budget = 0;   // #37: total peak-RSS target (0 = unlimited)
    bool force_budget = false;
    bool kv_q8 = false;
    AutoTunerOptions tuner_opt;
    bool schedule_overridden = false;
    bool threads_overridden = false;
    ThreadPool::SchedulePolicy schedule_policy = ThreadPool::SchedulePolicy::Proportional2;
    SamplerConfig scfg;
    LayerLoader::Options opt;
    bool use_kosh = false, use_rtk = false;
    int  kosh_max_run = 8;
    size_t semantic_cache_budget = 0;
    bool use_reuse = false;
    std::string save_path, load_path;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* def) -> std::string {
            return (i + 1 < argc) ? argv[++i] : def;
        };
        if (a == "-p") prompt = next("");
        else if (a == "-n") max_new = std::stoi(next("64"));
        else if (a == "-t") scfg.temperature = std::stof(next("0.8"));
        else if (a == "--top-k") scfg.top_k = std::stoi(next("40"));
        else if (a == "--top-p") scfg.top_p = std::stof(next("0.95"));
        else if (a == "--repeat-penalty") scfg.repeat_penalty = std::stof(next("1.1"));
        else if (a == "--repeat-last-n") scfg.repeat_last_n = std::stoi(next("64"));
        else if (a == "--seed") scfg.seed = std::stoull(next("1"));
        else if (a == "--greedy") scfg.temperature = 0.f;
        else if (a == "--residency") opt.residency = (next("quant") == "fp32") ? Residency::FP32 : Residency::Quantized;
        else if (a == "--mmap") opt.use_mmap = true;
        else if (a == "--no-async") { opt.async = false; opt.n_buffers = 1; }
        else if (a == "--stream-lm-head") opt.stream_lm_head = true;
        else if (a == "--buffers") buffers = std::stoi(next("2"));
        else if (a == "--ram-budget") ram_budget = parse_bytes(next("0"));
        else if (a == "--ram-budget-force") force_budget = true;
        else if (a == "--kv-q8") kv_q8 = true;
        else if (a == "--fast") opt.fast_quant = true;
        else if (a == "--ctx") ctx = std::stoi(next("0"));
        else if (a == "--schedule") {
            std::string s = next("proportional2");
            if (s == "static") schedule_policy = ThreadPool::SchedulePolicy::Static;
            else if (s == "fixed8") schedule_policy = ThreadPool::SchedulePolicy::Fixed8;
            else if (s == "fixed16") schedule_policy = ThreadPool::SchedulePolicy::Fixed16;
            else if (s == "fixed32") schedule_policy = ThreadPool::SchedulePolicy::Fixed32;
            else if (s == "proportional2") schedule_policy = ThreadPool::SchedulePolicy::Proportional2;
            else if (s == "proportional4") schedule_policy = ThreadPool::SchedulePolicy::Proportional4;
            else if (s == "adaptive") schedule_policy = ThreadPool::SchedulePolicy::Adaptive;
            else { fprintf(stderr, "unknown schedule: %s\n", s.c_str()); return 2; }
            schedule_overridden = true;
        }
        else if (a == "--threads") {
            std::string t = next("0");
            if (t != "auto") {
                threads = std::stoi(t);
                threads_overridden = true;
            }
        }
        else if (a == "--recalibrate") tuner_opt.force_recalibrate = true;
        else if (a == "--no-autotune") tuner_opt.disable_autotune = true;
        else if (a == "--semantic-cache-budget") semantic_cache_budget = parse_bytes(next("0"));
        else if (a == "--kosh") use_kosh = true;
        else if (a == "--kosh-max-run") kosh_max_run = std::stoi(next("8"));
        else if (a == "--rtk") use_rtk = true;
        else if (a == "--reuse") use_reuse = true;
        else if (a == "--save-session") { save_path = next(""); use_reuse = true; }
        else if (a == "--load-session") { load_path = next(""); use_reuse = true; }
        else if (a == "--profile-info") {
            HardwareInfo hw = get_hardware_info();
            printf("Hardware ID: %s\n", hw.hardware_id().c_str());
            printf("%s\n", hw.to_json(2).c_str());
            return 0;
        }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (opt.async) opt.n_buffers = buffers;

    try {
        double t0 = now_sec();
        
        HardwareInfo hw = get_hardware_info();
        RuntimeProfile rp = tune_if_needed(hw, tuner_opt);
        if (!threads_overridden && rp.threads > 0) threads = rp.threads;
        if (!schedule_overridden && rp.schedule_policy >= 0) schedule_policy = (ThreadPool::SchedulePolicy)rp.schedule_policy;
        
        auto src = open_model(model, opt.use_mmap);
        KVPrecision kv_precision = kv_q8 ? KVPrecision::Q8_0 : KVPrecision::FP32;
        Runtime rt(std::move(src), opt, ctx, threads, ram_budget, force_budget, kv_precision);
        if (rt.thread_pool()) rt.thread_pool()->set_policy(schedule_policy);
        double load_s = now_sec() - t0;

        // Optional plugin seam: Kosh (context) + RTK (runtime/KV). Default OFF
        // ⇒ generate() is byte-identical to the plugin-free engine.
        PluginHost plugins;
        if (use_kosh) plugins.set_kosh(make_kosh_v0(kosh_max_run));
        if (use_rtk)  plugins.set_rtk(make_rtk_v0());
        if (use_kosh || use_rtk) rt.set_plugins(&plugins);
        if (use_reuse) rt.set_context_reuse(true);
        if (semantic_cache_budget > 0) rt.enable_semantic_cache(semantic_cache_budget);
        if (!load_path.empty() && !rt.load_session(load_path))
            fprintf(stderr, "warning: could not load session '%s'\n", load_path.c_str());

        fprintf(stderr, "model: %s\nconfig: %s\ntokenizer: %s vocab=%lld\n",
                model.c_str(), rt.config().summary().c_str(),
                rt.tokenizer().kind() == Tokenizer::Kind::BPE ? "BPE" :
                rt.tokenizer().kind() == Tokenizer::Kind::SentencePiece ? "SPM" : "byte",
                (long long)rt.tokenizer().vocab_size());
        fprintf(stderr, "residency=%s async=%d buffers=%d mmap=%d\n\n",
                opt.residency == Residency::FP32 ? "fp32" : "quant",
                (int)opt.async, opt.n_buffers, (int)opt.use_mmap);

        printf("%s", prompt.c_str());
        fflush(stdout);
        GenStats st;
        rt.generate(prompt, max_new, scfg,
                    [](const std::string& piece, int64_t) {
                        printf("%s", piece.c_str()); fflush(stdout); return true;
                    }, &st);
        st.load_s = load_s;
        if (!save_path.empty() && !rt.save_session(save_path))
            fprintf(stderr, "warning: could not save session '%s'\n", save_path.c_str());

        size_t rss = current_rss_bytes();
        char budget_note[64] = "";
        if (ram_budget) snprintf(budget_note, sizeof(budget_note), "   (budget %.0f MB)", ram_budget / 1e6);
        fprintf(stderr,
            "\n\xE2\x94\x80\xE2\x94\x80 sipllm \xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\xE2\x94\x80\n"
            "peak rss:        %.0f MB\n"
            "pinned layers:   %d / %d%s\n"
            "fast kernel:     %s\n"
            "decode:          %.2f tok/s\n"
            "prefill:         %.2f tok/s\n"
            "TTFT:            %.3f s\n"
            "prompt tokens:   %d\n"
            "generated:       %d\n"
            "weights resident:%.1f MB\n"
            "kv cache:        %.1f MB\n"
            "streamed:        %.1f MB (from disk)\n"
            "prefetch:        %" PRIu64 " hits / %" PRIu64 " misses\n"
            "context:         %d / %d\n"
            "sched chunks:    %" PRIu64 "\n"
            "sched steals:    %" PRIu64 "\n"
            "sched idle:      %.2f ms\n"
            "sched barrier:   %.2f ms\n",
            rss / 1e6,
            st.pinned_layers, (int)rt.config().n_layers, budget_note,
            opt.fast_quant ? "on" : "off",
            st.decode_tok_s, st.prefill_tok_s, st.ttft_s,
            st.prompt_tokens, st.gen_tokens,
            st.weights_resident_bytes / 1e6, st.kv_bytes / 1e6,
            st.bytes_read / 1e6, st.prefetch_hits, st.prefetch_misses,
            st.ctx_used, st.ctx_max,
            rt.thread_pool()->stats.chunks_processed.load(),
            rt.thread_pool()->stats.steals.load(),
            rt.thread_pool()->stats.idle_time_us.load() / 1000.0,
            rt.thread_pool()->stats.barrier_time_us.load() / 1000.0);
        if (st.kosh_active || st.rtk_active || st.reuse_active || st.semantic_cache_hits > 0) {
            if (st.semantic_cache_hits > 0) {
                fprintf(stderr, "semantic cache:  %llu hits\n", (unsigned long long)st.semantic_cache_hits);
            }
            fprintf(stderr, "── plugins ───────────────\n");
            if (st.kosh_active) {
                double pct = st.kosh_tokens_in > 0
                    ? 100.0 * (double)(st.kosh_tokens_in - st.kosh_tokens_out) / (double)st.kosh_tokens_in
                    : 0.0;
                fprintf(stderr, "kosh:            %lld -> %lld tok (%.1f%% reduced)\n",
                        (long long)st.kosh_tokens_in, (long long)st.kosh_tokens_out, pct);
            }
            if (st.rtk_active) {
                fprintf(stderr, "rtk:             %llu steps, max seq %lld, peak kv %.1f MB\n",
                        (unsigned long long)st.rtk_steps, (long long)st.rtk_max_seq,
                        st.rtk_peak_kv_bytes / 1e6);
            }
            if (st.reuse_active) {
                fprintf(stderr, "reuse:           %d reused / %d processed tokens\n",
                        st.reused_prefix_tokens, st.processed_tokens);
            }
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
}
