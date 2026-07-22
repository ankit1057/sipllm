// main.cpp — command-line driver for the streaming inference engine.
//
//   llm <model.gguf|model.llmw> [-p "prompt"] [-n tokens] [-t temp]
//       [--residency fp32|quant] [--mmap] [--no-async] [--buffers N]
//       [--ctx N] [--threads N] [--seed S] [--greedy]
//
// Streams tokens to stdout as they are produced and prints a stats block.
#include "llm/runtime.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>

using namespace llm;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr,
            "usage: %s <model> [-p prompt] [-n tokens] [-t temp]\n"
            "          [--residency fp32|quant] [--mmap] [--no-async]\n"
            "          [--buffers N] [--ctx N] [--threads N] [--seed S] [--greedy]\n",
            argv[0]);
        return 2;
    }
    std::string model = argv[1];
    std::string prompt = "Hello";
    int max_new = 64, threads = 0, buffers = 2, ctx = 0;
    SamplerConfig scfg;
    LayerLoader::Options opt;

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* def) -> std::string {
            return (i + 1 < argc) ? argv[++i] : def;
        };
        if (a == "-p") prompt = next("");
        else if (a == "-n") max_new = std::stoi(next("64"));
        else if (a == "-t") scfg.temperature = std::stof(next("0.8"));
        else if (a == "--seed") scfg.seed = std::stoull(next("1"));
        else if (a == "--greedy") scfg.temperature = 0.f;
        else if (a == "--residency") opt.residency = (next("quant") == "fp32") ? Residency::FP32 : Residency::Quantized;
        else if (a == "--mmap") opt.use_mmap = true;
        else if (a == "--no-async") { opt.async = false; opt.n_buffers = 1; }
        else if (a == "--buffers") buffers = std::stoi(next("2"));
        else if (a == "--ctx") ctx = std::stoi(next("0"));
        else if (a == "--threads") threads = std::stoi(next("0"));
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (opt.async) opt.n_buffers = buffers;

    try {
        double t0 = now_sec();
        auto src = open_model(model, opt.use_mmap);
        Runtime rt(std::move(src), opt, ctx, threads);
        double load_s = now_sec() - t0;

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

        fprintf(stderr,
            "\n\n--- stats ---\n"
            "load:            %.3f s\n"
            "prompt tokens:   %d\n"
            "generated:       %d\n"
            "TTFT:            %.3f s\n"
            "prefill:         %.2f tok/s\n"
            "decode:          %.2f tok/s\n"
            "weights resident:%.1f MB\n"
            "kv cache:        %.1f MB\n"
            "streamed:        %.1f MB (from disk)\n"
            "prefetch:        %" PRIu64 " hits / %" PRIu64 " misses\n"
            "context:         %d / %d\n",
            st.load_s, st.prompt_tokens, st.gen_tokens, st.ttft_s,
            st.prefill_tok_s, st.decode_tok_s,
            st.weights_resident_bytes / 1e6, st.kv_bytes / 1e6,
            st.bytes_read / 1e6, st.prefetch_hits, st.prefetch_misses,
            st.ctx_used, st.ctx_max);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "\nerror: %s\n", e.what());
        return 1;
    }
}
