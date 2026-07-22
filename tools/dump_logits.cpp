// dump_logits — golden-test harness (Task 13).
//
// Runs a prompt through the engine and prints, in a stable diff-friendly form:
//   * the per-block hidden-state signature (L2 norm + checksum) after each layer
//   * the top-K final logits (token id + value)
//
// To validate against llama.cpp: build llama.cpp, run the SAME model + prompt,
// and compare. The final top-K token ids should match; hidden-state norms
// should agree to within quantization error (a few percent for Q4_K_M, ~1e-3
// for fp16/Q8_0). Divergence that grows layer-by-layer localizes the bug.
//
//   dump_logits <model> [-p prompt] [-k topk] [--no-layers]
#include "llm/runtime.h"
#include "llm/transformer.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace llm;

// FNV-1a over raw fp32 bytes — a determinism/regression fingerprint for OUR
// engine across runs. NOTE: this will NOT match llama.cpp bit-for-bit even when
// both are correct (summation order / rounding differ); the cross-engine check
// is the tolerance-based numeric diff in golden_compare.py.
static uint64_t fnv1a(const float* v, int64_t n) {
    uint64_t h = 1469598103934665603ULL;
    const uint8_t* p = reinterpret_cast<const uint8_t*>(v);
    for (int64_t i = 0; i < n * (int64_t)sizeof(float); ++i) {
        h ^= p[i]; h *= 1099511628211ULL;
    }
    return h;
}

// Binary dump format for cross-engine comparison:
//   magic "LGDN" | u32 n_layers | u32 dim | u32 vocab
//   n_layers × (dim × f32)   -- residual stream after each block (last pos)
//   vocab × f32              -- final logits (first prediction)
static void write_dump(const std::string& path,
                       const std::vector<std::vector<float>>& hidden,
                       const std::vector<float>& logits, int dim) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return;
    uint32_t magic = 0x4E44474C, nl = (uint32_t)hidden.size(),
             d = (uint32_t)dim, vc = (uint32_t)logits.size();
    std::fwrite(&magic, 4, 1, f); std::fwrite(&nl, 4, 1, f);
    std::fwrite(&d, 4, 1, f); std::fwrite(&vc, 4, 1, f);
    for (auto& h : hidden) std::fwrite(h.data(), 4, h.size(), f);
    std::fwrite(logits.data(), 4, logits.size(), f);
    std::fclose(f);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: dump_logits <model> [-p prompt] [-k K] [--no-layers]\n"); return 2; }
    std::string model = argv[1], prompt = "The capital of France is", raw_out;
    int topk = 8; bool layers = true;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-p" && i + 1 < argc) prompt = argv[++i];
        else if (a == "-k" && i + 1 < argc) topk = std::stoi(argv[++i]);
        else if (a == "--no-layers") layers = false;
        else if (a == "--raw" && i + 1 < argc) raw_out = argv[++i];
    }
    try {
        auto src = open_model(model);
        LayerLoader::Options opt; opt.residency = Residency::Quantized;
        Runtime rt(std::move(src), opt);
        printf("# model:  %s\n# config: %s\n# prompt: \"%s\"\n",
               model.c_str(), rt.config().summary().c_str(), prompt.c_str());

        // Hidden-state signatures for the LAST prompt position only (the one
        // that produces the first logits) keep output compact and comparable.
        // We collect the final forward's per-layer hidden states.
        std::vector<std::pair<double,double>> sig;  // (l2, checksum) per layer
        std::vector<uint64_t> hashes;                // per-layer FNV-1a
        std::vector<std::vector<float>> hidden;      // full per-layer state (last pos)
        // We need access to the Transformer's hook; Runtime hides it, so we run
        // through the public generate() with 0 new tokens after installing the
        // hook via a thin re-open is not possible. Instead expose via a tiny
        // dedicated path: generate 1 token and capture layer sigs of last pos.

        // Enable a hidden hook by generating with profiling off but hook on.
        // Runtime doesn't forward the hook, so use a fresh Transformer path:
        // simplest is to add the hook through Runtime — see note below.
        GenStats st;
        // We piggyback: install hook via Runtime accessor if available.
        //
        // The hook fires at the end of every block for EVERY position we run
        // through the network, and we keep the last write per layer. To match
        // llama.cpp's golden dump (which captures each block's residual at the
        // LAST PROMPT token) we must generate ZERO new tokens: prefill alone
        // leaves the hook holding position N-1 (the final prompt token). If we
        // generated even one token, a decode-step forward pass would fire the
        // hook again at position N (the freshly generated token) and every
        // per-layer state would be compared one token out of phase — which
        // shows up as low per-layer cosine even when the engine is exact.
        rt.set_hidden_hook([&](int layer, const float* x, int64_t dim) {
            double l2 = 0, cs = 0;
            for (int64_t i = 0; i < dim; ++i) { l2 += (double)x[i]*x[i]; cs += x[i]; }
            if ((int)sig.size() <= layer) { sig.resize(layer + 1, {0,0}); hashes.resize(layer + 1, 0); hidden.resize(layer + 1); }
            sig[layer] = { std::sqrt(l2), cs };            // overwrite -> keeps last prompt pos
            hashes[layer] = fnv1a(x, dim);
            hidden[layer].assign(x, x + dim);
        });

        rt.generate(prompt, 0, SamplerConfig{/*greedy via temp0*/ 0.f}, nullptr, &st);
        // Greedy first token from the last-prompt-position logits (argmax),
        // i.e. exactly the prediction llama.cpp's final logits describe.
        const std::vector<float>& lg0 = rt.first_logits();
        int64_t greedy = (int64_t)(std::max_element(lg0.begin(), lg0.end()) - lg0.begin());
        std::string out = rt.tokenizer().decode_token(greedy);

        if (layers) {
            printf("\n# per-layer hidden state (final prompt position)\n");
            printf("# %-6s %14s %14s   %-18s\n", "layer", "l2_norm", "checksum", "fnv1a_hash");
            for (size_t l = 0; l < sig.size(); ++l)
                printf("  %-6zu %14.6f %14.6f   %016llx\n", l, sig[l].first, sig[l].second,
                       (unsigned long long)hashes[l]);
        }

        // Top-K logits at the first prediction position — the primary check
        // to diff against llama.cpp's logits for the same prompt.
        const std::vector<float>& lg = rt.first_logits();
        std::vector<int64_t> idx(lg.size());
        for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int64_t)i;
        int k = std::min<int>(topk, (int)idx.size());
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int64_t a, int64_t b) { return lg[a] > lg[b]; });
        printf("\n# top-%d logits at first prediction position\n", k);
        printf("# %-8s %14s   %s\n", "token_id", "logit", "text");
        for (int i = 0; i < k; ++i) {
            std::string t = rt.tokenizer().decode_token(idx[i]);
            printf("  %-8lld %14.6f   %s\n", (long long)idx[i], lg[idx[i]], t.c_str());
        }
        printf("\n# final logits fnv1a: %016llx\n",
               (unsigned long long)fnv1a(lg.data(), (int64_t)lg.size()));
        printf("# greedy first token: \"%s\"\n", out.c_str());

        if (!raw_out.empty()) {
            write_dump(raw_out, hidden, lg, (int)rt.config().dim);
            printf("# wrote raw dump for cross-engine compare: %s\n", raw_out.c_str());
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
