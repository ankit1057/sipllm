// fast_accuracy.cpp — how far is an approximate kernel path (--fast) from the
// exact fp32-dequant path, measured on a whole model?
//
// Teacher-forces a fixed passage (or --text FILE) through the model once per
// mode, collecting the logits at EVERY position in one batched pass
// (Transformer::spec_forward: same layer kernels and chunking as prefill, then
// the LM head per position). Against the exact path it reports, per position:
//   KL(exact || approx) of the next-token distributions (nats),
//   whether the top-1 token agrees, and the largest absolute logit difference.
//
//   fast_accuracy <model.gguf> [--tokens N] [--text FILE] [--threads T]
#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/neon.h"
#include "llm/runtime.h"
#include "llm/sampler.h"
#include "llm/tokenizer.h"
#include "llm/transformer.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace llm;

static const char* kPassage =
    "The lighthouse keeper climbed the spiral stairs every evening at dusk, counting each of the one "
    "hundred and twelve steps as his father had taught him. At the top he trimmed the wick, polished the "
    "great lens, and wrote the weather in a ledger that had been kept without a gap for ninety years. "
    "Ships no longer needed the light; satellites told them where the rocks were. Still, the harbour "
    "council voted each spring to keep it burning, partly from habit and partly because the town's "
    "children liked to see it from their bedroom windows. In the winter of the great storm the power "
    "failed across the whole coast. Radios went silent, and the satellite receivers on the fishing boats "
    "showed nothing but error messages. Three boats were still out beyond the point. The keeper lit the "
    "old oil lamp, wound the clockwork that turned the lens, and stood beside it through the night. By "
    "morning all three crews were home, wet and frightened and grateful. Nobody voted on the light "
    "after that. A simple tool that works without a network, a battery, or a subscription turned out "
    "to matter more than anyone had guessed, and the ledger gained a new line in a steady hand: wind "
    "north east, very strong, visibility poor, light kept all night, all boats returned.";

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [--tokens N] [--text FILE] [--threads T]\n", argv[0]);
        return 2;
    }
    std::string path = argv[1], text = kPassage;
    int64_t want = 256;
    int threads = 4;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tokens" && i + 1 < argc) want = std::atoll(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
        else if (a == "--text" && i + 1 < argc) {
            std::ifstream f(argv[++i]); std::stringstream ss; ss << f.rdbuf(); text = ss.str();
        }
    }
    try {
        auto src = open_model(path, false);
        const ModelConfig cfg = ModelConfig::from_source(*src);
        const Tokenizer tok = Tokenizer::from_source(*src);
        std::vector<int64_t> ids = tok.encode(text, true);
        while ((int64_t)ids.size() < want) {            // repeat the passage if it is short
            std::vector<int64_t> more = tok.encode(text, false);
            ids.insert(ids.end(), more.begin(), more.end());
        }
        ids.resize((size_t)want);
        const int64_t n = (int64_t)ids.size(), V = cfg.vocab_size;

        LayerLoader::Options opt;          // fully resident so both modes read the same bytes
        ThreadPool pool(threads);
        opt.dequant_pool = &pool;
        LayerLoader loader(src.get(), cfg, opt);
        auto run = [&](bool fast) {
            set_fast_quant(fast);
            KVCache kv(cfg.n_layers, cfg.kv_dim(), std::max<int64_t>(n, 64));
            Transformer tf(&loader, &kv, &pool);
            const float* all = tf.spec_forward(ids.data(), n, 0);
            return std::vector<float>(all, all + (size_t)n * V);
        };
        const std::vector<float> ex = run(false);
        const std::vector<float> ap = run(true);
        const bool fast_active = fast_quant_enabled();

        std::vector<double> kls;
        int64_t agree = 0;
        double max_abs = 0, sum_max_abs = 0;
        std::vector<double> p(V), q(V);
        for (int64_t i = 0; i < n; ++i) {
            const float* a = ex.data() + (size_t)i * V;
            const float* b = ap.data() + (size_t)i * V;
            const float ma = *std::max_element(a, a + V), mb = *std::max_element(b, b + V);
            double za = 0, zb = 0, mx = 0;
            for (int64_t v = 0; v < V; ++v) {
                p[v] = std::exp((double)a[v] - ma); za += p[v];
                q[v] = std::exp((double)b[v] - mb); zb += q[v];
                mx = std::max(mx, (double)std::fabs(a[v] - b[v]));
            }
            double kl = 0;
            for (int64_t v = 0; v < V; ++v) {
                const double pv = p[v] / za, qv = q[v] / zb;
                if (pv > 0) kl += pv * (std::log(pv) - std::log(std::max(qv, 1e-300)));
            }
            kls.push_back(kl);
            agree += Sampler::argmax(a, V) == Sampler::argmax(b, V);
            max_abs = std::max(max_abs, mx);
            sum_max_abs += mx;
        }
        std::vector<double> s = kls;
        std::sort(s.begin(), s.end());
        double mean = 0; for (double k : kls) mean += k; mean /= (double)n;
        printf("model: %s  tokens: %lld  vocab: %lld  int8 kernels active: %s\n", path.c_str(),
               (long long)n, (long long)V, fast_active ? "yes" : "NO (fast path fell back to exact)");
        printf("KL(exact||fast) nats: mean %.6f  p50 %.6f  p99 %.6f  max %.6f\n", mean,
               s[s.size() / 2], s[(size_t)(0.99 * (s.size() - 1))], s.back());
        printf("top-1 agreement: %.2f%% (%lld/%lld)\n", 100.0 * agree / n, (long long)agree, (long long)n);
        printf("max |logit diff|: max %.4f  mean over positions %.4f\n", max_abs, sum_max_abs / n);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
