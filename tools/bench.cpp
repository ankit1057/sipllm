// bench — profiler + micro-benchmarks (Task 13).
//
// Prints:
//   * memory profile:  RSS after each layer + peak (proves streaming works —
//     RSS stays flat instead of climbing to the full model size)
//   * storage profile: per-layer bytes / io_ms / dequant_ms / compute_ms with an
//     ASCII bar (the "live visualization" in the terminal)
//   * throughput:      prefill and decode tokens/sec
//   * mmap vs pread:   the same run under both backends, side by side
//
//   bench <model> [-p prompt] [-n tokens]
#include "llm/runtime.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace llm;

static void bar(double v, double vmax, int width) {
    int n = vmax > 0 ? (int)(v / vmax * width + 0.5) : 0;
    for (int i = 0; i < width; ++i) putchar(i < n ? '#' : ' ');
}

static GenStats run_once(const std::string& model, const std::string& prompt,
                         int n, bool use_mmap, bool print_layers) {
    LayerLoader::Options opt;
    opt.residency = Residency::Quantized;
    opt.use_mmap = use_mmap;
    opt.async = true;
    opt.n_buffers = 2;
    auto src = open_model(model, use_mmap);
    Runtime rt(std::move(src), opt);

    // capture the last decode token's per-layer timing for the viz
    std::vector<Transformer::LayerTiming> last;
    size_t peak = 0;
    rt.set_profile_sink([&](int, const std::vector<Transformer::LayerTiming>& t, size_t pk) {
        last = t; peak = pk;
    });

    GenStats st;
    rt.generate(prompt, n, SamplerConfig{0.f}, nullptr, &st);

    if (print_layers && !last.empty()) {
        double cmax = 0, iomax = 0;
        for (auto& t : last) { cmax = std::max(cmax, t.compute_ms); iomax = std::max(iomax, t.io_ms + t.dequant_ms); }
        printf("\nper-layer profile (last decode token, %s):\n", use_mmap ? "mmap" : "pread");
        printf("%-5s %10s %8s %8s %9s  %-20s %-16s %8s\n",
               "layer", "bytes", "io_ms", "deq_ms", "cmp_ms", "compute", "io+deq", "rss_MB");
        for (size_t l = 0; l < last.size(); ++l) {
            const auto& t = last[l];
            printf("%-5zu %10.0f %8.2f %8.2f %9.2f  ", l,
                   0.0, t.io_ms, t.dequant_ms, t.compute_ms);
            bar(t.compute_ms, cmax, 20); printf(" ");
            bar(t.io_ms + t.dequant_ms, iomax, 16);
            printf(" %8.1f\n", t.rss_bytes / 1e6);
        }
        printf("peak RSS: %.1f MB\n", peak / 1e6);
    }
    return st;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: bench <model> [-p prompt] [-n tokens]\n"); return 2; }
    std::string model = argv[1], prompt = "Once upon a time";
    int n = 32;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-p" && i + 1 < argc) prompt = argv[++i];
        else if (a == "-n" && i + 1 < argc) n = std::stoi(argv[++i]);
    }
    try {
        printf("=== pread backend ===\n");
        GenStats a = run_once(model, prompt, n, /*mmap=*/false, /*layers=*/true);
        printf("\n=== mmap backend ===\n");
        GenStats b = run_once(model, prompt, n, /*mmap=*/true, /*layers=*/false);

        printf("\n=== summary ===\n");
        printf("%-10s %12s %12s %12s %12s %12s\n",
               "backend", "prefill t/s", "decode t/s", "streamed MB", "resident MB", "prefetch");
        printf("%-10s %12.2f %12.2f %12.1f %12.1f %6llu/%llu\n", "pread",
               a.prefill_tok_s, a.decode_tok_s, a.bytes_read / 1e6,
               a.weights_resident_bytes / 1e6,
               (unsigned long long)a.prefetch_hits, (unsigned long long)a.prefetch_misses);
        printf("%-10s %12.2f %12.2f %12.1f %12.1f %6llu/%llu\n", "mmap",
               b.prefill_tok_s, b.decode_tok_s, b.bytes_read / 1e6,
               b.weights_resident_bytes / 1e6,
               (unsigned long long)b.prefetch_hits, (unsigned long long)b.prefetch_misses);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
