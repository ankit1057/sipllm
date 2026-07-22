// inspect_gguf — Phase 5 deliverable: parse a GGUF and print, without running
// any inference, its metadata and full tensor table (name, type, shape, offset).
#include "llm/gguf.h"
#include "llm/model.h"

#include <cinttypes>
#include <cstdio>

using namespace llm;

static void print_meta(const char* key, const MetaValue& m) {
    switch (m.kind) {
        case MetaValue::Kind::Int:   printf("  %-40s = %lld\n", key, (long long)m.i); break;
        case MetaValue::Kind::Float: printf("  %-40s = %g\n", key, m.f); break;
        case MetaValue::Kind::Str:   printf("  %-40s = \"%s\"\n", key, m.s.c_str()); break;
        case MetaValue::Kind::IntArr:   printf("  %-40s = [int; %zu]\n", key, m.ia.size()); break;
        case MetaValue::Kind::FloatArr: printf("  %-40s = [float; %zu]\n", key, m.fa.size()); break;
        case MetaValue::Kind::StrArr:   printf("  %-40s = [str; %zu]\n", key, m.sa.size()); break;
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: inspect_gguf <model.gguf>\n"); return 2; }
    try {
        GgufFile g(argv[1]);
        printf("GGUF v%u  file_size=%.2f MB  data_offset=%" PRIu64 "\n",
               g.version(), g.file_size() / 1e6, g.data_offset());

        printf("\n== metadata (well-known keys) ==\n");
        // We don't expose the map directly; print the well-known keys + counts.
        static const char* keys[] = {
            "general.architecture", "general.name", "general.file_type",
            "llama.block_count", "llama.embedding_length", "llama.feed_forward_length",
            "llama.attention.head_count", "llama.attention.head_count_kv",
            "llama.context_length", "llama.rope.freq_base",
            "llama.attention.layer_norm_rms_epsilon",
            "tokenizer.ggml.model", "tokenizer.ggml.tokens",
            "tokenizer.ggml.merges", "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
        };
        for (const char* k : keys) if (const MetaValue* m = g.meta(k)) print_meta(k, *m);

        ModelConfig cfg = ModelConfig::from_source(g);
        printf("\n== derived config ==\n  %s\n", cfg.summary().c_str());

        printf("\n== tensors (%zu) ==\n", g.tensors().size());
        uint64_t total = 0;
        for (const auto& t : g.tensors()) {
            char shape[64] = {0}; int off = 0;
            for (size_t i = 0; i < t.shape.size(); ++i)
                off += snprintf(shape + off, sizeof(shape) - off, "%s%lld",
                                i ? "x" : "", (long long)t.shape[i]);
            printf("  %-28s %-6s [%-14s] off=%-12" PRIu64 " %8.2f KB\n",
                   t.name.c_str(), dtype_name(t.dtype), shape, t.offset, t.nbytes / 1024.0);
            total += t.nbytes;
        }
        printf("\ntotal tensor bytes: %.2f MB across %zu tensors\n",
               total / 1e6, g.tensors().size());
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
