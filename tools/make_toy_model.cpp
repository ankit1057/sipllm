// make_toy_model — generate a tiny random Llama fixture for experimentation.
//
//   make_toy_model out.llmw                 (fp32 .llmw)
//   make_toy_model out.gguf --gguf          (fp32 GGUF, with byte tokenizer)
//   make_toy_model out.gguf --gguf --q8     (Q8_0-quantized GGUF)
//   flags: --layers N --dim D --heads H --kv K --ffn F --vocab V --ctx C
#include "llm/gguf_writer.h"
#include "llm/toy_model.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace llm;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: make_toy_model <out> [--gguf] [--q8] "
                        "[--layers N --dim D --heads H --kv K --ffn F --vocab V --ctx C]\n");
        return 2;
    }
    std::string out = argv[1];
    bool gguf = false, q8 = false;
    ToyConfig tc; ToyGgufConfig gc;
    auto set = [&](int64_t& a, int64_t& b, const char* v) { a = b = std::stoll(v); };

    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]{ return (i + 1 < argc) ? argv[++i] : "0"; };
        if (a == "--gguf") gguf = true;
        else if (a == "--q8") { q8 = true; gguf = true; }
        else if (a == "--layers") set(tc.n_layers, gc.n_layers, nx());
        else if (a == "--dim") set(tc.dim, gc.dim, nx());
        else if (a == "--heads") set(tc.n_heads, gc.n_heads, nx());
        else if (a == "--kv") set(tc.n_kv_heads, gc.n_kv_heads, nx());
        else if (a == "--ffn") set(tc.ffn_dim, gc.ffn_dim, nx());
        else if (a == "--vocab") set(tc.vocab_size, gc.vocab_size, nx());
        else if (a == "--ctx") set(tc.ctx_len, gc.ctx_len, nx());
    }

    try {
        if (gguf) {
            gc.weight_type = q8 ? DType::Q8_0 : DType::F32;
            gc.with_tokenizer = true;
            write_toy_gguf(out, gc);
            printf("wrote GGUF %s (%s, %lld layers, dim %lld)\n", out.c_str(),
                   q8 ? "Q8_0" : "F32", (long long)gc.n_layers, (long long)gc.dim);
        } else {
            write_toy_model(out, tc);
            printf("wrote .llmw %s (%lld layers, dim %lld)\n", out.c_str(),
                   (long long)tc.n_layers, (long long)tc.dim);
        }
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
