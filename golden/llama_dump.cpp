// llama_dump.cpp — the llama.cpp side of the golden test.
//
// Runs a prompt through llama.cpp and writes the SAME LGDN dump format our
// engine emits (tools/dump_logits --raw), so golden_compare.py can diff them
// layer-by-layer. It captures each block's residual output ("l_out-<il>") via
// the eval callback and the final logits via llama_get_logits.
//
// Build (after building llama.cpp; adjust paths):
//   g++ -std=c++17 -O2 golden/llama_dump.cpp \
//       -I <llama.cpp> -I <llama.cpp>/ggml/include \
//       -L <llama.cpp>/build/src -L <llama.cpp>/build/ggml/src \
//       -lllama -lggml -lggml-base -lpthread -o build/llama_dump
//
// Run:
//   ./build/llama_dump <model.gguf> "prompt" llama.dump
//
// NOTE: llama.cpp's C API evolves; if a symbol is missing, check its current
// llama.h. The logic (decode prompt, grab l_out-<il> + final logits) is stable.
#include "llama.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

static int g_n_embd = 0, g_n_tokens = 0;
static std::map<int, std::vector<float>> g_hidden;   // il -> last-token residual

static bool cb_eval(struct ggml_tensor* t, bool ask, void* /*ud*/) {
    if (ask) {
        // Want per-layer residual outputs named "l_out-<il>".
        return std::strncmp(t->name, "l_out-", 6) == 0;
    }
    int il = std::atoi(t->name + 6);
    int ne0 = (int)t->ne[0];      // n_embd
    int ne1 = (int)t->ne[1];      // n_tokens
    g_n_embd = ne0; g_n_tokens = ne1;
    std::vector<float> col(ne0);
    // last token column: offset (ne1-1)*ne0 floats (ggml ne[0] is fastest dim)
    size_t off = (size_t)(ne1 - 1) * ne0 * sizeof(float);
    ggml_backend_tensor_get(t, col.data(), off, ne0 * sizeof(float));
    g_hidden[il] = std::move(col);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: llama_dump <model> <prompt> <out.dump>\n"); return 2; }
    const char* model_path = argv[1];
    std::string prompt = argv[2];
    const char* out = argv[3];

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;   // pure CPU, to match our engine's reference path
    llama_model* model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.cb_eval = cb_eval;
    cp.cb_eval_user_data = nullptr;
    cp.embeddings = false;
    llama_context* ctx = llama_init_from_model(model, cp);

    const llama_vocab* vocab = llama_model_get_vocab(model);
    // tokenize (add BOS)
    std::vector<llama_token> toks(prompt.size() + 8);
    int n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                           toks.data(), (int)toks.size(), /*add_special=*/true,
                           /*parse_special=*/true);
    if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), toks.data(), (int)toks.size(), true, true); }
    toks.resize(n);

    llama_batch batch = llama_batch_get_one(toks.data(), n);
    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "decode failed\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float* logits = llama_get_logits_ith(ctx, n - 1);

    // write LGDN
    FILE* f = std::fopen(out, "wb");
    uint32_t magic = 0x4E44474C, nl = (uint32_t)g_hidden.size(),
             dim = (uint32_t)g_n_embd, vc = (uint32_t)n_vocab;
    std::fwrite(&magic, 4, 1, f); std::fwrite(&nl, 4, 1, f);
    std::fwrite(&dim, 4, 1, f); std::fwrite(&vc, 4, 1, f);
    for (uint32_t il = 0; il < nl; ++il)
        std::fwrite(g_hidden[il].data(), 4, dim, f);
    std::fwrite(logits, 4, vc, f);
    std::fclose(f);
    printf("wrote %s: %u layers, dim %u, vocab %u\n", out, nl, dim, vc);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
