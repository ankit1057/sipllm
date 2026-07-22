// test_gguf.cpp — Phase 5 (parse) + Phase 6 (GGUF-backed streaming inference).
#include "llm/gguf.h"
#include "llm/gguf_writer.h"
#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/transformer.h"
#include "tests/ref_forward.h"
#include "tests/test_util.h"

#include <string>

using namespace llm;

static std::string scratch(const char* n) { return llmtest::scratch_path(n); }

TEST(gguf_parse_metadata_and_directory) {
    GgufWriter w;
    w.str("general.architecture", "llama");
    w.u32("answer", 42);
    w.f32("pi", 3.14159f);
    w.str_array("tokenizer.ggml.tokens", {"a", "bb", "ccc"});
    w.i32_array("tokenizer.ggml.token_type", {1, 1, 1});
    std::vector<float> mat(4 * 8);
    for (size_t i = 0; i < mat.size(); ++i) mat[i] = (float)i;
    w.add_tensor("blk.0.attn_q.weight", DType::F32, {4, 8}, mat.data(), mat.size() * 4);
    std::string path = scratch("mini.gguf");
    w.write(path);

    GgufFile g(path);
    CHECK(g.version() == 3);
    CHECK(g.meta_str("general.architecture") == "llama");
    CHECK(g.meta_int("answer") == 42);
    APPROX(g.meta_float("pi"), 3.14159, 1e-5);
    const MetaValue* toks = g.meta("tokenizer.ggml.tokens");
    CHECK(toks && toks->kind == MetaValue::Kind::StrArr);
    CHECK(toks->sa.size() == 3 && toks->sa[1] == "bb");
    const MetaValue* tt = g.meta("tokenizer.ggml.token_type");
    CHECK(tt && tt->kind == MetaValue::Kind::IntArr && tt->ia.size() == 3);

    const TensorInfo* q = g.find("blk.0.attn_q.weight");
    CHECK(q != nullptr);
    // Our shape is [n_out, n_in] = [4, 8]; stored ne-order [8,4] then reversed.
    CHECK(q->shape.size() == 2 && q->shape[0] == 4 && q->shape[1] == 8);
    CHECK(q->offset % 32 == 0);                 // aligned data section
    CHECK(q->offset >= g.data_offset());
    std::vector<float> back(32);
    g.read_raw(*q, back.data());
    for (int i = 0; i < 32; ++i) APPROX(back[i], (double)i, 0);
}

TEST(gguf_config_discovery) {
    ToyGgufConfig c; c.n_layers = 3; c.dim = 64; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 128; c.vocab_size = 64; c.weight_type = DType::F32;
    std::string path = scratch("cfg.gguf");
    write_toy_gguf(path, c);
    GgufFile g(path);
    ModelConfig cfg = ModelConfig::from_source(g);
    CHECK(cfg.n_layers == 3);
    CHECK(cfg.n_heads == 4 && cfg.n_kv_heads == 2);
    CHECK(cfg.dim == 64 && cfg.head_dim == 16);
    CHECK(cfg.ffn_dim == 128);
    CHECK(cfg.vocab_size == 64);
    CHECK(!cfg.tie_embeddings);   // output.weight present
    APPROX(cfg.rope_theta, 10000.0, 1e-3);
}

static std::vector<float> gguf_stream_forward(const std::string& path,
                                              Residency res,
                                              const std::vector<int64_t>& tokens) {
    GgufFile g(path);
    ModelConfig cfg = ModelConfig::from_source(g);
    LayerLoader::Options opt; opt.residency = res; opt.async = true; opt.n_buffers = 2;
    LayerLoader loader(&g, cfg, opt);
    KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    ThreadPool pool(4);
    Transformer tf(&loader, &kv, &pool);
    const float* logits = nullptr;
    for (int64_t pos = 0; pos < (int64_t)tokens.size(); ++pos)
        logits = tf.forward(tokens[pos], pos);
    return std::vector<float>(logits, logits + cfg.vocab_size);
}

TEST(gguf_streaming_inference_fp32) {
    ToyGgufConfig c; c.n_layers = 3; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 2;
    c.ffn_dim = 64; c.vocab_size = 48; c.weight_type = DType::F32; c.seed = 11;
    std::string path = scratch("toy_fp32.gguf");
    write_toy_gguf(path, c);

    GgufFile g(path);
    ModelConfig cfg = ModelConfig::from_source(g);
    std::vector<int64_t> tokens = {5, 2, 9, 1, 7};
    auto ref = llmtest::ref_forward(g, cfg, tokens);
    auto got = gguf_stream_forward(path, Residency::Quantized, tokens);
    for (size_t i = 0; i < ref.size(); ++i) APPROX(got[i], ref[i], 1e-3);
}

TEST(gguf_streaming_inference_q8_0) {
    // Weights quantized to Q8_0 on disk; the loader keeps them quantized and
    // matmul_quant dequantizes per row. Reference dequantizes the SAME bytes,
    // so results must match tightly (no separate rounding path).
    ToyGgufConfig c; c.n_layers = 3; c.dim = 32; c.n_heads = 4; c.n_kv_heads = 4;
    c.ffn_dim = 64; c.vocab_size = 32; c.weight_type = DType::Q8_0; c.seed = 21;
    std::string path = scratch("toy_q8.gguf");
    write_toy_gguf(path, c);

    GgufFile g(path);
    ModelConfig cfg = ModelConfig::from_source(g);
    std::vector<int64_t> tokens = {3, 8, 1, 4};
    auto ref = llmtest::ref_forward(g, cfg, tokens);            // dequant-then-math
    auto got = gguf_stream_forward(path, Residency::Quantized, tokens);  // fused
    for (size_t i = 0; i < ref.size(); ++i) APPROX(got[i], ref[i], 1e-3);
}

int main() {
    printf("== test_gguf ==\n");
    return llmtest::run_all();
}
