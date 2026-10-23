// test_batch_decode.cpp — Transformer::forward_multi (batched multi-sequence
// decode) must give every sequence exactly the logits and KV it would get when
// decoded alone with forward(): same kernels, same reduction order, its own
// cache. Toy models only use CPU weights, so the comparison is bit-for-bit.
#include "llm/format.h"
#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/threadpool.h"
#include "llm/toy_model.h"
#include "llm/transformer.h"
#include "tests/test_util.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

using namespace llm;
static std::string scratch(const char* n) { return llmtest::scratch_path(n); }

namespace {

struct Seq {
    std::vector<int64_t> prompt;   // prefilled first
    std::vector<int64_t> cont;     // then fed one token per decode step
};

// Reference: each sequence alone — prefill its prompt, then forward() each
// continuation token. Returns per-step logits [step][vocab] per sequence.
std::vector<std::vector<std::vector<float>>> run_alone(const std::string& path, LayerLoader::Options opt,
                                                       const std::vector<Seq>& seqs, int threads,
                                                       std::vector<std::vector<float>>* kv_out) {
    ModelFile f(path, opt.use_mmap);
    ModelConfig cfg = ModelConfig::from_source(f);
    LayerLoader loader(&f, cfg, opt);
    ThreadPool pool(threads);
    std::vector<std::vector<std::vector<float>>> out(seqs.size());
    for (size_t s = 0; s < seqs.size(); ++s) {
        KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
        Transformer tf(&loader, &kv, &pool);
        const float* lg = tf.prefill(seqs[s].prompt.data(), (int64_t)seqs[s].prompt.size(), 0);
        out[s].push_back(std::vector<float>(lg, lg + cfg.vocab_size));
        int64_t pos = (int64_t)seqs[s].prompt.size();
        for (int64_t tok : seqs[s].cont) {
            lg = tf.forward(tok, pos++);
            out[s].push_back(std::vector<float>(lg, lg + cfg.vocab_size));
        }
        if (kv_out) {
            std::vector<float> flat;
            for (int64_t l = 0; l < cfg.n_layers; ++l)
                for (int64_t p = 0; p < pos; ++p) {
                    flat.insert(flat.end(), kv.k(l, p), kv.k(l, p) + cfg.kv_dim());
                    flat.insert(flat.end(), kv.v(l, p), kv.v(l, p) + cfg.kv_dim());
                }
            kv_out->push_back(std::move(flat));
        }
    }
    return out;
}

// Batched: prefill every sequence into its own cache (set_kv), then decode all
// of them in lockstep with forward_multi. Sequences of different length drop
// out of the batch when their continuation is exhausted.
std::vector<std::vector<std::vector<float>>> run_batched(const std::string& path, LayerLoader::Options opt,
                                                         const std::vector<Seq>& seqs, int threads,
                                                         std::vector<std::vector<float>>* kv_out,
                                                         ThreadPool::SchedulePolicy policy =
                                                             ThreadPool::SchedulePolicy::Proportional4) {
    ModelFile f(path, opt.use_mmap);
    ModelConfig cfg = ModelConfig::from_source(f);
    LayerLoader loader(&f, cfg, opt);
    ThreadPool pool(threads);
    pool.set_policy(policy);
    std::vector<KVCache> kvs;
    for (size_t s = 0; s < seqs.size(); ++s) kvs.emplace_back(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    KVCache scratch(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    Transformer tf(&loader, &scratch, &pool);
    std::vector<std::vector<std::vector<float>>> out(seqs.size());
    std::vector<int64_t> pos(seqs.size());
    for (size_t s = 0; s < seqs.size(); ++s) {
        tf.set_kv(&kvs[s]);
        const float* lg = tf.prefill(seqs[s].prompt.data(), (int64_t)seqs[s].prompt.size(), 0);
        out[s].push_back(std::vector<float>(lg, lg + cfg.vocab_size));
        pos[s] = (int64_t)seqs[s].prompt.size();
    }
    for (size_t step = 0;; ++step) {
        std::vector<int64_t> toks, ps, idx;
        std::vector<KVCache*> ks;
        for (size_t s = 0; s < seqs.size(); ++s)
            if (step < seqs[s].cont.size()) {
                toks.push_back(seqs[s].cont[step]);
                ps.push_back(pos[s]);
                ks.push_back(&kvs[s]);
                idx.push_back((int64_t)s);
            }
        if (toks.empty()) break;
        const float* lg = tf.forward_multi(toks.data(), ps.data(), ks.data(), (int64_t)toks.size());
        for (size_t r = 0; r < idx.size(); ++r) {
            const float* row = lg + r * cfg.vocab_size;
            out[(size_t)idx[r]].push_back(std::vector<float>(row, row + cfg.vocab_size));
            ++pos[(size_t)idx[r]];
        }
    }
    if (kv_out)
        for (size_t s = 0; s < seqs.size(); ++s) {
            std::vector<float> flat;
            for (int64_t l = 0; l < cfg.n_layers; ++l)
                for (int64_t p = 0; p < pos[s]; ++p) {
                    flat.insert(flat.end(), kvs[s].k(l, p), kvs[s].k(l, p) + cfg.kv_dim());
                    flat.insert(flat.end(), kvs[s].v(l, p), kvs[s].v(l, p) + cfg.kv_dim());
                }
            kv_out->push_back(std::move(flat));
        }
    return out;
}

void expect_same(const std::vector<std::vector<std::vector<float>>>& a,
                 const std::vector<std::vector<std::vector<float>>>& b, const std::string& what) {
    CHECK_MSG(a.size() == b.size(), what + ": sequence count");
    for (size_t s = 0; s < a.size(); ++s) {
        CHECK_MSG(a[s].size() == b[s].size(), what + ": step count, seq " + std::to_string(s));
        for (size_t t = 0; t < a[s].size() && t < b[s].size(); ++t)
            CHECK_MSG(a[s][t].size() == b[s][t].size() &&
                          std::memcmp(a[s][t].data(), b[s][t].data(), a[s][t].size() * sizeof(float)) == 0,
                      what + ": logits differ, seq " + std::to_string(s) + " step " + std::to_string(t));
    }
}

std::vector<Seq> three_sequences(int64_t vocab) {
    auto mk = [&](int n, int mul, int add) {
        std::vector<int64_t> v;
        for (int i = 0; i < n; ++i) v.push_back((i * mul + add) % vocab);
        return v;
    };
    // Different prompt lengths (so rows sit at different positions) and
    // different continuation lengths (so the batch shrinks 3 -> 2 -> 1).
    return {{mk(9, 7, 3), mk(12, 5, 1)}, {mk(23, 11, 2), mk(7, 3, 4)}, {mk(4, 13, 5), mk(15, 17, 6)}};
}

}  // namespace

TEST(batch_decode_equals_alone_gqa_fp32) {
    ToyConfig tc; tc.n_layers = 3; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 48; tc.seed = 21;
    std::string path = scratch("toy_batch_gqa.llmw");
    write_toy_model(path, tc);
    LayerLoader::Options opt; opt.residency = Residency::FP32; opt.async = false; opt.n_buffers = 1;
    const auto seqs = three_sequences(tc.vocab_size);
    for (int threads : {1, 4, 12}) {
        std::vector<std::vector<float>> ka, kb;
        auto a = run_alone(path, opt, seqs, threads, &ka);
        auto b = run_batched(path, opt, seqs, threads, &kb);
        const std::string what = "gqa fp32 t=" + std::to_string(threads);
        expect_same(a, b, what);
        for (size_t s = 0; s < ka.size(); ++s)
            CHECK_MSG(ka[s] == kb[s], what + ": KV differs, seq " + std::to_string(s));
    }
}

TEST(batch_decode_equals_alone_mqa_quant) {
    ToyConfig tc; tc.n_layers = 2; tc.dim = 32; tc.n_heads = 8; tc.n_kv_heads = 1;
    tc.ffn_dim = 64; tc.vocab_size = 40; tc.seed = 33;
    std::string path = scratch("toy_batch_mqa.llmw");
    write_toy_model(path, tc);
    LayerLoader::Options opt; opt.residency = Residency::Quantized; opt.async = false; opt.n_buffers = 1;
    const auto seqs = three_sequences(tc.vocab_size);
    for (int threads : {1, 6})
        expect_same(run_alone(path, opt, seqs, threads, nullptr), run_batched(path, opt, seqs, threads, nullptr),
                    "mqa quant t=" + std::to_string(threads));
}

TEST(batch_decode_single_row_equals_forward) {
    ToyConfig tc; tc.n_layers = 2; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 48; tc.tied = true; tc.seed = 5;
    std::string path = scratch("toy_batch_one.llmw");
    write_toy_model(path, tc);
    LayerLoader::Options opt; opt.residency = Residency::FP32;
    std::vector<Seq> one = {{{7, 3, 11, 2}, {9, 1, 4, 4, 8}}};
    expect_same(run_alone(path, opt, one, 4, nullptr), run_batched(path, opt, one, 4, nullptr), "single row");
}

TEST(batch_decode_rejects_shared_cache) {
    ToyConfig tc; tc.n_layers = 1; tc.dim = 32; tc.n_heads = 2; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 32; tc.seed = 2;
    std::string path = scratch("toy_batch_shared.llmw");
    write_toy_model(path, tc);
    LayerLoader::Options opt; opt.residency = Residency::FP32;
    ModelFile f(path, opt.use_mmap);
    ModelConfig cfg = ModelConfig::from_source(f);
    LayerLoader loader(&f, cfg, opt);
    KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    Transformer tf(&loader, &kv, nullptr);
    int64_t toks[2] = {1, 2}, ps[2] = {0, 0};
    KVCache* ks[2] = {&kv, &kv};
    bool threw = false;
    try { tf.forward_multi(toks, ps, ks, 2); } catch (const std::exception&) { threw = true; }
    CHECK_MSG(threw, "two rows sharing one cache must be rejected");
}

TEST(pool_chunk_override_splits_small_ranges) {
    // The Adaptive policy's minimum chunk is 16, so a 16-item parallel_for runs
    // as one chunk on the calling thread and the workers get nothing (this left
    // decode attention single-threaded). A per-call chunk of 1 must hand out
    // single items under every policy — including Static, whose
    // one-chunk-per-thread split would otherwise drop items past the n-th.
    using P = ThreadPool::SchedulePolicy;
    for (P pol : {P::Adaptive, P::Static, P::Proportional4, P::Fixed32}) {
        ThreadPool pool(4);
        pool.set_policy(pol);
        const std::string what = "policy " + std::to_string((int)pol);
        std::vector<int> hits(16, 0);
        std::vector<int64_t> widths;
        std::mutex m;
        auto fn = [&](int, int64_t b, int64_t e) {
            for (int64_t i = b; i < e; ++i) ++hits[(size_t)i];   // distinct slots, no race
            std::lock_guard<std::mutex> lk(m);
            widths.push_back(e - b);
        };
        pool.parallel_for(16, fn, 1);
        for (int i = 0; i < 16; ++i) CHECK_MSG(hits[(size_t)i] == 1, what + ": item " + std::to_string(i));
        CHECK_MSG(widths.size() == 16, what + ": expected 16 single-item calls");
        for (int64_t w : widths) CHECK_MSG(w == 1, what + ": chunk wider than 1");
    }
    // Without an override the policy decides: Adaptive keeps 16 items together.
    ThreadPool pool(4);
    pool.set_policy(P::Adaptive);
    std::vector<int64_t> widths;
    std::mutex m;
    pool.parallel_for(16, [&](int, int64_t b, int64_t e) {
        std::lock_guard<std::mutex> lk(m);
        widths.push_back(e - b);
    });
    CHECK_MSG(widths.size() == 1 && widths[0] == 16, "adaptive default: one 16-item chunk");
}

TEST(batch_decode_same_under_every_schedule) {
    // Which thread runs which attention item / matmul row must not change a
    // single bit. Long prompts so decode attention splits into several time
    // chunks (512 positions each) and the merge path runs too.
    ToyConfig tc; tc.n_layers = 2; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 1;
    tc.ffn_dim = 64; tc.vocab_size = 48; tc.ctx_len = 1400; tc.seed = 77;
    std::string path = scratch("toy_batch_sched.llmw");
    write_toy_model(path, tc);
    LayerLoader::Options opt; opt.residency = Residency::FP32; opt.async = false; opt.n_buffers = 1;
    auto mk = [&](int n, int mul, int add) {
        std::vector<int64_t> v;
        for (int i = 0; i < n; ++i) v.push_back((i * mul + add) % tc.vocab_size);
        return v;
    };
    const std::vector<Seq> seqs = {{mk(700, 7, 3), mk(6, 5, 1)}, {mk(1030, 11, 2), mk(4, 3, 4)}, {mk(560, 13, 5), mk(5, 17, 6)}};
    using P = ThreadPool::SchedulePolicy;
    const auto ref = run_alone(path, opt, seqs, 6, nullptr);
    for (P pol : {P::Adaptive, P::Static, P::Proportional4, P::Fixed8})
        expect_same(ref, run_batched(path, opt, seqs, 6, nullptr, pol), "schedule " + std::to_string((int)pol));
}

int main() {
    printf("== test_batch_decode ==\n");
    return llmtest::run_all();
}
