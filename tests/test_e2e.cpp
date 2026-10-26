// test_e2e.cpp — Phase 4: prove the architecture end to end.
//
// Build a tiny random Llama (.llmw), then check that a forward pass driven
// entirely through the streaming LayerLoader matches the independent
// all-resident reference. Also confirm the async double-buffer pipeline and the
// synchronous single-buffer loader agree, and that the prefetcher scores hits.
#include "llm/format.h"
#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/toy_model.h"
#include "llm/transformer.h"
#include "tests/ref_forward.h"
#include "tests/test_util.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace llm;

static std::string scratch(const char* n) { return llmtest::scratch_path(n); }

// Run the streaming Transformer over a token sequence; return last logits.
static std::vector<float> stream_forward(const std::string& path,
                                         LayerLoader::Options opt,
                                         const std::vector<int64_t>& tokens,
                                         LayerLoader::Stats* out = nullptr) {
    ModelFile f(path, opt.use_mmap);
    ModelConfig cfg = ModelConfig::from_source(f);
    LayerLoader loader(&f, cfg, opt);
    KVCache kv(cfg.n_layers, cfg.kv_dim(), cfg.ctx_len);
    ThreadPool pool(4);
    Transformer tf(&loader, &kv, &pool);
    const float* logits = nullptr;
    for (int64_t pos = 0; pos < (int64_t)tokens.size(); ++pos)
        logits = tf.forward(tokens[pos], pos);
    std::vector<float> res(logits, logits + cfg.vocab_size);
    if (out) {
        out->prefetch_hits = loader.stats().prefetch_hits.load();
        out->prefetch_misses = loader.stats().prefetch_misses.load();
        out->bytes_read = loader.stats().bytes_read.load();
    }
    return res;
}

TEST(e2e_streaming_matches_reference) {
    ToyConfig tc; tc.n_layers = 3; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 48; tc.seed = 7;
    std::string path = scratch("toy_e2e.llmw");
    write_toy_model(path, tc);

    ModelFile f(path);
    ModelConfig cfg = ModelConfig::from_source(f);
    std::vector<int64_t> tokens = {3, 1, 4, 1, 5, 9, 2, 6};
    auto ref = llmtest::ref_forward(f, cfg, tokens);

    LayerLoader::Options opt;
    opt.residency = Residency::FP32; opt.async = true; opt.n_buffers = 2;
    auto got = stream_forward(path, opt, tokens);

    CHECK(ref.size() == got.size());
    for (size_t i = 0; i < ref.size(); ++i) APPROX(got[i], ref[i], 1e-3);
}

TEST(e2e_async_equals_sync) {
    ToyConfig tc; tc.n_layers = 4; tc.dim = 48; tc.n_heads = 4; tc.n_kv_heads = 4;
    tc.ffn_dim = 96; tc.vocab_size = 40; tc.seed = 99;
    std::string path = scratch("toy_sync.llmw");
    write_toy_model(path, tc);
    std::vector<int64_t> tokens = {1, 2, 3, 4, 5};

    LayerLoader::Options a; a.async = true;  a.n_buffers = 2; a.residency = Residency::Quantized;
    LayerLoader::Options b; b.async = false; b.n_buffers = 1; b.residency = Residency::Quantized;
    LayerLoader::Stats sa;
    auto ga = stream_forward(path, a, tokens, &sa);
    auto gb = stream_forward(path, b, tokens);
    for (size_t i = 0; i < ga.size(); ++i) APPROX(ga[i], gb[i], 1e-4);
    // Deterministic accounting: every loadLayer resolves to exactly one hit or
    // miss, so over 5 tokens x 4 layers the pipeline recorded 20 lookups. (How
    // many were *hits* is a compute-vs-IO race and is asserted separately in a
    // way that does not depend on scheduling — see e2e_prefetch_scores_hits.)
    CHECK(sa.prefetch_hits + sa.prefetch_misses == (uint64_t)tokens.size() * (uint64_t)tc.n_layers);
}

TEST(e2e_prefetch_scores_hits) {
    // Deterministic proof that the async double-buffer actually prefetches the
    // next block: load layer 0 (which enqueues layer 1), give the worker ample
    // time to finish the few-KB load, then request layer 1 — it must already be
    // Ready, i.e. a prefetch hit. The sleep dwarfs the load, so this never races.
    ToyConfig tc; tc.n_layers = 4; tc.dim = 48; tc.n_heads = 4; tc.n_kv_heads = 4;
    tc.ffn_dim = 96; tc.vocab_size = 40; tc.seed = 99;
    std::string path = scratch("toy_prefetch.llmw");
    write_toy_model(path, tc);

    ModelFile f(path);
    ModelConfig cfg = ModelConfig::from_source(f);
    LayerLoader::Options opt; opt.async = true; opt.n_buffers = 2;
    opt.residency = Residency::Quantized;
    LayerLoader loader(&f, cfg, opt);

    loader.loadLayer(0);                                   // enqueues prefetch of layer 1
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const uint64_t hits_before = loader.stats().prefetch_hits.load();
    loader.loadLayer(1);                                   // already resident -> hit
    loader.unloadLayer();
    CHECK_MSG(loader.stats().prefetch_hits.load() == hits_before + 1,
              "layer 1 must be a prefetch hit once the worker has run");
}

TEST(e2e_tied_embeddings) {
    ToyConfig tc; tc.n_layers = 2; tc.dim = 32; tc.n_heads = 2; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 32; tc.tied = true; tc.seed = 5;
    std::string path = scratch("toy_tied.llmw");
    write_toy_model(path, tc);

    ModelFile f(path);
    ModelConfig cfg = ModelConfig::from_source(f);
    CHECK(cfg.tie_embeddings);
    std::vector<int64_t> tokens = {7, 3, 11};
    auto ref = llmtest::ref_forward(f, cfg, tokens);
    LayerLoader::Options opt; opt.residency = Residency::FP32;
    auto got = stream_forward(path, opt, tokens);
    for (size_t i = 0; i < ref.size(); ++i) APPROX(got[i], ref[i], 1e-3);
}

TEST(e2e_streamed_lm_head_matches_resident) {
    // Non-tied toy: streaming the LM head off disk (row-blocked, project_output)
    // must produce identical logits to holding it resident. Guards --stream-lm-head.
    ToyConfig tc; tc.n_layers = 3; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 48; tc.tied = false; tc.seed = 13;
    std::string path = scratch("toy_lmhead.llmw");
    write_toy_model(path, tc);
    std::vector<int64_t> tokens = {5, 2, 9, 1, 7, 3};

    LayerLoader::Options res; res.residency = Residency::Quantized; res.stream_lm_head = false;
    LayerLoader::Options str; str.residency = Residency::Quantized; str.stream_lm_head = true;
    auto a = stream_forward(path, res, tokens);
    auto b = stream_forward(path, str, tokens);
    CHECK(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) APPROX(b[i], a[i], 1e-4);
}

// A WeightSource that can hold reads of chosen layers at the start of read_raw,
// before dst is written, so a test can freeze the prefetch worker mid-fill and
// queue jobs behind it in a known order.
class GatedSource : public WeightSource {
public:
    explicit GatedSource(const WeightSource& in) : in_(in) {}
    const std::vector<TensorInfo>& tensors() const override { return in_.tensors(); }
    const TensorInfo* find(const std::string& n) const override { return in_.find(n); }
    void read_raw_at(uint64_t off, void* dst, uint64_t n) const override { in_.read_raw_at(off, dst, n); }
    uint64_t file_size() const override { return in_.file_size(); }
    bool has_meta(const std::string& k) const override { return in_.has_meta(k); }
    const MetaValue* meta(const std::string& k) const override { return in_.meta(k); }

    void read_raw(const TensorInfo& t, void* dst) const override {
        const int l = layer_of(t.name);
        if (l >= 0) {
            std::unique_lock<std::mutex> lk(mu_);
            started_[l] += 1;
            cv_.notify_all();
            cv_.wait(lk, [&] { return !closed_[l]; });
        }
        in_.read_raw(t, dst);
    }
    void close(int l) { std::lock_guard<std::mutex> lk(mu_); closed_[l] = true; }
    void open(int l) { { std::lock_guard<std::mutex> lk(mu_); closed_[l] = false; } cv_.notify_all(); }
    void open_all() { { std::lock_guard<std::mutex> lk(mu_); for (bool& c : closed_) c = false; } cv_.notify_all(); }
    // True if a read of layer l starts within `ms` milliseconds.
    bool wait_started(int l, int ms) const {
        std::unique_lock<std::mutex> lk(mu_);
        return cv_.wait_for(lk, std::chrono::milliseconds(ms), [&] { return started_[l] > 0; });
    }

private:
    static constexpr int kMax = 8;
    static int layer_of(const std::string& name) {
        int l = -1;
        return std::sscanf(name.c_str(), "blk.%d.", &l) == 1 && l >= 0 && l < kMax ? l : -1;
    }
    const WeightSource& in_;
    mutable std::mutex mu_;
    mutable std::condition_variable cv_;
    mutable int started_[kMax] = {};
    bool closed_[kMax] = {};
};

struct ScriptedPolicy : LayerLoader::Options::PrefetchPolicy {
    std::atomic<int> next_layer{-1};
    int next(int, int, const std::vector<uint8_t>&) const override { return next_layer.load(); }
};

TEST(e2e_stale_prefetch_never_clobbers_requested_layer) {
    // The worker is held mid-fill of a prefetch (layer 1 -> slot 1). A second
    // prefetch (layer 3) then queues for the same slot, and an out-of-order
    // request for layer 2 jumps the queue into that slot. The older layer-3 job
    // must not run after it: it would overwrite the slot the caller is reading,
    // or leave the caller waiting forever (the hang seen on x86 CI).
    ToyConfig tc; tc.n_layers = 4; tc.dim = 48; tc.n_heads = 4; tc.n_kv_heads = 4;
    tc.ffn_dim = 96; tc.vocab_size = 40; tc.seed = 99;
    std::string path = scratch("toy_stale.llmw");
    write_toy_model(path, tc);

    // Heap-held so a deadlocked run can leak them instead of hanging in teardown.
    auto file = std::make_unique<ModelFile>(path);
    auto src = std::make_unique<GatedSource>(*file);
    ModelConfig cfg = ModelConfig::from_source(*src);
    auto policy = std::make_shared<ScriptedPolicy>();
    LayerLoader::Options opt; opt.async = true; opt.n_buffers = 2;
    opt.residency = Residency::Quantized; opt.prefetch_policy = policy;
    auto loader = std::make_unique<LayerLoader>(src.get(), cfg, opt);

    src->close(1); src->close(3);
    policy->next_layer = 1;
    loader->loadLayer(0);                     // slot 0; prefetch layer 1 -> slot 1
    if (!src->wait_started(1, 5000)) {
        src->open_all();
        CHECK_MSG(false, "the worker never started the layer-1 prefetch");
    }
    policy->next_layer = 3;
    loader->loadLayer(0);                     // hit; prefetch layer 3 -> slot 1, queued
    policy->next_layer = -1;                  // no prefetch after this point

    auto done = std::make_shared<std::atomic<bool>>(false);
    LayerLoader* raw = loader.get();
    std::thread t([raw, done] { raw->loadLayer(2); *done = true; });   // miss -> slot 1, queue front
    for (int i = 0; i < 5000 && loader->stats().prefetch_misses.load() < 2; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    src->open(1);                             // let the layer-1 fill finish
    for (int i = 0; i < 2000 && !done->load(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (!done->load()) {
        src->open_all();
        t.detach();                           // stuck inside the loader: leak, don't destroy
        (void)loader.release(); (void)src.release(); (void)file.release();
        CHECK_MSG(false, "loadLayer(2) never returned: a stale job took its slot (deadlock)");
    }
    t.join();

    const TensorInfo* q2 = file->find(names::blk(2, "attn_q.weight"));
    std::vector<uint8_t> want(q2->nbytes);
    file->read_raw(*q2, want.data());
    WeightRef w = loader->getWeight(Role::AttnQ);
    const bool right_bytes = w.valid() && std::memcmp(w.data, want.data(), want.size()) == 0;
    const bool stale_ran = src->wait_started(3, 200);
    src->open_all();
    loader.reset();
    CHECK_MSG(right_bytes, "loadLayer(2) returned with another layer's weights");
    CHECK_MSG(!stale_ran, "a stale prefetch (layer 3) ran after the request for layer 2");
}

int main() {
    printf("== test_e2e ==\n");
    return llmtest::run_all();
}
