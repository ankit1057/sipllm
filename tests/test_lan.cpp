// test_lan.cpp — Issue #70: splitting layers across machines must not change
// the arithmetic.
//
// The core test runs a real worker (serve_worker on 127.0.0.1, its own
// Runtime, its own thread) and a coordinator Transformer that owns layers
// [0, a) and hands [a, n) to the worker over TCP. Prefill and every greedy
// decode step must give logits BIT-IDENTICAL to one Transformer running every
// layer in one process (each decode step attends over the KV written on both
// machines, so identical logits across 8 steps also exercise the split KV).
// Also covered: frame bounds, token check, --nodes parsing,
// refusals (wrong token, mismatched settings, speculative decoding).
#include "llm/format.h"
#include "llm/kv_cache.h"
#include "llm/lan.h"
#include "llm/loader.h"
#include "llm/runtime.h"
#include "llm/sampler.h"
#include "llm/toy_model.h"
#include "llm/transformer.h"
#include "tests/test_util.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace llm;

static const std::string kToken = "0123456789abcdef-test-token";

TEST(header_bounds_and_token) {
    uint8_t h[lan::kHeaderSize];
    uint16_t type = 0; uint32_t len = 0; std::string err;
    lan::write_header(h, lan::MSG_RUN, 1234);
    CHECK(lan::parse_header(h, &type, &len, &err) && type == lan::MSG_RUN && len == 1234);
    lan::write_header(h, lan::MSG_RUN, lan::kMaxPayload + 1);          // over the cap
    CHECK(!lan::parse_header(h, &type, &len, &err));
    lan::write_header(h, 99, 8);                                          // unknown type
    CHECK(!lan::parse_header(h, &type, &len, &err));
    lan::write_header(h, lan::MSG_RUN, 8); h[0] ^= 0xFF;                 // bad magic
    CHECK(!lan::parse_header(h, &type, &len, &err));
    lan::write_header(h, lan::MSG_RUN, 8); h[4] = 9;                     // bad version
    CHECK(!lan::parse_header(h, &type, &len, &err));
    CHECK(lan::tokens_equal(kToken, kToken));
    CHECK(!lan::tokens_equal(kToken + "x", kToken));
    CHECK(!lan::tokens_equal(kToken.substr(0, 10), kToken));
    CHECK(!lan::tokens_equal("1123456789abcdef-test-token", kToken));
}

TEST(parse_nodes_validates_topology) {
    auto v = lan::parse_nodes("10.0.0.2:7071:6-11,10.0.0.1:7070:3-6");
    CHECK(v.size() == 2 && v[0].begin == 3 && v[0].end == 6 && v[1].host == "10.0.0.2" && v[1].port == 7071);
    bool threw = false;
    try { lan::parse_nodes("10.0.0.1:7070:3-6,10.0.0.2:7071:7-11"); } catch (const std::exception&) { threw = true; }
    CHECK_MSG(threw, "a gap between worker ranges must be rejected");
    threw = false;
    try { lan::parse_nodes("10.0.0.1:7070:3-6,10.0.0.2:7071:5-11"); } catch (const std::exception&) { threw = true; }
    CHECK_MSG(threw, "overlapping worker ranges must be rejected");
    threw = false;
    try { lan::parse_nodes("10.0.0.1:7070"); } catch (const std::exception&) { threw = true; }
    CHECK_MSG(threw, "an entry without a layer range must be rejected");
}

// A worker Runtime serving [a, n) on 127.0.0.1 in a background thread.
struct LoopbackWorker {
    std::unique_ptr<Runtime> rt;
    std::atomic<bool> stop{false}, ready{false};
    std::atomic<int> port{0};
    std::thread th;
    LoopbackWorker(const std::string& path, int a, int n_ctx, LayerLoader::Options opt) {
        opt.layer_begin = a; opt.layer_end = -1;
        rt = std::make_unique<Runtime>(open_model(path, false), opt, n_ctx, 2);
        lan::WorkerConfig wc; wc.bind_ip = "127.0.0.1"; wc.port = 0; wc.token = kToken;
        th = std::thread([this, wc] {
            try {
                lan::serve_worker(*rt, wc, &stop, [this](int p) { port = p; ready = true; });
            } catch (const std::exception& e) { fprintf(stderr, "worker thread: %s\n", e.what()); ready = true; }
        });
        while (!ready) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    ~LoopbackWorker() { stop = true; th.join(); }
};

struct Stack {   // a plain loader + KV + transformer for one layer range
    std::unique_ptr<WeightSource> src;
    ModelConfig cfg;
    std::unique_ptr<LayerLoader> loader;
    std::unique_ptr<KVCache> kv;
    std::unique_ptr<ThreadPool> pool;
    std::unique_ptr<Transformer> tf;
    Stack(const std::string& path, LayerLoader::Options opt, int begin, int end, int ctx) {
        src = open_model(path, false);
        cfg = ModelConfig::from_source(*src);
        opt.layer_begin = begin; opt.layer_end = end;
        const int e = end < 0 ? (int)cfg.n_layers : end;
        loader = std::make_unique<LayerLoader>(src.get(), cfg, opt);
        kv = std::make_unique<KVCache>(e - begin, cfg.kv_dim(), ctx, KVPrecision::FP32, begin);
        pool = std::make_unique<ThreadPool>(2);
        tf = std::make_unique<Transformer>(loader.get(), kv.get(), pool.get());
    }
};

static void split_bit_identity(const ToyConfig& tc, const char* name, int a) {
    const std::string path = llmtest::scratch_path(name);
    write_toy_model(path, tc);
    const int ctx = (int)tc.ctx_len;
    LayerLoader::Options opt; opt.residency = Residency::FP32; opt.async = true; opt.n_buffers = 2;

    LoopbackWorker w(path, a, ctx, opt);
    CHECK_MSG(w.port > 0, "worker failed to start");

    Stack ref(path, opt, 0, -1, ctx);       // every layer, one process
    Stack coord(path, opt, 0, a, ctx);      // layers [0, a) here, [a, n) over TCP
    lan::NodeSpec node{"127.0.0.1", w.port, a, (int)ref.cfg.n_layers};
    lan::RemoteExecutor remote(node, kToken, lan::fingerprint(*w.rt));
    coord.tf->set_remote_segments({RemoteSegment{a, ref.cfg.n_layers, &remote}});

    const std::vector<int64_t> prompt = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4,
                                         6, 2, 6, 4, 3, 3, 8, 3, 2, 7, 9, 5, 0, 2, 8};   // > 32: two chunks
    const int64_t V = ref.cfg.vocab_size;
    const float* lr = ref.tf->prefill(prompt.data(), (int64_t)prompt.size(), 0);
    std::vector<float> r(lr, lr + V);
    const float* lc = coord.tf->prefill(prompt.data(), (int64_t)prompt.size(), 0);
    CHECK_MSG(std::memcmp(r.data(), lc, V * sizeof(float)) == 0, std::string(name) + ": prefill logits differ");

    int64_t pos = (int64_t)prompt.size();
    for (int step = 0; step < 8; ++step, ++pos) {
        const int64_t next = Sampler::argmax(r.data(), V);
        lr = ref.tf->forward(next, pos);
        r.assign(lr, lr + V);
        lc = coord.tf->forward(next, pos);
        CHECK_MSG(std::memcmp(r.data(), lc, V * sizeof(float)) == 0,
                  std::string(name) + ": decode logits differ at step " + std::to_string(step));
    }
    CHECK_MSG(remote.calls() == 9, "one RUN for prefill + one per decode step");
}

TEST(split_is_bit_identical_mha) {
    ToyConfig tc; tc.n_layers = 4; tc.dim = 48; tc.n_heads = 4; tc.n_kv_heads = 4;
    tc.ffn_dim = 96; tc.vocab_size = 40; tc.seed = 77;
    split_bit_identity(tc, "toy_lan_mha.llmw", 2);
}

TEST(split_is_bit_identical_gqa_uneven) {
    ToyConfig tc; tc.n_layers = 5; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 48; tc.seed = 5;
    split_bit_identity(tc, "toy_lan_gqa.llmw", 1);   // coordinator 1 layer, worker 4
}

TEST(refuses_wrong_token_and_mismatch) {
    ToyConfig tc; tc.n_layers = 3; tc.dim = 32; tc.n_heads = 2; tc.n_kv_heads = 2; tc.seed = 3;
    const std::string path = llmtest::scratch_path("toy_lan_refuse.llmw");
    write_toy_model(path, tc);
    LayerLoader::Options opt; opt.residency = Residency::FP32;
    LoopbackWorker w(path, 1, (int)tc.ctx_len, opt);
    lan::NodeSpec node{"127.0.0.1", w.port, 1, 3};
    const lan::Fingerprint fp = lan::fingerprint(*w.rt);

    std::string msg;
    try { lan::RemoteExecutor bad(node, "wrong-token-but-long-enough", fp); } catch (const std::exception& e) { msg = e.what(); }
    CHECK_MSG(msg.find("authentication failed") != std::string::npos, "wrong token must be refused, got: " + msg);

    lan::Fingerprint other = fp; other.max_ctx = fp.max_ctx / 2;   // e.g. a different --ctx
    msg.clear();
    try { lan::RemoteExecutor bad(node, kToken, other); } catch (const std::exception& e) { msg = e.what(); }
    CHECK_MSG(msg.find("configuration mismatch") != std::string::npos && msg.find("context") != std::string::npos,
              "mismatched context must be refused, got: " + msg);

    other = fp; other.tensor_hash ^= 1;                              // a different model file
    msg.clear();
    try { lan::RemoteExecutor bad(node, kToken, other); } catch (const std::exception& e) { msg = e.what(); }
    CHECK_MSG(msg.find("tensor table") != std::string::npos, "a different model must be refused, got: " + msg);

    // The worker is still serving after refusals: a good connection succeeds.
    lan::RemoteExecutor good(node, kToken, fp);
    CHECK(good.calls() == 0);
}

TEST(runtime_coordinator_matches_and_refuses_spec) {
    ToyConfig tc; tc.n_layers = 4; tc.dim = 32; tc.n_heads = 4; tc.n_kv_heads = 2;
    tc.ffn_dim = 64; tc.vocab_size = 300; tc.seed = 11;   // >= 256: byte-level fallback ids
    const std::string path = llmtest::scratch_path("toy_lan_runtime.llmw");
    write_toy_model(path, tc);
    const int ctx = (int)tc.ctx_len;
    LayerLoader::Options opt; opt.residency = Residency::FP32;
    LoopbackWorker w(path, 2, ctx, opt);

    SamplerConfig greedy; greedy.temperature = 0.f;
    std::vector<int64_t> ids_ref, ids_split;
    {
        Runtime rt(open_model(path, false), opt, ctx, 2);
        rt.generate("hello world", 12, greedy, [&](const std::string&, int64_t id) { ids_ref.push_back(id); return true; }, nullptr);
    }
    {
        LayerLoader::Options copt = opt; copt.layer_end = 2;
        Runtime rt(open_model(path, false), copt, ctx, 2);
        lan::RemoteExecutor remote(lan::NodeSpec{"127.0.0.1", w.port, 2, 4}, kToken, lan::fingerprint(rt));
        rt.set_remote_segments({RemoteSegment{2, 4, &remote}});
        rt.generate("hello world", 12, greedy, [&](const std::string&, int64_t id) { ids_split.push_back(id); return true; }, nullptr);
        CHECK_MSG(!rt.save_session(llmtest::scratch_path("lan.session")), "sessions must be refused with remote layers");
        rt.enable_speculative_decoding(3);
        bool threw = false;
        try { rt.generate("again", 4, greedy, nullptr, nullptr); } catch (const std::exception&) { threw = true; }
        CHECK_MSG(threw, "--spec must be refused with remote layers");
    }
    CHECK_MSG(!ids_ref.empty() && ids_ref == ids_split, "greedy tokens must match single-process generation");
}

int main() {
    printf("== test_lan ==\n");
    return llmtest::run_all();
}
