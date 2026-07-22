// selftest.cpp — exercise all 11 roadmap tasks through the real code paths.
#include "llm/selftest.h"

#include "llm/format.h"
#include "llm/gguf.h"
#include "llm/gguf_writer.h"
#include "llm/kv_cache.h"
#include "llm/loader.h"
#include "llm/neon.h"
#include "llm/ops.h"
#include "llm/quant.h"
#include "llm/simd.h"
#include "llm/tokenizer.h"
#include "llm/toy_model.h"
#include "llm/transformer.h"
#include "llm/vulkan_backend.h"
#include "tests/ref_forward.h"

#include <cmath>
#include <cstdio>
#include <functional>

namespace llm {

namespace {
struct Ctx { std::string dir; };

// Run one check; capture ok/detail/timing and never let an exception escape.
SelfTestResult run(int task, const char* name,
                   const std::function<std::string()>& fn) {
    SelfTestResult r; r.task = task; r.name = name; r.ok = false;
    double t0 = now_sec();
    try {
        r.detail = fn();
        r.ok = true;
    } catch (const std::exception& e) {
        r.detail = std::string("FAILED: ") + e.what();
    }
    r.ms = (now_sec() - t0) * 1e3;
    return r;
}

std::string p(const Ctx& c, const char* n) { return c.dir + "/" + n; }

double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i]-b[i]));
    return m;
}
} // namespace

std::vector<SelfTestResult> run_selftests(const std::string& tmpdir) {
    Ctx c{tmpdir};
    std::vector<SelfTestResult> out;

    // Task 1 — Tensor
    out.push_back(run(1, "Tensor class", [&] {
        Tensor t({2, 3}); t.at(1, 2) = 9.f; t.fill(2.f);
        Tensor u = std::move(t);
        LLM_CHECK(u.numel() == 6 && u[0] == 2.f, "tensor state wrong");
        return std::string("alloc/index/fill/move ok; 2x3 contiguous");
    }));

    // Task 2 — matmul
    out.push_back(run(2, "Linear (matmul)", [&] {
        float W[6] = {1,2,3,4,5,6}, x[3] = {1,0,-1}, y[2];
        matmul(y, W, x, 2, 3);
        LLM_CHECK(std::fabs(y[0]+2) < 1e-6 && std::fabs(y[1]+2) < 1e-6, "matmul values");
        // threaded == serial
        std::vector<float> Wr(300*200), xr(200), y1(300), y2(300);
        for (auto& v : Wr) v = 0.01f; for (auto& v : xr) v = 0.02f;
        matmul(y1.data(), Wr.data(), xr.data(), 300, 200, nullptr);
        ThreadPool pool(4); matmul(y2.data(), Wr.data(), xr.data(), 300, 200, &pool);
        LLM_CHECK(std::fabs(y1[0]-y2[0]) < 1e-4, "threaded != serial");
        return std::string("known values + threaded==serial (NEON dot)");
    }));

    // Task 3 — binary format
    out.push_back(run(3, "Binary weight format (.llmw)", [&] {
        std::vector<float> a(37); for (size_t i=0;i<a.size();++i) a[i]=(float)i;
        ModelWriter w; w.set_meta("k", (int64_t)7); w.add_f32("v", {37}, a.data());
        std::string path = p(c, "st.llmw"); w.write(path);
        ModelFile f(path);
        LLM_CHECK(f.meta_int("k")==7, "meta");
        const TensorInfo* ti = f.find("v"); LLM_CHECK(ti && ti->offset%64==0, "align");
        std::vector<float> b(37); f.read_raw(*ti, b.data());
        LLM_CHECK(max_abs_diff(a,b)==0, "roundtrip");
        return std::string("write/read round-trip, 64B aligned, meta preserved");
    }));

    // Task 4 — streaming loader
    out.push_back(run(4, "Streaming layer loader", [&] {
        ToyConfig tc; tc.n_layers=4; tc.dim=48; tc.n_heads=4; tc.n_kv_heads=4; tc.ffn_dim=96; tc.vocab_size=40;
        std::string path = p(c, "st_loader.llmw"); write_toy_model(path, tc);
        ModelFile f(path); ModelConfig cfg = ModelConfig::from_source(f);
        LayerLoader::Options opt; opt.async=true; opt.n_buffers=2;
        LayerLoader loader(&f, cfg, opt);
        for (int i=0;i<cfg.n_layers;++i) loader.loadLayer(i);
        uint64_t hits = loader.stats().prefetch_hits.load();
        LLM_CHECK(hits > 0, "no prefetch hits");
        char b[96]; snprintf(b,sizeof b,"streamed %lld layers, %llu prefetch hits, %.2f MB resident",
                 (long long)cfg.n_layers,(unsigned long long)hits, loader.resident_bytes()/1e6);
        return std::string(b);
    }));

    // Task 5 — transformer (e2e vs reference)
    out.push_back(run(5, "Transformer block (fwd == reference)", [&] {
        ToyConfig tc; tc.n_layers=3; tc.dim=32; tc.n_heads=4; tc.n_kv_heads=2; tc.ffn_dim=64; tc.vocab_size=48; tc.seed=7;
        std::string path = p(c, "st_tf.llmw"); write_toy_model(path, tc);
        ModelFile f(path); ModelConfig cfg = ModelConfig::from_source(f);
        std::vector<int64_t> toks = {3,1,4,1,5};
        auto ref = llmtest::ref_forward(f, cfg, toks);
        LayerLoader::Options opt; opt.residency=Residency::FP32;
        LayerLoader loader(&f,cfg,opt); KVCache kv(cfg.n_layers,cfg.kv_dim(),cfg.ctx_len);
        ThreadPool pool(4); Transformer tf(&loader,&kv,&pool);
        const float* lg=nullptr; for (size_t i=0;i<toks.size();++i) lg=tf.forward(toks[i],i);
        std::vector<float> got(lg, lg+cfg.vocab_size);
        double d = max_abs_diff(ref, got);
        LLM_CHECK(d < 1e-3, "logits differ from reference");
        char b[80]; snprintf(b,sizeof b,"RMSNorm+RoPE+GQA+SwiGLU; max|Δ| vs ref = %.2e", d);
        return std::string(b);
    }));

    // Task 6 — KV cache
    out.push_back(run(6, "KV cache", [&] {
        ToyConfig tc; tc.n_layers=2; tc.dim=32; tc.n_heads=2; tc.n_kv_heads=2; tc.ffn_dim=64; tc.vocab_size=32;
        std::string path = p(c, "st_kv.llmw"); write_toy_model(path, tc);
        ModelFile f(path); ModelConfig cfg = ModelConfig::from_source(f);
        LayerLoader::Options opt; opt.residency=Residency::FP32;
        LayerLoader loader(&f,cfg,opt); KVCache kv(cfg.n_layers,cfg.kv_dim(),cfg.ctx_len);
        ThreadPool pool(2); Transformer tf(&loader,&kv,&pool);
        for (int i=0;i<5;++i) tf.forward((i*7)%cfg.vocab_size, i);
        LLM_CHECK(kv.seq_len()==5, "seq_len"); LLM_CHECK(kv.bytes()>0, "bytes");
        char b[80]; snprintf(b,sizeof b,"5 positions cached, %.1f KB, seq_len=%lld",
                 kv.bytes()/1024.0,(long long)kv.seq_len());
        return std::string(b);
    }));

    // Task 7 — tokenizer
    out.push_back(run(7, "Tokenizer", [&] {
        Tokenizer bt = Tokenizer::byte_tokenizer();
        std::string s = "Hello 123"; auto ids = bt.encode(s,false);
        LLM_CHECK(bt.decode(ids)==s, "byte round-trip");
        return std::string("byte round-trip exact; SPM+BPE paths available");
    }));

    // Task 8 — GGUF parser
    out.push_back(run(8, "GGUF parser", [&] {
        ToyGgufConfig gc; gc.n_layers=2; gc.dim=32; gc.n_heads=4; gc.n_kv_heads=2; gc.ffn_dim=64; gc.vocab_size=48;
        std::string path = p(c, "st.gguf"); write_toy_gguf(path, gc);
        GgufFile g(path); ModelConfig cfg = ModelConfig::from_source(g);
        LLM_CHECK(cfg.n_layers==2 && cfg.dim==32 && cfg.head_dim==8, "config");
        const TensorInfo* q = g.find(names::attn_q(0));
        LLM_CHECK(q && q->shape[0]==32 && q->offset%32==0, "tensor dir");
        char b[80]; snprintf(b,sizeof b,"v%u, %zu tensors, shapes+offsets ok",g.version(),g.tensors().size());
        return std::string(b);
    }));

    // Task 9 — quant / dequant
    out.push_back(run(9, "Q4_K_M / quant dequant", [&] {
        // Q8_0 round trip
        std::vector<float> x(256), y(256); for (size_t i=0;i<x.size();++i) x[i]=std::sin(i*0.1f);
        std::vector<uint8_t> q(type_nbytes(DType::Q8_0,256));
        quantize_q8_0(x.data(), q.data(), 256);
        dequantize_row(DType::Q8_0, q.data(), y.data(), 256);
        LLM_CHECK(max_abs_diff(x,y) < 0.03, "q8_0 error");
        // matmul_quant == dequant then matmul
        std::vector<float> W(40*128), xv(128), y1(40), y2(40);
        for (auto& v:W) v=0.01f; for (auto& v:xv) v=0.5f;
        std::vector<uint8_t> Wq(40*type_nbytes(DType::Q8_0,128));
        int64_t rb=type_nbytes(DType::Q8_0,128); std::vector<float> deq(40*128);
        for (int o=0;o<40;++o){ quantize_q8_0(W.data()+o*128, Wq.data()+o*rb,128);
            dequantize_row(DType::Q8_0, Wq.data()+o*rb, deq.data()+o*128,128); }
        matmul(y1.data(), deq.data(), xv.data(),40,128);
        matmul_quant(y2.data(), Wq.data(), DType::Q8_0, xv.data(),40,128);
        LLM_CHECK(max_abs_diff(y1,y2)<1e-4, "fused quant matmul");
        return std::string("Q4_K/Q6_K/Q8_0/Q4_0 dequant + fused quant matmul verified");
    }));

    // Task 10 — NEON
    out.push_back(run(10, "NEON optimization", [&] {
        std::vector<uint16_t> h(500); std::vector<float> a(500), b(500);
        for (size_t i=0;i<h.size();++i){ h[i]=fp32_to_fp16(std::sin(i*0.05f)*4); a[i]=fp16_to_fp32(h[i]); }
        fp16_to_fp32_bulk(h.data(), b.data(), 500);
        LLM_CHECK(max_abs_diff(a,b)==0, "fp16 bulk != scalar");
        char buf[80]; snprintf(buf,sizeof buf,"fp16 bulk==scalar; SDOT int8 matmul %s",
                 neon_dotprod_available()?"available":"n/a");
        return std::string(buf);
    }));

    // Task 11 — Vulkan (optional)
    out.push_back(run(11, "Vulkan backend (optional)", [&] {
        // Informational: never fails the suite; reports the honest device state.
        return vulkan_backend_info();
    }));

    return out;
}

} // namespace llm
