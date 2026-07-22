// server.cpp — web GUI backend for the streaming inference engine.
//
//   llm_server <model> [--port 8080] [--residency fp32|quant] [--mmap]
//              [--threads N] [--ctx N]
//
// Endpoints:
//   GET  /                -> the single-page app (server/index.html)
//   GET  /api/model       -> JSON model/config/tokenizer/backend info
//   GET  /api/generate    -> SSE stream: tokens + per-token per-layer profile + final stats
//   GET  /api/selftest    -> JSON results of the 11-task self-test
//
// One model, one generation at a time (guarded by a mutex).
#include "llm/runtime.h"
#include "llm/neon.h"
#include "llm/vulkan_backend.h"
#include "llm/selftest.h"
#include "server/http.h"

#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

using namespace llm;

static std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        switch (c) {
            case '"': o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if ((unsigned char)c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += c;
        }
    }
    return o;
}

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model> [--port N] [--residency fp32|quant] "
                        "[--mmap] [--threads N] [--ctx N]\n", argv[0]);
        return 2;
    }
    std::string model = argv[1];
    int port = 8080, threads = 0, ctx = 0;
    LayerLoader::Options opt; opt.residency = Residency::Quantized;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto nx = [&]{ return (i + 1 < argc) ? argv[++i] : "0"; };
        if (a == "--port") port = std::stoi(nx());
        else if (a == "--residency") opt.residency = std::string(nx()) == "fp32" ? Residency::FP32 : Residency::Quantized;
        else if (a == "--mmap") opt.use_mmap = true;
        else if (a == "--threads") threads = std::stoi(nx());
        else if (a == "--ctx") ctx = std::stoi(nx());
    }

    std::unique_ptr<Runtime> rt;
    std::string model_err;
    try {
        auto src = open_model(model, opt.use_mmap);
        rt = std::make_unique<Runtime>(std::move(src), opt, ctx, threads);
        printf("loaded %s\n  %s\n", model.c_str(), rt->config().summary().c_str());
    } catch (const std::exception& e) {
        model_err = e.what();
        fprintf(stderr, "model load failed: %s (server still serves UI + selftest)\n", e.what());
    }

    std::mutex gen_mutex;
    // Locate index.html relative to the binary's launch dir or server/.
    std::string html = read_file("server/index.html");
    if (html.empty()) html = read_file("index.html");
    if (html.empty()) html = "<!doctype html><h1>index.html not found</h1>"
                             "<p>Run llm_server from the project root.</p>";

    http::Server srv(port);

    srv.get("/", [&](http::Request&, http::Response& res) {
        res.send(200, "text/html; charset=utf-8", html);
    });

    srv.get("/api/model", [&](http::Request&, http::Response& res) {
        std::string j = "{";
        if (!rt) { j += "\"ok\":false,\"error\":\"" + json_escape(model_err) + "\""; }
        else {
            const auto& c = rt->config();
            const char* tk = rt->tokenizer().kind() == Tokenizer::Kind::BPE ? "BPE" :
                             rt->tokenizer().kind() == Tokenizer::Kind::SentencePiece ? "SentencePiece" : "byte";
            char b[1400];
            snprintf(b, sizeof b,
                "\"ok\":true,\"model\":\"%s\",\"layers\":%lld,\"heads\":%lld,\"kv_heads\":%lld,"
                "\"dim\":%lld,\"head_dim\":%lld,\"ffn\":%lld,\"vocab\":%lld,\"ctx\":%lld,"
                "\"rope_theta\":%.1f,\"rms_eps\":%.2e,\"tokenizer\":\"%s\","
                "\"residency\":\"%s\",\"neon_dotprod\":%s,\"backend\":\"%s\"",
                json_escape(model).c_str(), (long long)c.n_layers, (long long)c.n_heads,
                (long long)c.n_kv_heads, (long long)c.dim, (long long)c.head_dim,
                (long long)c.ffn_dim, (long long)c.vocab_size, (long long)c.ctx_len,
                c.rope_theta, c.rms_eps, tk,
                opt.residency == Residency::FP32 ? "fp32" : "quant",
                neon_dotprod_available() ? "true" : "false",
                json_escape(vulkan_backend_info()).c_str());
            j += b;
        }
        j += "}";
        res.send(200, "application/json", j);
    });

    srv.get("/api/selftest", [&](http::Request&, http::Response& res) {
        auto results = run_selftests("/tmp");
        std::string j = "{\"results\":[";
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            char b[64];
            snprintf(b, sizeof b, "%.2f", r.ms);
            j += std::string(i ? "," : "") + "{\"task\":" + std::to_string(r.task) +
                 ",\"name\":\"" + json_escape(r.name) + "\",\"ok\":" + (r.ok ? "true" : "false") +
                 ",\"detail\":\"" + json_escape(r.detail) + "\",\"ms\":" + b + "}";
        }
        j += "]}";
        res.send(200, "application/json", j);
    });

    srv.get("/api/generate", [&](http::Request& req, http::Response& res) {
        if (!rt) { res.send(503, "text/plain", "model not loaded: " + model_err); return; }
        std::string prompt = req.q("prompt", "Hello");
        int n = std::stoi(req.q("n", "64"));
        SamplerConfig scfg;
        scfg.temperature = std::stof(req.q("temp", "0.8"));
        if (req.q("greedy", "0") == "1") scfg.temperature = 0.f;
        bool profile = req.q("profile", "1") == "1";

        std::unique_lock<std::mutex> lk(gen_mutex, std::try_to_lock);
        if (!lk.owns_lock()) { res.send(429, "text/plain", "busy"); return; }

        res.begin_sse();
        rt->reset();
        bool alive = true;

        if (profile) {
            rt->set_profile_sink([&](int idx, const std::vector<Transformer::LayerTiming>& t, size_t peak) {
                if (!alive) return;
                std::string j = "{\"token\":" + std::to_string(idx) + ",\"peak_rss\":" +
                                std::to_string(peak) + ",\"layers\":[";
                for (size_t l = 0; l < t.size(); ++l) {
                    char b[160];
                    snprintf(b, sizeof b, "%s{\"io\":%.3f,\"deq\":%.3f,\"cmp\":%.3f,\"rss\":%zu}",
                             l ? "," : "", t[l].io_ms, t[l].dequant_ms, t[l].compute_ms, t[l].rss_bytes);
                    j += b;
                }
                j += "]}";
                alive = res.sse_event(j, "profile");
            });
        }

        GenStats st;
        rt->generate(prompt, n, scfg,
            [&](const std::string& piece, int64_t id) {
                std::string j = "{\"piece\":\"" + json_escape(piece) + "\",\"id\":" + std::to_string(id) + "}";
                alive = res.sse_event(j, "token");
                return alive;
            }, &st);

        // final stats
        char b[900];
        snprintf(b, sizeof b,
            "{\"prompt_tokens\":%d,\"gen_tokens\":%d,\"ttft\":%.4f,\"prefill_tok_s\":%.3f,"
            "\"decode_tok_s\":%.3f,\"weights_mb\":%.2f,\"kv_mb\":%.2f,\"streamed_mb\":%.1f,"
            "\"prefetch_hits\":%llu,\"prefetch_misses\":%llu,\"ctx_used\":%d,\"ctx_max\":%d,\"peak_rss_mb\":%.1f}",
            st.prompt_tokens, st.gen_tokens, st.ttft_s, st.prefill_tok_s, st.decode_tok_s,
            st.weights_resident_bytes / 1e6, st.kv_bytes / 1e6, st.bytes_read / 1e6,
            (unsigned long long)st.prefetch_hits, (unsigned long long)st.prefetch_misses,
            st.ctx_used, st.ctx_max, rt->peak_rss() / 1e6);
        res.sse_event(b, "done");
    });

    return srv.run();
}
