// sipllm_ffi.cpp — implementation of the stable C ABI in sipllm_ffi.h.
//
// A thin translation layer: it owns a llm::Runtime, marshals POD structs to/from
// the engine's C++ types, adapts the std::function streaming callback to a C
// function pointer, and exposes an atomic cancel flag. No inference logic lives
// here — it only wires the C boundary to the engine.
#include "sipllm_ffi.h"

#include "llm/runtime.h"
#include "llm/auto_tuner.h"
#include "llm/device_profile.h"
#include "llm/vulkan_backend.h"
#include "llm/threadpool.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace llm;

struct sipllm_ctx {
  std::unique_ptr<Runtime> rt;
  std::atomic<bool> cancel{false};
  int threads = 0;
};

namespace {

void set_err(char* buf, int32_t cap, const std::string& msg) {
  if (!buf || cap <= 0) return;
  std::snprintf(buf, static_cast<size_t>(cap), "%s", msg.c_str());
}

ThreadPool::SchedulePolicy to_policy(int32_t p) {
  switch (p) {
    case SIPLLM_SCHED_STATIC: return ThreadPool::SchedulePolicy::Static;
    case SIPLLM_SCHED_FIXED8: return ThreadPool::SchedulePolicy::Fixed8;
    case SIPLLM_SCHED_FIXED16: return ThreadPool::SchedulePolicy::Fixed16;
    case SIPLLM_SCHED_FIXED32: return ThreadPool::SchedulePolicy::Fixed32;
    case SIPLLM_SCHED_PROPORTIONAL2: return ThreadPool::SchedulePolicy::Proportional2;
    case SIPLLM_SCHED_PROPORTIONAL4: return ThreadPool::SchedulePolicy::Proportional4;
    case SIPLLM_SCHED_ADAPTIVE: return ThreadPool::SchedulePolicy::Adaptive;
    default: return ThreadPool::SchedulePolicy::Proportional2;
  }
}

LayerLoader::Options options_from(const sipllm_params& p) {
  LayerLoader::Options opt;
  opt.residency = p.residency_fp32 ? Residency::FP32 : Residency::Quantized;
  opt.use_mmap = p.use_mmap != 0;
  opt.async = p.async_prefetch != 0;
  opt.n_buffers = opt.async ? (p.n_buffers > 0 ? p.n_buffers : 2) : 1;
  opt.stream_lm_head = p.stream_lm_head != 0;
  opt.fast_quant = p.fast_quant != 0;
  return opt;
}

int32_t tokenizer_kind_code(Tokenizer::Kind k) {
  switch (k) {
    case Tokenizer::Kind::SentencePiece: return 0;
    case Tokenizer::Kind::BPE: return 1;
    case Tokenizer::Kind::Byte: return 2;
  }
  return 2;
}

}  // namespace

extern "C" {

void sipllm_params_default(sipllm_params* out) {
  if (!out) return;
  std::memset(out, 0, sizeof(*out));
  out->ram_budget_bytes = 0;
  out->threads = 0;   // hardware_concurrency
  out->max_ctx = 0;   // engine default
  out->n_buffers = 2;
  out->use_mmap = 0;
  out->async_prefetch = 1;
  out->fast_quant = 0;
  out->stream_lm_head = 0;
  out->residency_fp32 = 0;
  out->force_budget = 0;
  out->schedule_policy = SIPLLM_SCHED_PROPORTIONAL2;
}

void sipllm_sampler_default(sipllm_sampler* out) {
  if (!out) return;
  SamplerConfig d;
  out->temperature = d.temperature;
  out->top_k = d.top_k;
  out->top_p = d.top_p;
  out->repeat_penalty = 1.1f;
  out->repeat_last_n = d.repeat_last_n;
  out->seed = d.seed;
}

sipllm_ctx* sipllm_open(const char* model_path, const sipllm_params* p,
                        char* err_buf, int32_t err_cap) {
  if (!model_path) {
    set_err(err_buf, err_cap, "null model_path");
    return nullptr;
  }
  sipllm_params params;
  if (p) params = *p; else sipllm_params_default(&params);
  try {
    LayerLoader::Options opt = options_from(params);

    int threads = params.threads;
    int tuned_policy = -1;  // >=0 => override schedule policy from the auto-tuner
    if (threads < 0) {
      // Auto-tune: device micro-benchmarks pick both the thread count and the
      // schedule policy, cached under the device profile. Opt-in (costs time on
      // first open and needs a writable HOME), so it never runs for threads >= 0.
      const RuntimeProfile prof =
          tune_if_needed(get_hardware_info(), AutoTunerOptions{});
      threads = prof.threads;
      tuned_policy = prof.schedule_policy;
    }
    // threads == 0 -> Runtime/ThreadPool selects hardware_concurrency.

    auto src = open_model(model_path, opt.use_mmap);
    auto ctx = std::unique_ptr<sipllm_ctx>(new sipllm_ctx());
    ctx->rt = std::make_unique<Runtime>(std::move(src), opt, params.max_ctx, threads,
                                        params.ram_budget_bytes, params.force_budget != 0);
    if (ctx->rt->thread_pool()) {
      ctx->rt->thread_pool()->set_policy(
          to_policy(tuned_policy >= 0 ? tuned_policy : params.schedule_policy));
      ctx->threads = ctx->rt->thread_pool()->size();
    }
    return ctx.release();
  } catch (const std::exception& e) {
    set_err(err_buf, err_cap, e.what());
    return nullptr;
  } catch (...) {
    set_err(err_buf, err_cap, "unknown error opening model");
    return nullptr;
  }
}

void sipllm_close(sipllm_ctx* ctx) { delete ctx; }

int32_t sipllm_get_model_info(sipllm_ctx* ctx, sipllm_model_info* out) {
  if (!ctx || !ctx->rt || !out) return -1;
  const ModelConfig& c = ctx->rt->config();
  std::memset(out, 0, sizeof(*out));
  out->n_layers = static_cast<int32_t>(c.n_layers);
  out->n_heads = static_cast<int32_t>(c.n_heads);
  out->n_kv_heads = static_cast<int32_t>(c.n_kv_heads);
  out->dim = c.dim;
  out->vocab_size = c.vocab_size;
  out->ctx_len = c.ctx_len;
  out->tokenizer_kind = tokenizer_kind_code(ctx->rt->tokenizer().kind());
  std::snprintf(out->arch, sizeof(out->arch), "%s", c.arch.c_str());
  return 0;
}

int32_t sipllm_get_threads(sipllm_ctx* ctx) {
  if (!ctx || !ctx->rt) return -1;
  return ctx->rt->thread_pool() ? ctx->rt->thread_pool()->size() : 1;
}

int32_t sipllm_generate(sipllm_ctx* ctx, const char* prompt, int32_t max_new,
                        const sipllm_sampler* scfg, sipllm_token_cb cb, void* user,
                        sipllm_stats* stats, char* err_buf, int32_t err_cap) {
  if (!ctx || !ctx->rt) {
    set_err(err_buf, err_cap, "null context");
    return -1;
  }
  ctx->cancel.store(false, std::memory_order_relaxed);
  SamplerConfig s;
  if (scfg) {
    s.temperature = scfg->temperature;
    s.top_k = scfg->top_k;
    s.top_p = scfg->top_p;
    s.repeat_penalty = scfg->repeat_penalty;
    s.repeat_last_n = scfg->repeat_last_n;
    s.seed = scfg->seed;
  }
  GenStats gs;
  try {
    Runtime::TokenCallback on_token = [&](const std::string& piece, int64_t id) -> bool {
      if (ctx->cancel.load(std::memory_order_relaxed)) return false;
      if (cb) return cb(piece.c_str(), id, user) != 0;
      return true;
    };
    ctx->rt->generate(prompt ? prompt : "", max_new, s, on_token, &gs);
    if (stats) {
      std::memset(stats, 0, sizeof(*stats));
      stats->load_s = gs.load_s;
      stats->ttft_s = gs.ttft_s;
      stats->prefill_s = gs.prefill_s;
      stats->decode_s = gs.decode_s;
      stats->prefill_tok_s = gs.prefill_tok_s;
      stats->decode_tok_s = gs.decode_tok_s;
      {
        size_t peak = ctx->rt->peak_rss();
        stats->peak_rss_bytes = peak ? peak : current_rss_bytes();
      }
      stats->weights_resident_bytes = gs.weights_resident_bytes;
      stats->kv_bytes = gs.kv_bytes;
      stats->bytes_read = gs.bytes_read;
      stats->prefetch_hits = gs.prefetch_hits;
      stats->prefetch_misses = gs.prefetch_misses;
      stats->pinned_layers = gs.pinned_layers;
      stats->n_layers = static_cast<int32_t>(ctx->rt->config().n_layers);
      stats->prompt_tokens = gs.prompt_tokens;
      stats->gen_tokens = gs.gen_tokens;
      stats->ctx_used = gs.ctx_used;
      stats->ctx_max = gs.ctx_max;
    }
    return gs.gen_tokens;
  } catch (const std::exception& e) {
    set_err(err_buf, err_cap, e.what());
    return -1;
  } catch (...) {
    set_err(err_buf, err_cap, "unknown error during generation");
    return -1;
  }
}

void sipllm_cancel(sipllm_ctx* ctx) {
  if (ctx) ctx->cancel.store(true, std::memory_order_relaxed);
}

void sipllm_reset(sipllm_ctx* ctx) {
  if (ctx && ctx->rt) ctx->rt->reset();
}

int32_t sipllm_embed_dim(sipllm_ctx* ctx) {
  if (!ctx || !ctx->rt) return -1;
  return static_cast<int32_t>(ctx->rt->config().dim);
}

int32_t sipllm_embed(sipllm_ctx* ctx, const char* text, int32_t pooling,
                     float* out, int32_t out_cap, char* err_buf, int32_t err_cap) {
  if (!ctx || !ctx->rt || !out) {
    set_err(err_buf, err_cap, "null context or output");
    return -1;
  }
  const int64_t dim = ctx->rt->config().dim;
  const int last_layer = static_cast<int>(ctx->rt->config().n_layers) - 1;
  if (out_cap < dim) {
    set_err(err_buf, err_cap, "output buffer smaller than embedding dim");
    return -1;
  }
  try {
    std::vector<float> mean(static_cast<size_t>(dim), 0.0f);
    std::vector<float> last(static_cast<size_t>(dim), 0.0f);
    int64_t count = 0;
    // Capture the final transformer block's residual stream on each forward.
    // Causal attention makes the LAST position's vector see the whole sequence.
    ctx->rt->set_hidden_hook([&](int layer, const float* x, int64_t d) {
      if (layer != last_layer || d != dim) return;
      for (int64_t i = 0; i < dim; ++i) {
        last[static_cast<size_t>(i)] = x[i];
        mean[static_cast<size_t>(i)] += x[i];
      }
      ++count;
    });
    ctx->rt->reset();
    SamplerConfig greedy;
    greedy.temperature = 0.0f;
    GenStats gs;
    // max_new=1 forces prefill + one decode step; the hook fires per position.
    ctx->rt->generate(text ? text : "", 1, greedy,
                       [](const std::string&, int64_t) { return false; }, &gs);
    ctx->rt->set_hidden_hook(nullptr);
    ctx->rt->reset();

    const float* src = last.data();
    if (pooling == SIPLLM_POOL_MEAN && count > 0) {
      const float inv = 1.0f / static_cast<float>(count);
      for (int64_t i = 0; i < dim; ++i) mean[static_cast<size_t>(i)] *= inv;
      src = mean.data();
    }
    // L2-normalize so cosine similarity == dot product downstream.
    double norm = 0.0;
    for (int64_t i = 0; i < dim; ++i) norm += static_cast<double>(src[i]) * src[i];
    norm = std::sqrt(norm);
    const float inv = norm > 0.0 ? static_cast<float>(1.0 / norm) : 0.0f;
    for (int64_t i = 0; i < dim; ++i) out[i] = src[i] * inv;
    return static_cast<int32_t>(dim);
  } catch (const std::exception& e) {
    ctx->rt->set_hidden_hook(nullptr);
    set_err(err_buf, err_cap, e.what());
    return -1;
  } catch (...) {
    ctx->rt->set_hidden_hook(nullptr);
    set_err(err_buf, err_cap, "unknown error during embedding");
    return -1;
  }
}

const char* sipllm_version(void) { return "0.4.0"; }

int32_t sipllm_vulkan_compiled(void) { return vulkan_compiled() ? 1 : 0; }
int32_t sipllm_vulkan_available(void) { return vulkan_available() ? 1 : 0; }

const char* sipllm_vulkan_info(void) {
  static thread_local std::string buf;
  buf = vulkan_backend_info();
  return buf.c_str();
}

int32_t sipllm_hardware_concurrency(void) {
  int hw = static_cast<int>(std::thread::hardware_concurrency());
  return hw > 0 ? hw : 4;
}

int32_t sipllm_optimal_threads(const char* model_path, uint64_t ram_budget) {
  // model_path / ram_budget are retained for ABI stability; the auto-tuner now
  // profiles the device (model-independent) and caches the result to disk.
  (void)model_path;
  (void)ram_budget;
  try {
    return tune_if_needed(get_hardware_info(), AutoTunerOptions{}).threads;
  } catch (...) {
    return -1;
  }
}

void sipllm_set_log_level(int32_t level) {
  if (level < 0) level = 0;
  if (level > 4) level = 4;
  log_level() = static_cast<LogLevel>(level);
}

}  // extern "C"
