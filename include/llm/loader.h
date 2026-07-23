// loader.h — Task 4 / Phase 3: the streaming LayerLoader.
//
// Contract (from the design brief):
//     bool     loadLayer(int layer);   // make a block resident
//     void     unloadLayer();          // release the current block
//     WeightRef getWeight(role);       // fetch a resident weight
//
// Only one transformer block need be resident at a time. With async prefetch
// enabled we keep a small ring of buffers so that while the compute thread runs
// block N, a background storage+dequant thread is already materializing block
// N+1 — the three-stage pipeline: Storage(pread) -> Dequant -> ready buffer.
//
// Residency modes:
//   Quantized : keep each weight's raw on-disk bytes resident (~140 MB/layer
//               for 8B-Q4_K_M) and dequantize per-row inside matmul_quant.
//               This is what makes a 7-8B model fit an 8 GB device.
//   FP32      : dequantize each weight to fp32 on load (simplest; larger). Used
//               by the toy model and the numeric-equivalence tests.
// Norm weights are always fp32 (they are tiny and read every block).
#pragma once

#include "llm/model.h"
#include "llm/threadpool.h"
#include "llm/weight_source.h"

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace llm {

// Per-block weight roles, in the order they are consumed by a forward pass.
enum class Role {
    AttnNorm = 0, AttnQ, AttnK, AttnV, AttnOut,
    FfnNorm, FfnGate, FfnUp, FfnDown,
    // Optional roles — absent in most architectures, loaded only when the
    // tensor exists (see LayerLoader::is_optional). Qwen2 adds q/k/v biases.
    AttnQBias, AttnKBias, AttnVBias,
    // Optional post-sublayer norms (Gemma 2): applied to the attn / FFN output
    // before the residual add.
    AttnPostNorm, FfnPostNorm,
    // Optional per-head query/key norms (Gemma 3), applied before RoPE.
    AttnQNorm, AttnKNorm,
    // Optional fused projection (Phi-3): one attn_qkv.weight split into q/k/v.
    AttnQKV,
    COUNT
};

enum class Residency { Quantized, FP32 };

class LayerLoader {
public:
    struct Options {
        Residency residency   = Residency::Quantized;
        int       n_buffers    = 2;      // 1 = strict single block, 2 = double buffer
        bool      async        = true;   // run the background prefetch thread
        bool      use_mmap     = false;  // pread by default (Phase 6 spec)
        ThreadPool* dequant_pool = nullptr; // parallelize per-layer dequant
    };

    struct Stats {
        std::atomic<uint64_t> bytes_read{0};
        std::atomic<uint64_t> layers_loaded{0};
        std::atomic<uint64_t> prefetch_hits{0};   // loadLayer found it ready
        std::atomic<uint64_t> prefetch_misses{0}; // had to load synchronously
        std::atomic<uint64_t> io_us{0};           // microseconds in pread
        std::atomic<uint64_t> dequant_us{0};      // microseconds dequantizing
    };

    // Per-layer accounting (last time the layer was materialized). Indexed by
    // layer id; feeds the storage profiler and the live per-layer visualization.
    struct LayerStat {
        uint64_t bytes = 0;       // bytes read for this layer
        uint64_t io_us = 0;       // pread time
        uint64_t dequant_us = 0;  // dequant time
    };
    LayerStat layer_stat(int layer) const {
        return (layer >= 0 && layer < (int)layer_stats_.size()) ? layer_stats_[layer]
                                                                : LayerStat{};
    }

    LayerLoader(WeightSource* src, ModelConfig cfg, Options opt);
    ~LayerLoader();

    // Make `layer` resident and current. Blocks until ready. Returns true.
    bool loadLayer(int layer);
    // Release the current block for reuse (double-buffer recycling).
    void unloadLayer();
    // Weight from the current resident block.
    WeightRef getWeight(Role role) const;

    // ---- global (always-resident) weights --------------------------------
    // Write the embedding row for `token` (cfg.dim fp32 values) into dst.
    void embed_token(int64_t token, float* dst) const;
    WeightRef output_norm_weight() const;   // fp32, [dim]
    WeightRef output_weight() const;         // native dtype, [vocab, dim]

    const ModelConfig& config() const { return cfg_; }
    const Stats& stats() const { return stats_; }
    size_t resident_bytes() const;           // approx current RAM for weights

private:
    struct Slot {
        int   layer = -1;
        enum class State { Empty, Loading, Ready } state = State::Empty;
        // One resident buffer per role.
        std::vector<uint8_t> buf[(int)Role::COUNT];
        WeightRef            ref[(int)Role::COUNT];
    };
    struct Job { int slot; int layer; };

    void   fill_slot(Slot& s, int layer);    // no lock held; does I/O + dequant
    void   load_weight_into(Slot& s, Role role, int layer);
    int    other_slot(int s) const { return (s + 1) % (int)slots_.size(); }
    void   worker_loop();
    void   enqueue(int slot, int layer);     // caller holds mutex_

    static const char* role_suffix(Role r);
    std::string role_name(int layer, Role r) const;

    WeightSource* src_;
    ModelConfig   cfg_;
    Options       opt_;
    int64_t       n_layers_ = 0;

    std::vector<Slot> slots_;
    int current_ = -1;
    std::vector<LayerStat> layer_stats_;   // per-layer io/dequant accounting

    // global weights
    const TensorInfo* embd_info_ = nullptr;   // token_embd (streamed per-row)
    std::vector<uint8_t> embd_resident_;       // used when tied & kept resident
    bool embd_is_resident_ = false;
    std::vector<uint8_t> out_norm_;            // fp32
    std::vector<uint8_t> out_weight_;          // native dtype
    WeightRef out_weight_ref_;

    // prefetch pipeline
    std::thread worker_;
    mutable std::mutex mutex_;
    std::condition_variable cv_job_, cv_ready_;
    std::deque<Job> jobs_;
    bool stop_ = false;

    Stats stats_;
};

} // namespace llm
