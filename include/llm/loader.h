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
    // Optional Mixtral MoE tensors: router + packed 3D expert projections.
    FfnGateInp, FfnGateExps, FfnUpExps, FfnDownExps,
    // Optional LayerNorm + projection biases (GPT-2 / Phi-2, fully-biased archs).
    AttnNormBias, FfnNormBias, AttnQKVBias, AttnOutBias, FfnUpBias, FfnDownBias,
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
        bool      stream_lm_head = false; // stream non-tied LM head off disk (RAM<->speed knob)
        ThreadPool* dequant_pool = nullptr; // parallelize per-layer dequant
        // Hard ceiling (bytes) on WEIGHT-resident RAM: globals + pinned hot
        // layers + the streaming ring. 0 = unlimited (today's behavior, exact).
        // When >0 the loader pins as many contiguous layers as fit under the
        // ceiling and streams the rest, so peak weight RSS never exceeds it (the
        // RAM<->speed dial, issue #37). Runtime derives this from a total
        // peak-RSS target by subtracting the KV cache + a scratch reserve.
        size_t    ram_budget_bytes = 0;
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
    // y = lm_head @ x -> logits. Streams the LM head off disk (row-blocked) when
    // non-tied so the full vocab*dim head is never resident; tied path is resident.
    void project_output(const float* x, float* y, ThreadPool* pool) const;
    // Optional globals (GPT-2). output_norm_bias: fp32 [dim] or invalid.
    WeightRef output_norm_bias_weight() const;
    bool has_pos_embd() const { return pos_embd_is_resident_; }
    void add_pos_embd(int64_t pos, float* dst) const;   // dst[i] += position_embd[pos][i]

    const ModelConfig& config() const { return cfg_; }
    const Stats& stats() const { return stats_; }
    size_t resident_bytes() const;           // approx current RAM for weights
    int    pinned_layers() const { return n_pinned_; } // resident hot layers (#37)

private:
    struct Slot {
        int   layer = -1;
        enum class State { Empty, Loading, Ready } state = State::Empty;
        // One resident buffer per role.
        std::vector<uint8_t> buf[(int)Role::COUNT];
        WeightRef            ref[(int)Role::COUNT];
    };
    struct Job { int slot = -1; int layer = -1; };

    void   plan_and_pin_layers();            // #37: pin hot layers under budget
    size_t estimate_layer_bytes(int layer) const; // predicted resident bytes
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

    // ---- ram-budget residency (issue #37) --------------------------------
    // With Options.ram_budget_bytes > 0, layers [0, n_pinned_) are materialized
    // once and kept resident in pinned_[l]; loadLayer serves them with no I/O.
    // Remaining cold layers stream through slots_ as before. active_ is the slot
    // getWeight reads from (a pinned slot or the current ring slot), set by every
    // loadLayer path so budget==0 stays byte-for-byte identical to before.
    std::vector<Slot>    pinned_;         // indexed by layer; filled iff pinned
    std::vector<uint8_t> pinned_mask_;    // 1 iff layer is pinned resident
    int                  n_pinned_ = 0;
    size_t               pinned_bytes_ = 0;
    const Slot*          active_ = nullptr;
    std::vector<LayerStat> layer_stats_;   // per-layer io/dequant accounting

    // global weights
    const TensorInfo* embd_info_ = nullptr;   // token_embd (streamed per-row)
    std::vector<uint8_t> embd_resident_;       // used when tied & kept resident
    bool embd_is_resident_ = false;
    std::vector<uint8_t> out_norm_;            // fp32
    std::vector<uint8_t> out_weight_;          // native dtype
    WeightRef out_weight_ref_;
    const TensorInfo* out_weight_info_ = nullptr; // non-tied LM head (streamed)
    std::vector<uint8_t> out_norm_bias_;        // fp32, optional (GPT-2)
    bool out_norm_bias_present_ = false;
    const TensorInfo* pos_embd_info_ = nullptr; // position_embd (GPT-2)
    std::vector<float> pos_embd_;               // resident fp32 [ctx, dim]
    bool pos_embd_is_resident_ = false;

    // prefetch pipeline
    std::thread worker_;
    mutable std::mutex mutex_;
    std::condition_variable cv_job_, cv_ready_;
    std::deque<Job> jobs_;
    bool stop_ = false;

    Stats stats_;
};

} // namespace llm
