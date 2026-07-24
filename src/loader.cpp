// loader.cpp — streaming LayerLoader implementation (see loader.h).
#include "llm/loader.h"
#include "llm/common.h"
#include "llm/quant.h"
#include "llm/linear.h"

#include <cstring>
#include <algorithm>

namespace llm {

const char* LayerLoader::role_suffix(Role r) {
    switch (r) {
        case Role::AttnNorm: return "attn_norm.weight";
        case Role::AttnQ:    return "attn_q.weight";
        case Role::AttnK:    return "attn_k.weight";
        case Role::AttnV:    return "attn_v.weight";
        case Role::AttnOut:  return "attn_output.weight";
        case Role::FfnNorm:  return "ffn_norm.weight";
        case Role::FfnGate:  return "ffn_gate.weight";
        case Role::FfnUp:    return "ffn_up.weight";
        case Role::FfnDown:  return "ffn_down.weight";
        case Role::AttnQBias: return "attn_q.bias";
        case Role::AttnKBias: return "attn_k.bias";
        case Role::AttnVBias: return "attn_v.bias";
        case Role::AttnPostNorm: return "post_attention_norm.weight";
        case Role::FfnPostNorm:  return "post_ffw_norm.weight";
        case Role::AttnQNorm:    return "attn_q_norm.weight";
        case Role::AttnKNorm:    return "attn_k_norm.weight";
        case Role::AttnQKV:      return "attn_qkv.weight";
        case Role::FfnGateInp:   return "ffn_gate_inp.weight";
        case Role::FfnGateExps:  return "ffn_gate_exps.weight";
        case Role::FfnUpExps:    return "ffn_up_exps.weight";
        case Role::FfnDownExps:  return "ffn_down_exps.weight";
        case Role::AttnNormBias: return "attn_norm.bias";
        case Role::FfnNormBias:  return "ffn_norm.bias";
        case Role::AttnQKVBias:  return "attn_qkv.bias";
        case Role::AttnOutBias:  return "attn_output.bias";
        case Role::FfnUpBias:    return "ffn_up.bias";
        case Role::FfnDownBias:  return "ffn_down.bias";
        default:             return "?";
    }
}

std::string LayerLoader::role_name(int layer, Role r) const {
    return names::blk(layer, role_suffix(r));
}

static bool is_post_norm(Role r) {
    return r == Role::AttnPostNorm || r == Role::FfnPostNorm;
}
static bool is_qk_norm(Role r) {
    return r == Role::AttnQNorm || r == Role::AttnKNorm;
}
static bool is_norm(Role r) {
    return r == Role::AttnNorm || r == Role::FfnNorm || is_post_norm(r) || is_qk_norm(r);
}
static bool is_bias(Role r) {
    return r == Role::AttnQBias || r == Role::AttnKBias || r == Role::AttnVBias ||
           r == Role::AttnNormBias || r == Role::FfnNormBias || r == Role::AttnQKVBias ||
           r == Role::AttnOutBias || r == Role::FfnUpBias || r == Role::FfnDownBias;
}
// Dense projections that a fused-projection arch (Phi-3) or an MoE arch
// (Mixtral) replaces with fused / packed expert tensors — absent there, so
// optional. The active block only reads the ones its architecture uses.
static bool is_split_proj(Role r) {
    return r == Role::AttnQ || r == Role::AttnK || r == Role::AttnV ||
           r == Role::FfnGate || r == Role::FfnUp || r == Role::FfnDown;
}
// Mixtral MoE tensors: router + the packed 3D expert projections.
static bool is_moe_role(Role r) {
    return r == Role::FfnGateInp || r == Role::FfnGateExps ||
           r == Role::FfnUpExps || r == Role::FfnDownExps;
}
// 1-D fp32 weights (norms, biases): tiny, always dequantized on load.
static bool is_1d_fp32(Role r) { return is_norm(r) || is_bias(r); }
// Roles that may legitimately be absent (arch-dependent). Missing -> invalid ref.
static bool is_optional(Role r) {
    return is_bias(r) || is_post_norm(r) || is_qk_norm(r) || is_split_proj(r) ||
           r == Role::AttnQKV || is_moe_role(r);
}
LayerLoader::LayerLoader(WeightSource* src, ModelConfig cfg, Options opt)
    : src_(src), cfg_(cfg), opt_(opt), n_layers_(cfg.n_layers) {
    LLM_CHECK(src_ != nullptr, "LayerLoader: null source");
    if (opt_.n_buffers < 1) opt_.n_buffers = 1;
    slots_.resize(opt_.n_buffers);
    layer_stats_.resize(n_layers_ > 0 ? n_layers_ : 0);

    // ---- resolve global weights -----------------------------------------
    embd_info_ = src_->find(names::token_embd);
    LLM_CHECK(embd_info_ != nullptr, "missing token_embd.weight");

    // output_norm (fp32, small — dequantize once).
    if (const TensorInfo* on = src_->find(names::output_norm)) {
        out_norm_.resize(on->numel() * sizeof(float));
        std::vector<uint8_t> raw(on->nbytes);
        src_->read_raw(*on, raw.data());
        dequantize_row(on->dtype, raw.data(),
                       reinterpret_cast<float*>(out_norm_.data()), on->numel());
    }

    // output / lm_head. Default: resident (fastest — used every token). With
    // Options.stream_lm_head, stream it row-blocked in project_output so the full
    // vocab*dim head is not resident (the RAM<->speed knob for RAM-bound models).
    const TensorInfo* out = src_->find(names::output);
    if (out) {
        if (opt_.stream_lm_head) {
            out_weight_info_ = out;   // streamed per token in project_output()
        } else {
            out_weight_.resize(out->nbytes);           // resident (default)
            src_->read_raw(*out, out_weight_.data());
            out_weight_ref_ = {out_weight_.data(), out->dtype, out->shape[0],
                               out->shape.size() > 1 ? out->shape[1] : cfg_.dim};
        }
    } else {
        // Tied: lm_head == token_embd. Keep the embedding table resident so we
        // can both look up rows and run the final projection against it.
        embd_resident_.resize(embd_info_->nbytes);
        src_->read_raw(*embd_info_, embd_resident_.data());
        embd_is_resident_ = true;
        out_weight_ref_ = {embd_resident_.data(), embd_info_->dtype,
                           embd_info_->shape[0],
                           embd_info_->shape.size() > 1 ? embd_info_->shape[1] : cfg_.dim};
    }

    // Optional output_norm bias (GPT-2 LayerNorm).
    if (const TensorInfo* onb = src_->find("output_norm.bias")) {
        out_norm_bias_.resize(onb->numel() * sizeof(float));
        std::vector<uint8_t> raw(onb->nbytes);
        src_->read_raw(*onb, raw.data());
        dequantize_row(onb->dtype, raw.data(),
                       reinterpret_cast<float*>(out_norm_bias_.data()), onb->numel());
        out_norm_bias_present_ = true;
    }

    // Optional learned position embeddings (GPT-2): kept resident, fp32.
    pos_embd_info_ = src_->find("position_embd.weight");
    if (pos_embd_info_) {
        pos_embd_.resize(pos_embd_info_->numel());
        std::vector<uint8_t> raw(pos_embd_info_->nbytes);
        src_->read_raw(*pos_embd_info_, raw.data());
        dequantize_row(pos_embd_info_->dtype, raw.data(), pos_embd_.data(),
                       pos_embd_info_->numel());
        pos_embd_is_resident_ = true;
    }

    // ---- #37: pin as many hot layers as fit under the RAM budget ---------
    pinned_mask_.assign(n_layers_ > 0 ? (size_t)n_layers_ : 0, 0);
    if (opt_.ram_budget_bytes > 0 && n_layers_ > 0)
        plan_and_pin_layers();

    // Start the prefetch worker only if cold layers remain to stream; with every
    // layer pinned there is nothing left to prefetch.
    if (opt_.async && opt_.n_buffers > 1 && n_pinned_ < n_layers_)
        worker_ = std::thread([this] { worker_loop(); });
}

// ---- #37: ram-budget residency planning ----------------------------------
// Predicted resident bytes for one layer's weights, computed from the tensor
// directory WITHOUT loading. Matches load_weight_into's buffer sizing exactly
// (fp32 -> numel*4, quantized -> on-disk nbytes), so the plan is precise for a
// homogeneous transformer stack (every block identical shape — the real case).
size_t LayerLoader::estimate_layer_bytes(int layer) const {
    size_t total = 0;
    for (int r = 0; r < (int)Role::COUNT; ++r) {
        const TensorInfo* ti = src_->find(role_name(layer, (Role)r));
        if (!ti) continue;
        const bool one_d    = is_1d_fp32((Role)r);
        const bool want_fp32 = one_d || opt_.residency == Residency::FP32;
        total += want_fp32 ? (size_t)ti->numel() * sizeof(float) : (size_t)ti->nbytes;
    }
    return total;
}

// Pin the leading run of layers [0, target) resident under the byte ceiling.
// Two regimes: budget covers every layer -> pin all (zero per-token streaming,
// decode goes compute-bound); otherwise reserve headroom for the cold-streaming
// ring and pin as many contiguous layers as fit. A per-layer guard using the
// ACTUAL materialized size keeps peak weight RSS <= budget even if a layer's
// cost differs from the estimate.
void LayerLoader::plan_and_pin_layers() {
    const size_t budget  = opt_.ram_budget_bytes;
    const size_t globals = out_norm_.size() + out_weight_.size() + embd_resident_.size();
    const size_t per_layer = estimate_layer_bytes(0);
    if (per_layer == 0) return;

    const size_t all_weights = globals + (size_t)n_layers_ * per_layer;
    size_t target;
    if (budget >= all_weights) {
        target = (size_t)n_layers_;                    // pin everything
    } else {
        const size_t ring = (size_t)opt_.n_buffers * per_layer;  // cold-stream headroom
        const size_t base = globals + ring;
        target = budget > base ? std::min<size_t>((budget - base) / per_layer,
                                                  (size_t)n_layers_) : 0;
    }
    if (target == 0) return;

    pinned_.resize(n_layers_);
    // Reserve ring headroom while pinning unless we pin every layer (then nothing
    // streams and the ring stays empty).
    const size_t ring_reserve = (target < (size_t)n_layers_)
                              ? (size_t)opt_.n_buffers * per_layer : 0;
    for (size_t l = 0; l < target; ++l) {
        fill_slot(pinned_[l], (int)l);
        size_t sb = 0;
        for (int r = 0; r < (int)Role::COUNT; ++r) sb += pinned_[l].buf[r].size();
        if (globals + pinned_bytes_ + sb + ring_reserve > budget) {  // ceiling guard
            for (int r = 0; r < (int)Role::COUNT; ++r) { pinned_[l].buf[r] = {}; pinned_[l].ref[r] = WeightRef{}; }
            pinned_[l].layer = -1;
            break;
        }
        pinned_[l].state = Slot::State::Ready;
        pinned_mask_[l]  = 1;
        pinned_bytes_   += sb;
        ++n_pinned_;
    }
}

LayerLoader::~LayerLoader() {
    if (worker_.joinable()) {
        { std::lock_guard<std::mutex> lk(mutex_); stop_ = true; }
        cv_job_.notify_all();
        worker_.join();
    }
}

// ---- per-weight materialization ------------------------------------------
void LayerLoader::load_weight_into(Slot& s, Role role, int layer) {
    const std::string name = role_name(layer, role);
    const TensorInfo* ti = src_->find(name);
    if (ti == nullptr) {
        // Optional roles (e.g. Qwen2 q/k/v biases) are simply absent in most
        // architectures — leave an invalid ref and skip.
        LLM_CHECK(is_optional(role), "missing tensor: " + name);
        s.buf[(int)role].clear();
        s.ref[(int)role] = WeightRef{};
        return;
    }

    // Flatten packed 3D expert tensors [n_expert, a, b] to [n_expert*a, b]; the
    // MoE block indexes each expert's [a, b] slice. Others are 1-D or 2-D.
    const int64_t n_out = ti->shape.size() == 3 ? ti->shape[0] * ti->shape[1] : ti->shape[0];
    const int64_t n_in  = ti->shape.size() == 3 ? ti->shape[2]
                        : (ti->shape.size() > 1 ? ti->shape[1] : 1);

    double t0 = now_sec();
    const bool one_d = is_1d_fp32(role);
    const bool want_fp32 = one_d || opt_.residency == Residency::FP32;

    auto& buf = s.buf[(int)role];
    if (want_fp32) {
        // read raw -> dequantize into fp32 resident buffer
        std::vector<uint8_t> raw(ti->nbytes);
        src_->read_raw(*ti, raw.data());
        stats_.io_us += (uint64_t)((now_sec() - t0) * 1e6);
        stats_.bytes_read += ti->nbytes;

        double t1 = now_sec();
        buf.resize(ti->numel() * sizeof(float));
        dequantize_row(ti->dtype, raw.data(),
                       reinterpret_cast<float*>(buf.data()), ti->numel());
        stats_.dequant_us += (uint64_t)((now_sec() - t1) * 1e6);
        // 1-D vectors (norms, biases) are stored flat: n_out=1, n_in=length.
        s.ref[(int)role] = {buf.data(), DType::F32, one_d ? 1 : n_out,
                            one_d ? n_out : n_in};
    } else {
        // keep raw quantized bytes resident (dequant happens in matmul_quant)
        buf.resize(ti->nbytes);
        src_->read_raw(*ti, buf.data());
        stats_.io_us += (uint64_t)((now_sec() - t0) * 1e6);
        stats_.bytes_read += ti->nbytes;
        s.ref[(int)role] = {buf.data(), ti->dtype, n_out, n_in};
    }
}

void LayerLoader::fill_slot(Slot& s, int layer) {
    // Snapshot global counters to derive this layer's own io/dequant/bytes.
    uint64_t io0 = stats_.io_us.load(), dq0 = stats_.dequant_us.load(),
             by0 = stats_.bytes_read.load();
    for (int r = 0; r < (int)Role::COUNT; ++r)
        load_weight_into(s, (Role)r, layer);
    s.layer = layer;
    if (layer >= 0 && layer < (int)layer_stats_.size()) {
        layer_stats_[layer].io_us = stats_.io_us.load() - io0;
        layer_stats_[layer].dequant_us = stats_.dequant_us.load() - dq0;
        layer_stats_[layer].bytes = stats_.bytes_read.load() - by0;
    }
}

// ---- prefetch worker ------------------------------------------------------
void LayerLoader::worker_loop() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mutex_);
            cv_job_.wait(lk, [this] { return stop_ || !jobs_.empty(); });
            if (stop_) return;
            job = jobs_.front();
            jobs_.pop_front();
            slots_[job.slot].state = Slot::State::Loading;
            slots_[job.slot].layer = job.layer;
        }
        fill_slot(slots_[job.slot], job.layer);   // I/O + dequant, unlocked
        {
            std::lock_guard<std::mutex> lk(mutex_);
            slots_[job.slot].state = Slot::State::Ready;
            stats_.layers_loaded += 1;
        }
        cv_ready_.notify_all();
    }
}

void LayerLoader::enqueue(int slot, int layer) {
    // caller holds mutex_
    for (const auto& j : jobs_) if (j.slot == slot && j.layer == layer) return;
    if (slots_[slot].state == Slot::State::Loading && slots_[slot].layer == layer) return;
    jobs_.push_back({slot, layer});
    cv_job_.notify_one();
}

// ---- public API -----------------------------------------------------------
bool LayerLoader::loadLayer(int layer) {
    LLM_CHECK(layer >= 0 && layer < n_layers_, "loadLayer: out of range");

    // #37: pinned hot layers are always resident — no I/O, no ring slot.
    if (!pinned_mask_.empty() && pinned_mask_[layer]) {
        active_ = &pinned_[layer];
        current_ = -1;                 // the ring is not the active source
        stats_.prefetch_hits += 1;
        return true;
    }

    if (!opt_.async || opt_.n_buffers == 1) {
        // Synchronous single-buffer path: strictly one block resident.
        Slot& s = slots_[0];
        if (s.layer != layer) { fill_slot(s, layer); stats_.layers_loaded += 1; s.state = Slot::State::Ready; stats_.prefetch_misses += 1; }
        else stats_.prefetch_hits += 1;
        current_ = 0;
        active_ = &slots_[0];
        return true;
    }

    std::unique_lock<std::mutex> lk(mutex_);
    // Is `layer` already ready or being loaded in some slot?
    int s = -1;
    for (int i = 0; i < (int)slots_.size(); ++i)
        if (slots_[i].layer == layer &&
            (slots_[i].state == Slot::State::Ready || slots_[i].state == Slot::State::Loading)) {
            s = i; break;
        }
    if (s >= 0) {
        if (slots_[s].state == Slot::State::Ready) stats_.prefetch_hits += 1;
        else stats_.prefetch_misses += 1;   // requested before prefetch finished
    } else {
        // Not present: pick a victim (any slot other than current) and load it
        // with priority (front of queue).
        stats_.prefetch_misses += 1;
        s = (current_ >= 0) ? other_slot(current_) : 0;
        // don't clobber a slot still being consumed
        if (s == current_) s = other_slot(s);
        jobs_.push_front({s, layer});
        slots_[s].state = Slot::State::Loading;
        slots_[s].layer = layer;
        cv_job_.notify_one();
    }
    // Wait until that slot holds `layer` and is Ready.
    cv_ready_.wait(lk, [&] {
        return slots_[s].layer == layer && slots_[s].state == Slot::State::Ready;
    });
    current_ = s;
    active_ = &slots_[current_];

    // Kick off prefetch of the next block into the other buffer.
    if (opt_.n_buffers > 1 && layer + 1 < n_layers_) {
        int nb = other_slot(current_);
        if (!(slots_[nb].layer == layer + 1)) enqueue(nb, layer + 1);
    }
    return true;
}

void LayerLoader::unloadLayer() {
    // With the ring buffer we recycle lazily; unload just marks the current
    // slot reusable. Buffers are freed on process exit or when overwritten.
    if (current_ < 0) return;
    // (Intentionally keep the bytes; the prefetcher will overwrite this slot.)
}

WeightRef LayerLoader::getWeight(Role role) const {
    LLM_CHECK(active_ != nullptr, "getWeight before loadLayer");
    return active_->ref[(int)role];
}

void LayerLoader::embed_token(int64_t token, float* dst) const {
    LLM_CHECK(token >= 0 && token < cfg_.vocab_size, "embed_token: id out of range");
    const int64_t dim = cfg_.dim;
    const int64_t row_bytes = type_nbytes(embd_info_->dtype, dim);
    if (embd_is_resident_) {
        dequantize_row(embd_info_->dtype,
                       embd_resident_.data() + token * row_bytes, dst, dim);
    } else {
        // Stream just this one row from disk — cheap, one row per token.
        std::vector<uint8_t> raw(row_bytes);
        src_->read_raw_at(embd_info_->offset + token * row_bytes, raw.data(), row_bytes);
        dequantize_row(embd_info_->dtype, raw.data(), dst, dim);
    }
}

WeightRef LayerLoader::output_norm_weight() const {
    return {out_norm_.data(), DType::F32, 1, cfg_.dim};
}
WeightRef LayerLoader::output_weight() const { return out_weight_ref_; }

void LayerLoader::project_output(const float* x, float* y, ThreadPool* pool) const {
    if (out_weight_ref_.valid()) {          // tied (lm_head == resident embd) or FP32-resident
        linear(y, out_weight_ref_, x, pool);
        return;
    }
    // Non-tied: stream the LM head off disk in row blocks so only a small window
    // is resident. Read-once-per-token, same pattern as a transformer layer.
    const TensorInfo* ti = out_weight_info_;
    const int64_t n_out = ti->shape[0];
    const int64_t n_in  = ti->row_elems();
    const int64_t row_bytes = type_nbytes(ti->dtype, n_in);
    const int64_t BLK = 1024;                // rows per streamed chunk
    std::vector<uint8_t> buf((size_t)std::min<int64_t>(BLK, n_out) * row_bytes);
    for (int64_t r0 = 0; r0 < n_out; r0 += BLK) {
        int64_t rows = std::min<int64_t>(BLK, n_out - r0);
        src_->read_raw_at(ti->offset + (uint64_t)r0 * row_bytes, buf.data(),
                          (uint64_t)rows * row_bytes);
        WeightRef w{buf.data(), ti->dtype, rows, n_in};
        linear(y + r0, w, x, pool);
    }
}

WeightRef LayerLoader::output_norm_bias_weight() const {
    if (!out_norm_bias_present_) return WeightRef{};
    return {out_norm_bias_.data(), DType::F32, 1, cfg_.dim};
}

void LayerLoader::add_pos_embd(int64_t pos, float* dst) const {
    if (!pos_embd_is_resident_) return;
    const int64_t dim = cfg_.dim;
    const float* row = pos_embd_.data() + pos * dim;
    for (int64_t i = 0; i < dim; ++i) dst[i] += row[i];
}

size_t LayerLoader::resident_bytes() const {
    size_t total = out_norm_.size() + out_weight_.size() + embd_resident_.size();
    for (const auto& s : slots_)
        for (int r = 0; r < (int)Role::COUNT; ++r) total += s.buf[r].size();
    total += pinned_bytes_;   // #37: pinned hot layers
    return total;
}

} // namespace llm
