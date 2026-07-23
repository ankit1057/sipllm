// loader.cpp — streaming LayerLoader implementation (see loader.h).
#include "llm/loader.h"
#include "llm/common.h"
#include "llm/quant.h"

#include <cstring>

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
        default:             return "?";
    }
}

std::string LayerLoader::role_name(int layer, Role r) const {
    return names::blk(layer, role_suffix(r));
}

static bool is_norm(Role r) { return r == Role::AttnNorm || r == Role::FfnNorm; }
static bool is_bias(Role r) {
    return r == Role::AttnQBias || r == Role::AttnKBias || r == Role::AttnVBias;
}
// 1-D fp32 weights (norms, biases): tiny, always dequantized on load.
static bool is_1d_fp32(Role r) { return is_norm(r) || is_bias(r); }
// Roles that may legitimately be absent (arch-dependent). Missing -> invalid ref.
static bool is_optional(Role r) { return is_bias(r); }

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

    // output / lm_head: keep resident in native dtype (needed every token).
    const TensorInfo* out = src_->find(names::output);
    if (out) {
        out_weight_.resize(out->nbytes);
        src_->read_raw(*out, out_weight_.data());
        out_weight_ref_ = {out_weight_.data(), out->dtype,
                           out->shape[0], out->shape.size() > 1 ? out->shape[1] : cfg_.dim};
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

    if (opt_.async && opt_.n_buffers > 1)
        worker_ = std::thread([this] { worker_loop(); });
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

    const int64_t n_out = ti->shape[0];
    const int64_t n_in  = ti->shape.size() > 1 ? ti->shape[1] : 1;

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

    if (!opt_.async || opt_.n_buffers == 1) {
        // Synchronous single-buffer path: strictly one block resident.
        Slot& s = slots_[0];
        if (s.layer != layer) { fill_slot(s, layer); stats_.layers_loaded += 1; s.state = Slot::State::Ready; stats_.prefetch_misses += 1; }
        else stats_.prefetch_hits += 1;
        current_ = 0;
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
    LLM_CHECK(current_ >= 0, "getWeight before loadLayer");
    return slots_[current_].ref[(int)role];
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

size_t LayerLoader::resident_bytes() const {
    size_t total = out_norm_.size() + out_weight_.size() + embd_resident_.size();
    for (const auto& s : slots_)
        for (int r = 0; r < (int)Role::COUNT; ++r) total += s.buf[r].size();
    return total;
}

} // namespace llm
