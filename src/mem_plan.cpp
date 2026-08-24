#include "llm/mem_plan.h"
#include "llm/common.h"
#include "llm/format.h"
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <sstream>

namespace llm {

std::string human_bytes(size_t bytes) {
    char buf[64];
    if (bytes < 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    } else if (bytes < 1024 * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.1f MB", (double)bytes / 1e6);
    } else {
        snprintf(buf, sizeof(buf), "%.2f GB", (double)bytes / 1e9);
    }
    return buf;
}

// Compute the size in bytes for a given dtype and number of elements.
static int64_t type_nbytes_plan(DType dtype, int64_t numel) {
    switch (dtype) {
        case DType::F32:  return numel * 4;
        case DType::F16:  return numel * 2;
        case DType::Q4_0: return (numel / 32) * 18; // 32 elems in 18 bytes
        case DType::Q4_1: return (numel / 32) * 20;
        case DType::Q5_0: return (numel / 32) * 22;
        case DType::Q5_1: return (numel / 32) * 24;
        case DType::Q8_0: return (numel / 32) * 34;
        default: return numel * 4;
    }
}

MemoryPlan plan_memory(const WeightSource& src, const ModelConfig& cfg,
                       const BudgetRequest& req) {
    MemoryPlan plan;
    if (req.budget_bytes == 0) {
        plan.feasible = true;
        plan.ctx = req.ctx_req > 0 ? req.ctx_req : 
                   (cfg.ctx_len > 0 ? std::min<int>((int)cfg.ctx_len, req.default_ctx_cap) : 2048);
        plan.n_buffers = req.async_req ? req.n_buffers_req : 1;
        plan.stream_lm_head = req.stream_head_req;
        plan.weight_ceiling = 0; // Legacy logic in runtime.cpp
        return plan;
    }

    // 1. Ledger fixed costs
    MemLedger fixed;
    fixed.base = 24 * 1024 * 1024; // 24 MB process slop
    fixed.scratch = (size_t)(cfg.dim * 8 + cfg.ffn_dim * 2 + cfg.q_dim() + 2 * cfg.kv_dim()) * sizeof(float);
    fixed.logits = (size_t)cfg.vocab_size * sizeof(float);

    size_t layer_size = 0;
    size_t out_norm_size = 0;
    size_t out_resident_size = 0;
    size_t out_streamed_size = 0;
    size_t embd_size = 0;
    bool tied = true;

    if (auto* out = src.find(names::output)) {
        tied = false;
        out_resident_size = req.residency == Residency::FP32 ? out->numel() * sizeof(float) : out->nbytes;
        int64_t n_in = out->shape.size() > 1 ? out->shape[1] : cfg.dim;
        out_streamed_size = std::min<int64_t>(1024, out->shape[0]) * type_nbytes_plan(out->dtype, n_in);
    }

    for (const auto& ti : src.tensors()) {
        bool one_d = (ti.name.find("norm") != std::string::npos) || (ti.name.find("bias") != std::string::npos);
        bool want_fp32 = one_d || req.residency == Residency::FP32;
        size_t size = want_fp32 ? ti.numel() * sizeof(float) : ti.nbytes;

        if (ti.name == names::token_embd) {
            embd_size = size;
        } else if (ti.name == names::output_norm || ti.name == "output_norm.bias" || ti.name == "position_embd.weight") {
            out_norm_size += size;
        } else if (ti.name.find("blk.0.") == 0) {
            layer_size += size;
        }
    }

    fixed.out_norm = out_norm_size;
    fixed.embeddings = tied ? embd_size : 0;
    if (tied) out_resident_size = 0;

    // We will search for the best configuration.
    // Dimensions to search:
    // ctx: [req.ctx_req] if explicit, otherwise try down to 128
    // stream_lm_head: [req.stream_head_req, true]
    // n_buffers: [req.n_buffers_req, 1] if async, else [1]
    
    std::vector<int> ctx_candidates;
    if (req.ctx_explicit) {
        ctx_candidates.push_back(req.ctx_req);
    } else {
        int start_ctx = req.ctx_req > 0 ? req.ctx_req : (cfg.ctx_len > 0 ? std::min<int>((int)cfg.ctx_len, req.default_ctx_cap) : 2048);
        for (int c = start_ctx; c >= 128; c -= 128) {
            ctx_candidates.push_back(c);
        }
    }

    struct Candidate {
        int ctx;
        bool stream_lm_head;
        int n_buffers;
        int n_pinned;
        size_t total_rss;
        MemLedger ledger;
        int score() const {
            // Score priorities:
            // 1. higher ctx is better (up to requested)
            // 2. resident head > streamed head
            // 3. n_pinned is higher
            // 4. n_buffers 2 > 1
            int s = 0;
            s += ctx * 10000;
            if (n_pinned == 0 && n_buffers == 0) s += 500000; // All layers pinned
            else s += n_pinned * 1000;
            if (!stream_lm_head) s += 5000;
            if (n_buffers > 1) s += 100;
            return s;
        }
    };

    Candidate best = {-1, false, 0, 0, SIZE_MAX, fixed};
    bool found = false;

    for (int ctx : ctx_candidates) {
        size_t attn = (size_t)ctx * sizeof(float);
        size_t resid = (size_t)ctx * cfg.dim * sizeof(float);
        
        size_t kv_row_bytes = req.kv_precision == KVPrecision::Q8_0 ? type_nbytes(DType::Q8_0, cfg.kv_dim()) : cfg.kv_dim() * sizeof(float);
        size_t kv = (size_t)cfg.n_layers * kv_row_bytes * ctx * 2;

        std::vector<bool> stream_head_opts = {req.stream_head_req};
        if (!req.stream_head_req && !tied) stream_head_opts.push_back(true);

        std::vector<int> buf_opts = {1};
        if (req.async_req) buf_opts.insert(buf_opts.begin(), req.n_buffers_req);

        for (bool stream_head : stream_head_opts) {
            size_t head_cost = stream_head ? out_streamed_size : out_resident_size;
            size_t base_cost = fixed.total() + attn + resid + kv + head_cost;

            // Try to pin all layers
            if (base_cost + cfg.n_layers * layer_size <= req.budget_bytes) {
                Candidate c;
                c.ctx = ctx; c.stream_lm_head = stream_head; c.n_buffers = 0; c.n_pinned = cfg.n_layers;
                c.ledger = fixed; c.ledger.attn = attn; c.ledger.resid = resid; c.ledger.kv = kv;
                c.ledger.lm_head = head_cost; c.ledger.ring = 0; c.ledger.pinned = cfg.n_layers * layer_size;
                c.total_rss = c.ledger.total();
                if (!found || c.score() > best.score()) { best = c; found = true; }
            }

            // Try streaming layers
            for (int bufs : buf_opts) {
                size_t ring_cost = bufs * layer_size;
                if (base_cost + ring_cost > req.budget_bytes) continue; // Cannot even stream 0 pinned
                
                int max_pinned = (req.budget_bytes - base_cost - ring_cost) / layer_size;
                max_pinned = std::min<int>(max_pinned, (int)cfg.n_layers - 1);
                
                Candidate c;
                c.ctx = ctx; c.stream_lm_head = stream_head; c.n_buffers = bufs; c.n_pinned = max_pinned;
                c.ledger = fixed; c.ledger.attn = attn; c.ledger.resid = resid; c.ledger.kv = kv;
                c.ledger.lm_head = head_cost; c.ledger.ring = ring_cost; c.ledger.pinned = max_pinned * layer_size;
                c.total_rss = c.ledger.total();
                if (!found || c.score() > best.score()) { best = c; found = true; }
            }
        }
    }

    if (!found) {
        // Impossible to fit even the minimum configuration.
        int min_ctx = ctx_candidates.back();
        size_t min_attn = (size_t)min_ctx * sizeof(float);
        size_t min_resid = (size_t)min_ctx * cfg.dim * sizeof(float);
        size_t min_kv = (size_t)cfg.n_layers * cfg.kv_dim() * min_ctx * 2 * sizeof(float);
        size_t min_head = tied ? 0 : out_streamed_size;
        size_t min_ring = layer_size; // 1 buffer
        
        plan.feasible = false;
        plan.overridden = req.force;
        plan.ledger = fixed;
        plan.ledger.attn = min_attn;
        plan.ledger.resid = min_resid;
        plan.ledger.kv = min_kv;
        plan.ledger.lm_head = min_head;
        plan.ledger.ring = min_ring;
        plan.ledger.pinned = 0;
        
        std::ostringstream oss;
        oss << "RAM budget impossible.\n"
            << "Requested: " << human_bytes(req.budget_bytes) << "\n"
            << "Minimum Required: " << human_bytes(plan.ledger.total()) << "\n"
            << "Breakdown of minimum requirements:\n"
            << "  Base Slop:   " << human_bytes(plan.ledger.base) << "\n"
            << "  Scratch:     " << human_bytes(plan.ledger.scratch) << "\n"
            << "  Logits:      " << human_bytes(plan.ledger.logits) << "\n"
            << "  KV Cache:    " << human_bytes(plan.ledger.kv) << " (ctx=" << min_ctx << ")\n"
            << "  Resid+Attn:  " << human_bytes(plan.ledger.resid + plan.ledger.attn) << "\n"
            << "  Norm+Embd:   " << human_bytes(plan.ledger.out_norm + plan.ledger.embeddings) << "\n"
            << "  LM Head:     " << human_bytes(plan.ledger.lm_head) << "\n"
            << "  Ring Buf:    " << human_bytes(plan.ledger.ring) << " (1 buffer)\n";
        plan.report = oss.str();
        
        // If overridden, just set safe defaults
        plan.ctx = min_ctx;
        plan.stream_lm_head = true;
        plan.n_buffers = 1;
        plan.n_pinned_est = 0;
        plan.weight_ceiling = fixed.out_norm + fixed.embeddings + min_head + min_ring;
    } else {
        plan.feasible = true;
        plan.ctx = best.ctx;
        int req_ctx = req.ctx_req > 0 ? req.ctx_req : (cfg.ctx_len > 0 ? std::min<int>((int)cfg.ctx_len, req.default_ctx_cap) : 2048);
        plan.ctx_capped = (best.ctx < req_ctx);
        plan.stream_lm_head = best.stream_lm_head;
        plan.n_buffers = best.n_buffers == 0 ? 1 : best.n_buffers; // if 0, we pinned all, but LayerLoader expects at least 1 buffer in opt_.n_buffers
        plan.n_pinned_est = best.n_pinned;
        plan.n_layers = cfg.n_layers;
        plan.ledger = best.ledger;
        plan.weight_ceiling = req.budget_bytes - fixed.base - best.ledger.scratch - best.ledger.logits - best.ledger.attn - best.ledger.resid - best.ledger.kv; 
        
        std::ostringstream oss;
        oss << "Execution Plan\n"
            << "--------------\n"
            << "Requested RAM:      " << human_bytes(req.budget_bytes) << "\n"
            << "Mandatory memory:   " << human_bytes(fixed.base + fixed.scratch + fixed.logits + fixed.out_norm + fixed.embeddings) << "\n"
            << "Estimated Peak:     " << human_bytes(best.total_rss) << "\n"
            << "Context Window:     " << plan.ctx << (plan.ctx_capped ? " (capped)\n" : "\n")
            << "Estimated KV:       " << human_bytes(best.ledger.kv) << "\n"
            << "LM Head Mode:       " << (plan.stream_lm_head ? "streamed" : "resident") << " (" << human_bytes(best.ledger.lm_head) << ")\n"
            << "Streaming ring:     " << (best.n_buffers == 0 ? 0 : plan.n_buffers) << " buffer(s) (" << human_bytes(best.ledger.ring) << ")\n"
            << "Pinned Layers:      " << plan.n_pinned_est << " / " << plan.n_layers << " (" << human_bytes(best.ledger.pinned) << ")\n";
        plan.report = oss.str();
    }

    return plan;
}

} // namespace llm
