// kosh.h — Phase 3: Token/Context Optimization Layer (Semantic LRU Cache)
//
// semantic_cache.h — Phase 3: Token/Context Optimization Layer (Semantic LRU Cache)
//
// SemanticCache is the context cache. It stores the KV cache (keys and values) for previously
// seen token sequences in a Radix Tree. When a new prompt shares a prefix with
// a cached prompt, SemanticCache injects the cached KV tensors into the runtime, bypassing
// the expensive prefill matrix math for those tokens.
//
// To bound memory usage, SemanticCache implements an O(1) LRU (Least Recently Used) eviction
// policy, discarding the oldest leaves when the cache exceeds the specified ram_budget.

#pragma once

#include "llm/common.h"
#include <vector>
#include <unordered_map>
#include <list>
#include <memory>
#include <cinttypes>
#include <cstring>
#include <mutex>

namespace llm {

class SemanticCache {
public:
    // Initializes the cache with a hard memory limit.
    SemanticCache(size_t max_budget_bytes, int64_t n_layers, int64_t kv_dim);

    // Queries the cache for the longest matching prefix of `prompt_ids`.
    // Populates `out_k` and `out_v` with the cached KV blocks, sized perfectly
    // for the match length using [layer][match_len][kv_dim] layout.
    // Returns the number of tokens matched.
    int64_t find_longest_prefix(const std::vector<int64_t>& prompt_ids,
                                std::vector<float>& out_k,
                                std::vector<float>& out_v);

    // Commits a newly generated context to the cache.
    // `k` and `v` must contain the data for the given `tokens`, stored in
    // [layer][pos][kv_dim] layout with a stride of `cap`.
    void commit(const std::vector<int64_t>& tokens,
                const float* k, const float* v,
                int64_t cap);

    size_t current_bytes() const { return current_bytes_; }
    size_t max_bytes() const { return max_budget_bytes_; }

private:
    struct RadixNode {
        std::vector<int64_t> prefix;
        std::unordered_map<int64_t, std::shared_ptr<RadixNode>> children;

        // KV cache data for the tokens in this node's prefix.
        // Shape: [n_layers][prefix.size()][kv_dim]
        // Stored flat.
        std::vector<float> k_block;
        std::vector<float> v_block;

        // LRU Tracking
        std::list<std::shared_ptr<RadixNode>>::iterator lru_it;
        bool is_in_lru = false;
        std::weak_ptr<RadixNode> parent;

        size_t bytes() const {
            return (k_block.size() + v_block.size()) * sizeof(float) +
                   prefix.size() * sizeof(int64_t);
        }
    };

    std::shared_ptr<RadixNode> root_;
    std::list<std::shared_ptr<RadixNode>> lru_queue_; // Front = newest, Back = oldest
    std::mutex mutex_;

    size_t max_budget_bytes_;
    size_t current_bytes_ = 0;
    int64_t n_layers_;
    int64_t kv_dim_;

    void mark_used(std::shared_ptr<RadixNode> node);
    void evict_until_fits();
    void remove_leaf(std::shared_ptr<RadixNode> node);
};

} // namespace llm
