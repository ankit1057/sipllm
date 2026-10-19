// semantic_cache.h — Phase 3: Token/Context Optimization Layer (Semantic LRU Cache)
//
// SemanticCache is a context cache storing KV states for previously seen token
// sequences in a Radix Tree. When a new prompt shares a prefix with a cached
// prompt, SemanticCache injects the cached KV tensors directly into the KV cache,
// bypassing the expensive prefill matrix math for those tokens.
//
// Memory usage is bounded via O(1) LRU eviction, discarding oldest leaves
// when the cache exceeds max_budget_bytes.

#pragma once

#include "llm/common.h"
#include <vector>
#include <unordered_map>
#include <list>
#include <memory>
#include <cinttypes>
#include <cstring>
#include <mutex>
#include <functional>

namespace llm {

class SemanticCache {
public:
    // Initializes the cache with a hard memory limit.
    SemanticCache(size_t max_budget_bytes, int64_t n_layers, int64_t kv_dim);

    // Queries the cache for the longest matching prefix of prompt_ids.
    // Populates out_k and out_v with the cached KV blocks.
    // Returns the number of tokens matched.
    int64_t find_longest_prefix(const std::vector<int64_t>& prompt_ids,
                                std::vector<float>& out_k,
                                std::vector<float>& out_v);

    // Commits a generated context to the cache.
    void commit(const std::vector<int64_t>& tokens,
                const std::function<const float*(int64_t layer, int64_t pos)>& k_fn,
                const std::function<const float*(int64_t layer, int64_t pos)>& v_fn);

    size_t current_bytes() const { return current_bytes_; }
    size_t max_bytes() const { return max_budget_bytes_; }

private:
    struct RadixNode {
        std::vector<int64_t> prefix;
        std::unordered_map<int64_t, std::shared_ptr<RadixNode>> children;

        // KV cache data for the tokens in this node's prefix.
        // Shape: [n_layers][prefix.size()][kv_dim]
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
