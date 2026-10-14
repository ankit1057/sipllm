// kosh.cpp — Phase 3: Token/Context Optimization Layer
#include "llm/semantic_cache.h"
#include <algorithm>
#include <iostream>

namespace llm {

SemanticCache::SemanticCache(size_t max_budget_bytes, int64_t n_layers, int64_t kv_dim)
    : max_budget_bytes_(max_budget_bytes), n_layers_(n_layers), kv_dim_(kv_dim) {
    root_ = std::make_shared<RadixNode>();
}

void SemanticCache::mark_used(std::shared_ptr<RadixNode> node) {
    if (node == root_) return;
    if (node->is_in_lru) {
        lru_queue_.erase(node->lru_it);
    }
    lru_queue_.push_front(node);
    node->lru_it = lru_queue_.begin();
    node->is_in_lru = true;
}

void SemanticCache::remove_leaf(std::shared_ptr<RadixNode> node) {
    if (!node->children.empty()) return; // Only remove leaves safely
    if (node->is_in_lru) {
        lru_queue_.erase(node->lru_it);
        node->is_in_lru = false;
    }
    current_bytes_ -= node->bytes();
    if (auto p = node->parent.lock()) {
        p->children.erase(node->prefix[0]);
    }
}

void SemanticCache::evict_until_fits() {
    while (current_bytes_ > max_budget_bytes_ && !lru_queue_.empty()) {
        // Find the oldest LEAF node to evict (bottom-up pruning)
        std::shared_ptr<RadixNode> oldest_leaf = nullptr;
        for (auto it = lru_queue_.rbegin(); it != lru_queue_.rend(); ++it) {
            if ((*it)->children.empty()) {
                oldest_leaf = *it;
                break;
            }
        }
        if (!oldest_leaf) {
            // If there are no leaves in the LRU queue (should be impossible in a proper tree),
            // just pop the absolute oldest to avoid infinite loop.
            oldest_leaf = lru_queue_.back();
            if (auto p = oldest_leaf->parent.lock()) p->children.erase(oldest_leaf->prefix[0]);
            oldest_leaf->children.clear(); // Will leak bytes in current_bytes_ tracking, but prevents crash
            lru_queue_.pop_back();
            break;
        }
        remove_leaf(oldest_leaf);
    }
}

int64_t SemanticCache::find_longest_prefix(const std::vector<int64_t>& prompt_ids,
                                       std::vector<float>& out_k,
                                       std::vector<float>& out_v) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int64_t matched_len = 0;
    auto curr = root_;
    int64_t i = 0;
    
    std::vector<std::pair<std::shared_ptr<RadixNode>, int64_t>> matched_nodes;

    while (i < (int64_t)prompt_ids.size()) {
        auto it = curr->children.find(prompt_ids[i]);
        if (it == curr->children.end()) {
            break;
        }
        
        auto child = it->second;
        int64_t match = 0;
        while (match < (int64_t)child->prefix.size() && 
               i + match < (int64_t)prompt_ids.size() && 
               child->prefix[match] == prompt_ids[i + match]) {
            match++;
        }
        
        matched_nodes.push_back({child, match});
        matched_len += match;
        i += match;
        mark_used(child);
        
        if (match < (int64_t)child->prefix.size()) {
            // Partial match of this edge. We stop here.
            break;
        } else {
            // Full match, continue down the tree
            curr = child;
        }
    }

    if (matched_len > 0) {
        // Allocate contiguous output buffers [layer][matched_len][kv_dim]
        size_t total_floats = (size_t)n_layers_ * matched_len * kv_dim_;
        out_k.assign(total_floats, 0.f);
        out_v.assign(total_floats, 0.f);

        // Copy data layer by layer
        int64_t current_pos = 0;
        for (auto& pair : matched_nodes) {
            auto node = pair.first;
            int64_t slice_len = pair.second;
            int64_t node_len = node->prefix.size();
            
            for (int64_t l = 0; l < n_layers_; ++l) {
                const float* src_k = node->k_block.data() + l * node_len * kv_dim_;
                const float* src_v = node->v_block.data() + l * node_len * kv_dim_;
                
                float* dst_k = out_k.data() + (l * matched_len + current_pos) * kv_dim_;
                float* dst_v = out_v.data() + (l * matched_len + current_pos) * kv_dim_;
                
                std::memcpy(dst_k, src_k, slice_len * kv_dim_ * sizeof(float));
                std::memcpy(dst_v, src_v, slice_len * kv_dim_ * sizeof(float));
            }
            current_pos += slice_len;
        }
    }

    return matched_len;
}

void SemanticCache::commit(const std::vector<int64_t>& tokens,
                       const float* k, const float* v,
                       int64_t cap) {
    if (tokens.empty() || max_budget_bytes_ == 0) return;
    std::lock_guard<std::mutex> lock(mutex_);

    auto curr = root_;
    int64_t i = 0;
    
    while (i < (int64_t)tokens.size()) {
        auto it = curr->children.find(tokens[i]);
        if (it == curr->children.end()) break;
        
        auto child = it->second;
        int64_t match = 0;
        while (match < (int64_t)child->prefix.size() && 
               i + match < (int64_t)tokens.size() && 
               child->prefix[match] == tokens[i + match]) {
            match++;
        }
        
        if (match == (int64_t)child->prefix.size()) {
            curr = child;
            i += match;
            mark_used(curr);
        } else {
            // ---- RADIX SPLIT ----
            // We matched `match` tokens of `child->prefix`.
            std::vector<int64_t> common_prefix(child->prefix.begin(), child->prefix.begin() + match);
            std::vector<int64_t> remainder_child(child->prefix.begin() + match, child->prefix.end());
            
            auto split_node = std::make_shared<RadixNode>();
            split_node->prefix = common_prefix;
            split_node->parent = curr;
            
            // 1. Slice KV Cache for split_node
            split_node->k_block.assign(n_layers_ * match * kv_dim_, 0.f);
            split_node->v_block.assign(n_layers_ * match * kv_dim_, 0.f);
            
            // 2. Slice KV Cache for child (which is now remainder_child)
            std::vector<float> new_child_k(n_layers_ * remainder_child.size() * kv_dim_, 0.f);
            std::vector<float> new_child_v(n_layers_ * remainder_child.size() * kv_dim_, 0.f);
            
            for (int64_t l = 0; l < n_layers_; ++l) {
                // To split_node
                std::memcpy(split_node->k_block.data() + l * match * kv_dim_,
                            child->k_block.data() + l * child->prefix.size() * kv_dim_,
                            match * kv_dim_ * sizeof(float));
                std::memcpy(split_node->v_block.data() + l * match * kv_dim_,
                            child->v_block.data() + l * child->prefix.size() * kv_dim_,
                            match * kv_dim_ * sizeof(float));
                
                // To child (remainder)
                std::memcpy(new_child_k.data() + l * remainder_child.size() * kv_dim_,
                            child->k_block.data() + (l * child->prefix.size() + match) * kv_dim_,
                            remainder_child.size() * kv_dim_ * sizeof(float));
                std::memcpy(new_child_v.data() + l * remainder_child.size() * kv_dim_,
                            child->v_block.data() + (l * child->prefix.size() + match) * kv_dim_,
                            remainder_child.size() * kv_dim_ * sizeof(float));
            }
            
            // Re-assign child blocks
            current_bytes_ -= child->bytes();
            child->prefix = remainder_child;
            child->k_block = std::move(new_child_k);
            child->v_block = std::move(new_child_v);
            current_bytes_ += child->bytes();
            
            // Wire up
            split_node->children[child->prefix[0]] = child;
            child->parent = split_node;
            
            curr->children[common_prefix[0]] = split_node;
            
            current_bytes_ += split_node->bytes();
            mark_used(split_node);
            
            curr = split_node;
            i += match;
            break;
        }
    }
    
    // Insert remaining tokens as a new leaf
    if (i < (int64_t)tokens.size()) {
        int64_t remainder_len = tokens.size() - i;
        auto leaf = std::make_shared<RadixNode>();
        leaf->prefix.assign(tokens.begin() + i, tokens.end());
        leaf->parent = curr;
        
        leaf->k_block.assign(n_layers_ * remainder_len * kv_dim_, 0.f);
        leaf->v_block.assign(n_layers_ * remainder_len * kv_dim_, 0.f);
        
        // Extract from the provided Runtime KV (which is [layer][pos][kv_dim] stride=cap)
        for (int64_t l = 0; l < n_layers_; ++l) {
            const float* src_k = k + (l * cap + i) * kv_dim_;
            const float* src_v = v + (l * cap + i) * kv_dim_;
            float* dst_k = leaf->k_block.data() + l * remainder_len * kv_dim_;
            float* dst_v = leaf->v_block.data() + l * remainder_len * kv_dim_;
            
            std::memcpy(dst_k, src_k, remainder_len * kv_dim_ * sizeof(float));
            std::memcpy(dst_v, src_v, remainder_len * kv_dim_ * sizeof(float));
        }
        
        curr->children[leaf->prefix[0]] = leaf;
        current_bytes_ += leaf->bytes();
        mark_used(leaf);
    }
    
    evict_until_fits();
}

} // namespace llm
