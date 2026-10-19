#include "llm/spec_decoder.h"
#include <algorithm>

namespace llm {

std::vector<int64_t> PromptLookupDecoder::draft(const std::vector<int64_t>& context, int max_draft) {
    if (context.size() < (size_t)min_ngram_) return {};

    int64_t n = (int64_t)context.size();
    
    // Search for the longest matching n-gram from the end of the context
    // starting with max_ngram_ and backing down to min_ngram_.
    for (int ngram = std::min((int)max_ngram_, (int)n - 1); ngram >= min_ngram_; --ngram) {
        // The pattern we want to match (the last `ngram` tokens)
        auto pattern_start = context.end() - ngram;
        auto pattern_end = context.end();
        
        // We search in the context up to the start of the pattern itself
        auto search_end = context.end() - ngram;
        
        // Find the last occurrence of the pattern in the context
        auto it = std::find_end(context.begin(), search_end, pattern_start, pattern_end);
        
        if (it != search_end) {
            // Found a match! The draft is the tokens immediately following this match.
            auto draft_start = it + ngram;
            int available = std::distance(draft_start, context.end());
            int draft_len = std::min(max_draft, available);
            
            if (draft_len > 0) {
                return std::vector<int64_t>(draft_start, draft_start + draft_len);
            }
        }
    }
    
    return {};
}

} // namespace llm
