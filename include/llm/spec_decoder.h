#pragma once

#include <vector>
#include <cstdint>
#include <cstddef>

namespace llm {

class SpecDecoder {
public:
    virtual ~SpecDecoder() = default;
    
    // Given the context of committed tokens, generate up to max_draft tokens.
    // Returns an empty vector if no draft can be confidently generated.
    virtual std::vector<int64_t> draft(const std::vector<int64_t>& context, int max_draft) = 0;
};

// Prompt Lookup Decoding: finds the longest n-gram match in the context
// and drafts the tokens that followed it. Extremely fast, zero-memory heuristic.
class PromptLookupDecoder : public SpecDecoder {
public:
    PromptLookupDecoder(int min_ngram = 2, int max_ngram = 5) 
        : min_ngram_(min_ngram), max_ngram_(max_ngram) {}

    std::vector<int64_t> draft(const std::vector<int64_t>& context, int max_draft) override;

private:
    int min_ngram_;
    int max_ngram_;
};

} // namespace llm
