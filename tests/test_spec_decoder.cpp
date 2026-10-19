#include "llm/spec_decoder.h"
#include <cstdio>
#include <vector>
#include <cassert>

using namespace llm;

int main() {
    PromptLookupDecoder decoder(2, 5);

    // Test 1: No match
    std::vector<int64_t> ctx1 = {1, 2, 3, 4, 5, 6, 7};
    auto d1 = decoder.draft(ctx1, 3);
    assert(d1.empty());

    // Test 2: Match exists!
    // Context: [10, 11, 12, 13, 14,  99, 10, 11, 12]
    // The longest suffix n-gram with an earlier occurrence is [10, 11, 12]
    // (length 3, at index 0). Prompt lookup copies the tokens that followed
    // that earlier occurrence, up to max_draft: [13, 14, 99].
    std::vector<int64_t> ctx2 = {10, 11, 12, 13, 14, 99, 10, 11, 12};
    auto d2 = decoder.draft(ctx2, 3);
    assert(d2.size() == 3);
    assert(d2[0] == 13);
    assert(d2[1] == 14);
    assert(d2[2] == 99);

    // Test 2b: a large max_draft is clamped to the tokens actually available
    // after the match, i.e. the draft never runs past the end of the context.
    auto d2b = decoder.draft(ctx2, 10);
    assert(d2b.size() == 6);   // [13, 14, 99, 10, 11, 12]
    assert(d2b[5] == 12);

    // Test 2c: longest n-gram wins over a shorter, later one.
    // Suffix [7, 8] also occurs at index 1 but [6, 7, 8] occurs at index 0 and
    // is longer, so the draft follows the longer match: [9].
    std::vector<int64_t> ctx2c = {6, 7, 8, 9, 5, 7, 8, 1, 6, 7, 8};
    auto d2c = decoder.draft(ctx2c, 3);
    assert(d2c.size() == 3);
    assert(d2c[0] == 9);
    assert(d2c[1] == 5);
    assert(d2c[2] == 7);

    // Test 2d: context shorter than min_ngram -> no draft.
    std::vector<int64_t> ctx2d = {4};
    assert(decoder.draft(ctx2d, 3).empty());

    // Test 3: Truncated by max_draft
    auto d3 = decoder.draft(ctx2, 1);
    assert(d3.size() == 1);
    assert(d3[0] == 13);

    printf("  PASS  prompt_lookup_drafts_correctly\n");
    return 0;
}
