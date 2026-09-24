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
    // The n-gram is [10, 11, 12] (length 3). It appears at the start.
    // The following tokens are [13, 14]. So draft should be [13, 14].
    std::vector<int64_t> ctx2 = {10, 11, 12, 13, 14, 99, 10, 11, 12};
    auto d2 = decoder.draft(ctx2, 3);
    assert(d2.size() == 2);
    assert(d2[0] == 13);
    assert(d2[1] == 14);

    // Test 3: Truncated by max_draft
    auto d3 = decoder.draft(ctx2, 1);
    assert(d3.size() == 1);
    assert(d3[0] == 13);

    printf("  PASS  prompt_lookup_drafts_correctly\n");
    return 0;
}
