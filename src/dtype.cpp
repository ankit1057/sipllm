// dtype.cpp — block geometry table mirroring ggml's type_traits.
#include "llm/dtype.h"
#include "llm/common.h"

namespace llm {

// GGML block constants: QK_K = 256 for K-quants, 32 for the legacy quants.
// type_size values are the exact on-disk struct sizes ggml uses, so our
// offsets into a GGUF file line up byte-for-byte.
static const TypeTraits kTraits[] = {
    /* F32  */ {"F32",  1,   4,   false},
    /* F16  */ {"F16",  1,   2,   false},
    /* Q4_0 */ {"Q4_0", 32,  18,  true},   // half d + 16B nibbles
    /* Q4_1 */ {"Q4_1", 32,  20,  true},   // half d, half m + 16B
    /* 4    */ {"?",    1,   0,   false},
    /* 5    */ {"?",    1,   0,   false},
    /* Q5_0 */ {"Q5_0", 32,  22,  true},   // half d + 4B qh + 16B
    /* Q5_1 */ {"Q5_1", 32,  24,  true},
    /* Q8_0 */ {"Q8_0", 32,  34,  true},   // half d + 32B int8
    /* Q8_1 */ {"Q8_1", 32,  36,  true},   // half d, half s + 32B
    /* Q2_K */ {"Q2_K", 256, 84,  true},
    /* Q3_K */ {"Q3_K", 256, 110, true},
    /* Q4_K */ {"Q4_K", 256, 144, true},   // half d,dmin + 12B scales + 128B
    /* Q5_K */ {"Q5_K", 256, 176, true},
    /* Q6_K */ {"Q6_K", 256, 210, true},   // 128B ql + 64B qh + 16B sc + half d
    /* Q8_K */ {"Q8_K", 256, 292, true},
    /* 16 IQ2_XXS */ {"?", 1, 0, false},
    /* 17 IQ2_XS  */ {"?", 1, 0, false},
    /* 18 IQ3_XXS */ {"?", 1, 0, false},
    /* 19 IQ1_S   */ {"?", 1, 0, false},
    /* IQ4_NL */ {"IQ4_NL", 32, 18, true},   // half d + 16B nibbles -> LUT
};

const TypeTraits& type_traits(DType t) {
    int i = static_cast<int>(t);
    if (t == DType::BF16) { static const TypeTraits bf{"BF16", 1, 2, false}; return bf; }
    LLM_CHECK(i >= 0 && i < (int)(sizeof(kTraits) / sizeof(kTraits[0])),
              "type_traits: unknown dtype code " + std::to_string(i));
    LLM_CHECK(kTraits[i].type_size != 0,
              "type_traits: unsupported dtype code " + std::to_string(i));
    return kTraits[i];
}

} // namespace llm
