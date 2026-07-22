// tokenizer.h — Task 7: text <-> token ids.
//
// Three vocab kinds, all driven from GGUF tokenizer.ggml.* metadata:
//   Byte          : identity byte-level (toy models / no vocab). Exact.
//   SentencePiece : Llama-1/2 SPM — greedy score-based bigram merges, '▁' for
//                   space, <0xNN> byte fallback.
//   BPE           : Llama-3 / GPT-2 byte-level BPE — byte->unicode remap, then
//                   rank-ordered merges from the merges list.
//
// decode() is the inverse and, importantly, is safe to call incrementally on a
// single new id during streaming generation.
#pragma once

#include "llm/weight_source.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

class Tokenizer {
public:
    enum class Kind { Byte, SentencePiece, BPE };

    // Build from a model's tokenizer metadata; falls back to Byte if absent.
    static Tokenizer from_source(const WeightSource& src);
    // Plain byte-level tokenizer over `vocab` symbols (default 256).
    static Tokenizer byte_tokenizer(int64_t vocab = 256);

    std::vector<int64_t> encode(const std::string& text, bool add_bos = true) const;
    std::string decode(const std::vector<int64_t>& ids) const;
    std::string decode_token(int64_t id) const;  // one token (for streaming UI)

    Kind    kind() const { return kind_; }
    int64_t vocab_size() const { return (int64_t)id_to_tok_.size(); }
    int64_t bos_id() const { return bos_; }
    int64_t eos_id() const { return eos_; }
    bool    is_eog(int64_t id) const;             // end-of-generation?

private:
    std::vector<int64_t> encode_spm(const std::string& text) const;
    std::vector<int64_t> encode_bpe(const std::string& text) const;

    Kind kind_ = Kind::Byte;
    std::vector<std::string> id_to_tok_;
    std::unordered_map<std::string, int64_t> tok_to_id_;
    std::vector<float> scores_;                   // SPM merge scores
    std::vector<int32_t> types_;                  // token types (BYTE=6, etc.)
    std::unordered_map<std::string, int32_t> merge_rank_;  // BPE "A B" -> rank
    // byte-level BPE remap tables
    std::vector<std::string> byte_to_unicode_;    // 256 entries
    std::unordered_map<std::string, uint8_t> unicode_to_byte_;
    int64_t bos_ = -1, eos_ = -1;
    std::vector<int64_t> eog_ids_;
};

} // namespace llm
