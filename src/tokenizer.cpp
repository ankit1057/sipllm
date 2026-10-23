// tokenizer.cpp — see tokenizer.h.
#include "llm/tokenizer.h"
#include "llm/common.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>

namespace llm {

// ---- GPT-2 byte<->unicode table ------------------------------------------
// Bytes that are printable ASCII/Latin map to themselves; the rest map to
// codepoints 256+ so every byte becomes a visible unicode char. Standard GPT-2.
static void build_byte_unicode(std::vector<std::string>& b2u,
                               std::unordered_map<std::string, uint8_t>& u2b) {
    b2u.assign(256, "");
    auto put = [&](int byte, int cp) {
        std::string s;
        if (cp < 0x80) s = std::string(1, (char)cp);
        else if (cp < 0x800) {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        } else {
            s += (char)(0xE0 | (cp >> 12));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        }
        b2u[byte] = s;
        u2b[s] = (uint8_t)byte;
    };
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                         (b >= 0xAE && b <= 0xFF);
        if (printable) put(b, b);
    }
    for (int b = 0; b < 256; ++b) {
        bool printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                         (b >= 0xAE && b <= 0xFF);
        if (!printable) put(b, 256 + n++);
    }
}

Tokenizer Tokenizer::byte_tokenizer(int64_t vocab) {
    Tokenizer t;
    t.kind_ = Kind::Byte;
    t.id_to_tok_.resize(vocab);
    for (int64_t i = 0; i < vocab; ++i) t.id_to_tok_[i] = std::string(1, (char)i);
    return t;
}

Tokenizer Tokenizer::from_source(const WeightSource& src) {
    const MetaValue* toks = src.meta("tokenizer.ggml.tokens");
    if (!toks || toks->kind != MetaValue::Kind::StrArr || toks->sa.empty())
        return byte_tokenizer();   // no vocab -> byte fallback

    Tokenizer t;
    t.id_to_tok_ = toks->sa;
    for (int64_t i = 0; i < (int64_t)t.id_to_tok_.size(); ++i)
        t.tok_to_id_[t.id_to_tok_[i]] = i;

    if (const MetaValue* sc = src.meta("tokenizer.ggml.scores"); sc && sc->kind == MetaValue::Kind::FloatArr)
        for (double v : sc->fa) t.scores_.push_back((float)v);
    if (const MetaValue* ty = src.meta("tokenizer.ggml.token_type"); ty && ty->kind == MetaValue::Kind::IntArr)
        for (int64_t v : ty->ia) t.types_.push_back((int32_t)v);

    // The declared model wins. A "llama" GGUF is SentencePiece even when it
    // also ships a merges list (used only for BPE-style byte fallback); using
    // merge-BPE there would mis-tokenize. Only "gpt2" (Llama-3 family) is BPE.
    std::string model = src.meta_str("tokenizer.ggml.model", "");
    const MetaValue* merges = src.meta("tokenizer.ggml.merges");
    bool is_bpe = (model == "gpt2") ||
                  (model.empty() && merges && merges->kind == MetaValue::Kind::StrArr &&
                   !merges->sa.empty() && t.scores_.empty());
    if (is_bpe) {
        t.kind_ = Kind::BPE;
        if (merges && merges->kind == MetaValue::Kind::StrArr)
            for (int32_t r = 0; r < (int32_t)merges->sa.size(); ++r)
                t.merge_rank_[merges->sa[r]] = r;
        build_byte_unicode(t.byte_to_unicode_, t.unicode_to_byte_);
    } else {
        t.kind_ = Kind::SentencePiece;   // "llama" / SPM with scores
    }

    t.bos_ = src.meta_int("tokenizer.ggml.bos_token_id", -1);
    t.eos_ = src.meta_int("tokenizer.ggml.eos_token_id", -1);

    // BOS policy, as llama.cpp: the GGUF flag wins; otherwise SentencePiece adds
    // it and BPE adds it only for the Llama-3 pre-tokenizer family (SmolLM2 and
    // Qwen2 do not).
    if (const int64_t f = src.meta_int("tokenizer.ggml.add_bos_token", -1); f >= 0) {
        t.add_bos_ = f != 0;
    } else if (t.kind_ == Kind::BPE) {
        static const char* bos_pre[] = {"llama3", "llama-v3", "llama-bpe", "falcon3",
                                        "falcon-h1", "pixtral"};
        const std::string pre = src.meta_str("tokenizer.ggml.pre", "");
        t.add_bos_ = std::find(std::begin(bos_pre), std::end(bos_pre), pre) != std::end(bos_pre);
    }
    t.add_space_prefix_ = src.meta_int("tokenizer.ggml.add_space_prefix", 1) != 0;

    // End-of-generation: EOS, the declared EOT/EOM ids, and the turn-end tokens
    // llama.cpp recognizes by text (Llama-3, ChatML/Qwen, Phi-3, Gemma).
    auto add_eog = [&](int64_t id) {
        if (id >= 0 && std::find(t.eog_ids_.begin(), t.eog_ids_.end(), id) == t.eog_ids_.end())
            t.eog_ids_.push_back(id);
    };
    add_eog(t.eos_);
    add_eog(src.meta_int("tokenizer.ggml.eot_token_id", -1));
    add_eog(src.meta_int("tokenizer.ggml.eom_token_id", -1));
    static const char* eog_text[] = {"<|eot_id|>", "<|eom_id|>", "<|im_end|>", "<|end|>",
                                     "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>",
                                     "<EOT>", "<end_of_utterance>", "</s>", "<eos>"};
    for (const char* e : eog_text)
        if (auto it = t.tok_to_id_.find(e); it != t.tok_to_id_.end()) add_eog(it->second);

    // Special tokens are matched as whole strings before BPE/SPM sees the text
    // (llama.cpp parse_special): token types UNKNOWN=2, CONTROL=3,
    // USER_DEFINED=4, plus the turn-end tokens above. Longest first, so a token
    // that contains a shorter one wins.
    for (int64_t id = 0; id < (int64_t)t.types_.size() && id < (int64_t)t.id_to_tok_.size(); ++id) {
        const int32_t ty = t.types_[id];
        if ((ty == 2 || ty == 3 || ty == 4) && !t.id_to_tok_[id].empty()) t.special_ids_.push_back(id);
    }
    for (int64_t id : t.eog_ids_)
        if (std::find(t.special_ids_.begin(), t.special_ids_.end(), id) == t.special_ids_.end())
            t.special_ids_.push_back(id);
    std::stable_sort(t.special_ids_.begin(), t.special_ids_.end(), [&](int64_t a, int64_t b) {
        return t.id_to_tok_[a].size() > t.id_to_tok_[b].size();
    });
    return t;
}

bool Tokenizer::is_eog(int64_t id) const {
    for (int64_t e : eog_ids_) if (id == e) return true;
    return false;
}

// ---- SentencePiece (Llama-1/2) -------------------------------------------
// Space is encoded as U+2581 '▁'. Start from single-char symbols, then greedily
// merge the adjacent pair with the highest vocab score until none improve.
std::vector<int64_t> Tokenizer::encode_spm(const std::string& text, bool space_prefix) const {
    const std::string SPACE = "\xE2\x96\x81";   // ▁
    std::string s;
    if (space_prefix) s += SPACE;               // add_dummy_prefix
    for (char c : text) s += (c == ' ') ? SPACE : std::string(1, c);

    // Split into UTF-8 symbols.
    std::vector<std::string> syms;
    for (size_t i = 0; i < s.size();) {
        int len = 1;
        unsigned char c = s[i];
        if (c >= 0xF0) len = 4; else if (c >= 0xE0) len = 3; else if (c >= 0xC0) len = 2;
        syms.push_back(s.substr(i, len));
        i += len;
    }
    // token id for a symbol, or -1
    auto tid = [&](const std::string& x) -> int64_t {
        auto it = tok_to_id_.find(x);
        return it == tok_to_id_.end() ? -1 : it->second;
    };

    for (;;) {
        float best = -1e30f; int besti = -1; int64_t bestid = -1;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            int64_t id = tid(syms[i] + syms[i + 1]);
            if (id >= 0 && id < (int64_t)scores_.size() && scores_[id] > best) {
                best = scores_[id]; besti = (int)i; bestid = id;
            }
        }
        if (besti < 0) break;
        syms[besti] = syms[besti] + syms[besti + 1];
        syms.erase(syms.begin() + besti + 1);
        (void)bestid;
    }

    std::vector<int64_t> out;
    for (auto& sym : syms) {
        int64_t id = tid(sym);
        if (id >= 0) { out.push_back(id); continue; }
        // byte fallback: <0xNN> tokens
        for (unsigned char b : sym) {
            char buf[8]; snprintf(buf, sizeof(buf), "<0x%02X>", b);
            auto it = tok_to_id_.find(buf);
            if (it != tok_to_id_.end()) out.push_back(it->second);
        }
    }
    return out;
}

// ---- byte-level BPE (Llama-3 / GPT-2) ------------------------------------
// Pragmatic pre-tokenizer over ASCII classes (letters/digits/space/punct);
// then byte->unicode remap and rank-ordered pair merges within each chunk.
static std::vector<std::string> pretokenize(const std::string& text) {
    std::vector<std::string> out;
    size_t i = 0, n = text.size();
    auto is_letter = [](unsigned char c){ return std::isalpha(c) || c >= 0x80; };
    auto is_digit  = [](unsigned char c){ return std::isdigit(c); };
    while (i < n) {
        unsigned char c = text[i];
        // contractions
        if (c == '\'' && i + 1 < n) {
            static const char* cc[] = {"'s","'t","'re","'ve","'m","'ll","'d"};
            bool matched = false;
            for (auto p : cc) {
                size_t l = std::strlen(p);
                if (text.compare(i, l, p) == 0) { out.push_back(text.substr(i, l)); i += l; matched = true; break; }
            }
            if (matched) continue;
        }
        if (c == ' ' && i + 1 < n && (is_letter(text[i+1]) || is_digit(text[i+1]))) {
            // leading space attaches to the following word/number chunk
            size_t j = i + 1;
            if (is_letter(text[j])) { while (j < n && is_letter(text[j])) ++j; }
            else { size_t cnt = 0; while (j < n && is_digit(text[j]) && cnt < 3) { ++j; ++cnt; } }
            out.push_back(text.substr(i, j - i)); i = j; continue;
        }
        if (is_letter(c)) { size_t j = i; while (j < n && is_letter(text[j])) ++j; out.push_back(text.substr(i, j - i)); i = j; continue; }
        if (is_digit(c))  { size_t j = i, cnt = 0; while (j < n && is_digit(text[j]) && cnt < 3) { ++j; ++cnt; } out.push_back(text.substr(i, j - i)); i = j; continue; }
        // whitespace run
        if (std::isspace(c)) { size_t j = i; while (j < n && std::isspace((unsigned char)text[j])) ++j; out.push_back(text.substr(i, j - i)); i = j; continue; }
        // punctuation run (optionally space-prefixed handled above)
        { size_t j = i; while (j < n && !is_letter(text[j]) && !is_digit(text[j]) && !std::isspace((unsigned char)text[j])) ++j; out.push_back(text.substr(i, j - i)); i = j; }
    }
    return out;
}

std::vector<int64_t> Tokenizer::encode_bpe(const std::string& text) const {
    std::vector<int64_t> out;
    for (const std::string& chunk : pretokenize(text)) {
        // map bytes -> unicode symbols
        std::vector<std::string> syms;
        for (unsigned char b : chunk) syms.push_back(byte_to_unicode_[b]);
        // merge by best (lowest) rank
        for (;;) {
            int best = INT32_MAX, besti = -1;
            for (size_t i = 0; i + 1 < syms.size(); ++i) {
                auto it = merge_rank_.find(syms[i] + " " + syms[i + 1]);
                if (it != merge_rank_.end() && it->second < best) { best = it->second; besti = (int)i; }
            }
            if (besti < 0) break;
            syms[besti] += syms[besti + 1];
            syms.erase(syms.begin() + besti + 1);
        }
        for (auto& s : syms) {
            auto it = tok_to_id_.find(s);
            if (it != tok_to_id_.end()) out.push_back(it->second);
            // else: unknown symbol dropped (should not happen with full vocab)
        }
    }
    return out;
}

// One stretch of plain text between special tokens. SPM gives every stretch
// the dummy prefix: each one starts the text or follows a special token, which
// is where llama.cpp adds it.
std::vector<int64_t> Tokenizer::encode_fragment(const std::string& text) const {
    switch (kind_) {
        case Kind::Byte: {
            std::vector<int64_t> body;
            for (unsigned char c : text) body.push_back(c);
            return body;
        }
        case Kind::SentencePiece: return encode_spm(text, add_space_prefix_);
        case Kind::BPE:           return encode_bpe(text);
    }
    return {};
}

std::vector<int64_t> Tokenizer::encode(const std::string& text, bool add_bos) const {
    std::vector<int64_t> ids;
    const bool bos = add_bos && add_bos_ && bos_ >= 0;
    if (bos) ids.push_back(bos_);

    // Split the text on special-token strings (longest token first): specials
    // become their id, the stretches between them are tokenized normally.
    struct Frag { int64_t id; size_t off, len; };   // id < 0: plain text
    std::vector<Frag> frags;
    if (!text.empty()) frags.push_back({-1, 0, text.size()});
    std::vector<Frag> next;
    for (int64_t sid : special_ids_) {
        const std::string& st = id_to_tok_[sid];
        if (text.find(st) == std::string::npos) continue;
        next.clear();
        for (const Frag& f : frags) {
            if (f.id >= 0) { next.push_back(f); continue; }
            size_t base = f.off;
            const size_t end = f.off + f.len;
            for (size_t m; base < end && (m = text.find(st, base)) != std::string::npos &&
                           m + st.size() <= end; base = m + st.size()) {
                if (m > base) next.push_back({-1, base, m - base});
                next.push_back({sid, m, st.size()});
            }
            if (base < end) next.push_back({-1, base, end - base});
        }
        frags.swap(next);
    }
    for (const Frag& f : frags) {
        if (f.id >= 0) { ids.push_back(f.id); continue; }
        const std::vector<int64_t> body = encode_fragment(text.substr(f.off, f.len));
        ids.insert(ids.end(), body.begin(), body.end());
    }
    // A chat template that spells out BOS itself must not get a second one.
    if (bos && ids.size() >= 2 && ids[1] == bos_) ids.erase(ids.begin() + 1);
    return ids;
}

std::string Tokenizer::decode_token(int64_t id) const {
    if (id < 0 || id >= (int64_t)id_to_tok_.size()) return "";
    if (kind_ == Kind::Byte) return id_to_tok_[id];
    const std::string& t = id_to_tok_[id];
    if (kind_ == Kind::SentencePiece) {
        // <0xNN> -> raw byte; ▁ -> space
        if (t.size() == 6 && t[0] == '<' && t[1] == '0' && t[2] == 'x') {
            int v = std::stoi(t.substr(3, 2), nullptr, 16);
            return std::string(1, (char)v);
        }
        std::string out;
        for (size_t i = 0; i < t.size();) {
            if (t.compare(i, 3, "\xE2\x96\x81") == 0) { out += ' '; i += 3; }
            else out += t[i++];
        }
        return out;
    }
    // BPE: reverse byte->unicode mapping
    std::string out;
    for (size_t i = 0; i < t.size();) {
        int len = 1; unsigned char c = t[i];
        if (c >= 0xE0) len = 3; else if (c >= 0xC0) len = 2;
        std::string sym = t.substr(i, len);
        auto it = unicode_to_byte_.find(sym);
        if (it != unicode_to_byte_.end()) out += (char)it->second;
        else out += sym;   // literal (shouldn't happen)
        i += len;
    }
    return out;
}

std::string Tokenizer::decode(const std::vector<int64_t>& ids) const {
    std::string out;
    for (int64_t id : ids) out += decode_token(id);
    return out;
}

} // namespace llm
