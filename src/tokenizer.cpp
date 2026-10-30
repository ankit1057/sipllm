// tokenizer.cpp — see tokenizer.h.
#include "llm/tokenizer.h"
#include "llm/common.h"
#include "unicode_ranges.h"

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
    if (model == "gemma4") {
        t.kind_ = Kind::SpmBpe;
        if (merges && merges->kind == MetaValue::Kind::StrArr)
            for (int32_t r = 0; r < (int32_t)merges->sa.size(); ++r)
                t.merge_rank_[merges->sa[r]] = r;
        // CONTROL (3) and USER_DEFINED (4) tokens are matched literally.
        for (size_t i = 0; i < t.types_.size() && i < t.id_to_tok_.size(); ++i)
            if ((t.types_[i] == 3 || t.types_[i] == 4) && !t.id_to_tok_[i].empty())
                t.specials_.push_back(t.id_to_tok_[i]);
        std::sort(t.specials_.begin(), t.specials_.end(),
                  [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    } else if (is_bpe) {
        t.kind_ = Kind::BPE;
        if (merges && merges->kind == MetaValue::Kind::StrArr)
            for (int32_t r = 0; r < (int32_t)merges->sa.size(); ++r)
                t.merge_rank_[merges->sa[r]] = r;
        build_byte_unicode(t.byte_to_unicode_, t.unicode_to_byte_);
        // Families whose split regex is reproduced exactly (checked against the
        // models' tokenizer.json); llama.cpp's aliases of them map the same way.
        // Any other value (or none, as in Phi-2) keeps the legacy splitter.
        t.pre_ = bpe_pre_from_name(src.meta_str("tokenizer.ggml.pre", ""));
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
                                     "<EOT>", "<end_of_utterance>", "</s>", "<eos>",
                                     "<turn|>"};   // Gemma 4
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

// ---- exact pre-tokenizers ------------------------------------------------
// The Hugging Face split regexes, matched by hand over codepoints:
//   Qwen2 / Llama3:
//     (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,K}
//     | ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//     with K = 1 (Qwen2) or 3 (Llama-3).
//   SmolLM: every \p{N} isolated, then GPT-2's ByteLevel regex on the rest:
//     's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
// Alternatives are tried in order at each position (leftmost-first), as the
// regex engines do.
namespace {

bool in_ranges(const uint32_t (*r)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (cp < r[mid][0]) hi = mid;
        else if (cp > r[mid][1]) lo = mid + 1;
        else return true;
    }
    return false;
}

enum class CpClass : uint8_t { Letter, Number, Space, Other };

// One codepoint and its byte span. Invalid UTF-8 bytes stand alone as Other.
struct Cp { uint32_t cp; uint32_t off; uint32_t len; CpClass cls; };

CpClass classify(uint32_t cp) {
    if (cp < 0x80) {
        if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return CpClass::Letter;
        if (cp >= '0' && cp <= '9') return CpClass::Number;
        if (cp == ' ' || (cp >= 0x09 && cp <= 0x0D)) return CpClass::Space;
        return CpClass::Other;
    }
    // White_Space property, non-ASCII part.
    if (cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
        cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000)
        return CpClass::Space;
    if (in_ranges(kUnicodeLetterRanges, sizeof kUnicodeLetterRanges / sizeof *kUnicodeLetterRanges, cp))
        return CpClass::Letter;
    if (in_ranges(kUnicodeNumberRanges, sizeof kUnicodeNumberRanges / sizeof *kUnicodeNumberRanges, cp))
        return CpClass::Number;
    return CpClass::Other;
}

std::vector<Cp> decode_utf8(const std::string& s) {
    std::vector<Cp> out;
    out.reserve(s.size());
    size_t i = 0, n = s.size();
    while (i < n) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp = c, len = 1;
        if (c >= 0xC0 && c < 0xF8) {
            len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
            cp = c & (0x7F >> len);
            bool ok = i + len <= n;
            for (uint32_t k = 1; ok && k < len; ++k) {
                const unsigned char d = (unsigned char)s[i + k];
                ok = (d & 0xC0) == 0x80;
                cp = (cp << 6) | (d & 0x3F);
            }
            if (!ok) { cp = c; len = 1; out.push_back({cp, (uint32_t)i, 1, CpClass::Other}); ++i; continue; }
        } else if (c >= 0x80) {
            out.push_back({cp, (uint32_t)i, 1, CpClass::Other}); ++i; continue;
        }
        out.push_back({cp, (uint32_t)i, len, classify(cp)});
        i += len;
    }
    return out;
}

inline bool is_nl(const Cp& c) { return c.cp == '\r' || c.cp == '\n'; }
// ASCII lowercase; 0 for anything non-ASCII, so a codepoint whose low byte
// happens to be 's' or 't' never reads as a contraction letter.
inline char lower_ascii(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return (char)(cp + 32);
    return cp < 0x80 ? (char)cp : 0;
}

// Length (in codepoints) of a contraction at i: 's 't 're 've 'm 'll 'd.
size_t contraction_len(const std::vector<Cp>& c, size_t i, bool icase) {
    if (c[i].cp != '\'' || i + 1 >= c.size()) return 0;
    auto ch = [&](size_t k) { return icase ? lower_ascii(c[k].cp) : (char)(c[k].cp < 0x80 ? c[k].cp : 0); };
    const char a = ch(i + 1);
    if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    if (i + 2 < c.size()) {
        const char b = ch(i + 2);
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
    }
    return 0;
}

// \s*[\r\n]+ | \s+(?!\S) | \s+ at a whitespace codepoint i; returns the end.
size_t whitespace_end(const std::vector<Cp>& c, size_t i, size_t n, bool newline_rule) {
    size_t j = i;
    while (j < n && c[j].cls == CpClass::Space) ++j;
    if (newline_rule)
        for (size_t k = j; k > i; --k)
            if (is_nl(c[k - 1])) return k;          // up to the last newline of the run
    if (j == n) return j;                           // (?!\S) holds at the end of text
    if (j - 1 > i) return j - 1;                    // leave one space for the next word
    return j;                                       // \s+
}

// Split codepoints [b, e) with the Qwen2 / Llama-3 regex.
void split_llama3(const std::vector<Cp>& c, size_t b, size_t e, size_t max_digits,
                  std::vector<std::pair<size_t, size_t>>& out) {
    auto is_punct = [&](size_t k) { return c[k].cls == CpClass::Other; };
    size_t i = b;
    while (i < e) {
        size_t j = i;
        if (size_t l = contraction_len(c, i, /*icase=*/true); l && i + l <= e) {
            j = i + l;
        } else if (c[i].cls == CpClass::Letter ||
                   (c[i].cls != CpClass::Number && !is_nl(c[i]) && i + 1 < e && c[i + 1].cls == CpClass::Letter)) {
            j = i + 1;
            while (j < e && c[j].cls == CpClass::Letter) ++j;
        } else if (c[i].cls == CpClass::Number) {
            while (j < e && c[j].cls == CpClass::Number && j - i < max_digits) ++j;
        } else if (is_punct(i) || (c[i].cp == ' ' && i + 1 < e && is_punct(i + 1))) {
            j = c[i].cp == ' ' ? i + 1 : i;
            while (j < e && is_punct(j)) ++j;
            while (j < e && is_nl(c[j])) ++j;
        } else {
            j = whitespace_end(c, i, e, /*newline_rule=*/true);
        }
        out.push_back({i, j});
        i = j;
    }
}

// Split codepoints [b, e) with GPT-2's ByteLevel regex.
void split_gpt2(const std::vector<Cp>& c, size_t b, size_t e,
                std::vector<std::pair<size_t, size_t>>& out) {
    size_t i = b;
    while (i < e) {
        size_t j = i;
        if (size_t l = contraction_len(c, i, /*icase=*/false); l && i + l <= e) {
            j = i + l;
        } else {
            const size_t k = (c[i].cp == ' ' && i + 1 < e && c[i + 1].cls != CpClass::Space) ? i + 1 : i;
            if (c[k].cls != CpClass::Space) {
                const CpClass cls = c[k].cls;               // ' ?\p{L}+', ' ?\p{N}+' or ' ?[^\s\p{L}\p{N}]+'
                j = k;
                while (j < e && c[j].cls == cls) ++j;
            } else {
                j = whitespace_end(c, i, e, /*newline_rule=*/false);
            }
        }
        out.push_back({i, j});
        i = j;
    }
}

} // namespace

BpePre bpe_pre_from_name(const std::string& pre) {
    if (pre == "qwen2" || pre == "deepseek-r1-qwen")                     return BpePre::Qwen2;
    if (pre == "llama-bpe" || pre == "llama3" || pre == "llama-v3")      return BpePre::Llama3;
    if (pre == "smollm")                                                 return BpePre::SmolLM;
    return BpePre::Legacy;
}

std::vector<std::string> bpe_pretokenize(const std::string& text, BpePre pre) {
    if (pre == BpePre::Legacy) return pretokenize(text);
    const std::vector<Cp> c = decode_utf8(text);
    std::vector<std::pair<size_t, size_t>> spans;
    switch (pre) {
        case BpePre::Qwen2:  split_llama3(c, 0, c.size(), 1, spans); break;
        case BpePre::Llama3: split_llama3(c, 0, c.size(), 3, spans); break;
        case BpePre::SmolLM: {
            size_t i = 0;
            while (i < c.size()) {
                if (c[i].cls == CpClass::Number) { spans.push_back({i, i + 1}); ++i; continue; }
                size_t j = i;
                while (j < c.size() && c[j].cls != CpClass::Number) ++j;
                split_gpt2(c, i, j, spans);
                i = j;
            }
            break;
        }
        case BpePre::Legacy: break;
    }
    std::vector<std::string> out;
    out.reserve(spans.size());
    for (const auto& sp : spans) {
        const size_t from = c[sp.first].off;
        const size_t to = sp.second < c.size() ? c[sp.second].off : text.size();
        out.push_back(text.substr(from, to - from));
    }
    return out;
}

std::vector<int64_t> Tokenizer::encode_bpe(const std::string& text) const {
    std::vector<int64_t> out;
    for (const std::string& chunk : bpe_pretokenize(text, pre_)) {
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

// ---- Gemma 4 SPM-style BPE ------------------------------------------------
// Mirrors llama.cpp's LLAMA_VOCAB_PRE_TYPE_GEMMA4: special tokens are split out
// first; in the remaining text ' ' -> '▁', then the text is cut into runs of
// non-newlines and runs of newlines, and each run is merged by BPE rank over
// UTF-8 characters. Symbols missing from the vocab fall back to <0xNN> bytes.
void Tokenizer::spm_bpe_chunk(const std::string& chunk, std::vector<int64_t>& out) const {
    if (chunk.empty()) return;
    if (chunk.find_first_not_of('\n') == std::string::npos) {
        // A pure newline run is a single vocab token when one exists.
        if (auto it = tok_to_id_.find(chunk); it != tok_to_id_.end()) { out.push_back(it->second); return; }
    }
    std::vector<std::string> syms;
    for (size_t i = 0; i < chunk.size();) {
        unsigned char c = chunk[i];
        size_t len = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 1;
        len = std::min(len, chunk.size() - i);
        syms.push_back(chunk.substr(i, len));
        i += len;
    }
    // Merge the lowest-rank adjacent pair until none applies. Ties resolve to
    // the leftmost pair, as llama.cpp's bigram queue does.
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
    for (const std::string& sym : syms) {
        if (auto it = tok_to_id_.find(sym); it != tok_to_id_.end()) { out.push_back(it->second); continue; }
        for (unsigned char b : sym) {
            char buf[8]; snprintf(buf, sizeof(buf), "<0x%02X>", b);
            if (auto it = tok_to_id_.find(buf); it != tok_to_id_.end()) out.push_back(it->second);
        }
    }
}

std::vector<int64_t> Tokenizer::encode_spm_bpe(const std::string& text) const {
    std::vector<int64_t> out;
    auto encode_raw = [&](const std::string& raw) {
        std::string esc;
        esc.reserve(raw.size() + raw.size() / 2);
        for (char c : raw) { if (c == ' ') esc += "\xE2\x96\x81"; else esc += c; }
        for (size_t i = 0; i < esc.size();) {
            const bool nl = esc[i] == '\n';
            size_t j = i;
            while (j < esc.size() && (esc[j] == '\n') == nl) ++j;
            spm_bpe_chunk(esc.substr(i, j - i), out);
            i = j;
        }
    };
    size_t start = 0, i = 0;
    while (i < text.size()) {
        const std::string* hit = nullptr;
        if (text[i] == '<' || text[i] == '[')
            for (const std::string& sp : specials_)
                if (text.compare(i, sp.size(), sp) == 0) { hit = &sp; break; }
        if (!hit) { ++i; continue; }
        encode_raw(text.substr(start, i - start));
        out.push_back(tok_to_id_.at(*hit));
        i += hit->size();
        start = i;
    }
    encode_raw(text.substr(start));
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
        case Kind::SpmBpe:        return encode_spm_bpe(text);
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
    if (kind_ == Kind::SentencePiece || kind_ == Kind::SpmBpe) {
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
