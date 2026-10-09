// nishachar.cpp — Nishachar Path A: the autonomous goal->plan->act->verify loop.
//
// Implements the loop foundation declared in nishachar.h over the tool-calling
// core (tools.h): render chat -> generate -> parse tool call -> dispatch handler
// -> feed result back -> repeat until a final (no-tool) answer or a step bound.
// Model-agnostic: never touches the engine (runtime.h), so it links standalone.
#include "llm/nishachar.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <exception>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace llm {

int FailedCallTracker::record(const std::string& tool, const std::string& result) {
    if (result.compare(0, 6, "error:") != 0) return 0;
    const std::string key = tool + "\n" + result.substr(0, result.find('\n'));
    for (auto& f : fails_)
        if (f.first == key) return ++f.second;
    fails_.emplace_back(key, 1);
    return 1;
}

std::string repeat_failure_note(const std::string& tool, int n) {
    std::string note = " NOTE: this is failure #" + std::to_string(n) + " of " + tool +
                       " with this same error in this run. Stop retrying it; take a different approach";
    if (tool == "edit_file")
        note += ": delete or replace lines by number instead — edit_file{path, start_line, end_line} "
                "plus first_line (the current text of start_line), using line numbers from read_file; omit new_string "
                "to delete those lines.";
    else
        note += ".";
    return note;
}

std::string degenerate_tail_unit(const std::string& text) {
    for (size_t p = 1; p <= 160; ++p) {
        const size_t need = p <= 24 ? std::max<size_t>(96, p * 10) : p * 8;
        if (text.size() < need) continue;
        const char* t = text.data() + text.size() - need;
        bool periodic = true;
        for (size_t i = p; i < need && periodic; ++i) periodic = t[i] == t[i - p];
        if (!periodic) continue;
        std::string unit(t, p);
        // A long run of one whitespace/punctuation char is legitimate (padding,
        // rules) — at any period, since "  " repeats as cleanly as " ".
        const bool one_char = unit.find_first_not_of(unit[0]) == std::string::npos;
        if (one_char && std::string(" \n-=/").find(unit[0]) != std::string::npos) continue;
        return unit;
    }
    return "";
}

static const char* kEmptyThought = "<|channel>thought\n<channel|>";

NativeStep classify_native_step(const std::string& out, bool call_closed, const std::string& rut, bool hit_cap) {
    NativeStep r;
    r.in_call = out.find("<|tool_call>") != std::string::npos;
    if (!rut.empty()) { r.kind = NativeStepKind::Rut; return r; }
    if (r.in_call) { r.kind = call_closed ? NativeStepKind::ToolCall : NativeStepKind::CutInCall; return r; }
    if (hit_cap) { r.kind = NativeStepKind::CutNoCall; return r; }
    std::string fin = out;
    const std::string ch = kEmptyThought;
    for (size_t k; (k = fin.find(ch)) != std::string::npos;) fin.erase(k, ch.size());
    const size_t a = fin.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) { r.kind = NativeStepKind::JunkFinal; return r; }
    const size_t b = fin.find_last_not_of(" \t\r\n");
    r.final_text = fin.substr(a, b - a + 1);
    r.kind = NativeStepKind::Final;
    return r;
}

// Only control-token-shaped strings are rewritten — "<|word>", "<word|>", "<|word|>",
// word = 1..24 of [A-Za-z0-9_"] — so code such as `x |> f` is left alone.
std::string strip_control_tokens(std::string s) {
    auto is_word = [](char c) { return std::isalnum((unsigned char)c) || c == '_' || c == '"'; };
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();) {
        if (s[i] == '<') {
            const bool open_form = i + 1 < s.size() && s[i + 1] == '|';   // <|word>
            size_t j = i + (open_form ? 2 : 1);
            const size_t w0 = j;
            while (j < s.size() && j - w0 < 24 && is_word(s[j])) ++j;
            if (j > w0) {
                if (open_form && j < s.size() && s[j] == '>') {                       // <|word>
                    out += "<" + s.substr(w0, j - w0) + ">";
                    i = j + 1;
                    continue;
                }
                if (open_form && j + 1 < s.size() && s[j] == '|' && s[j + 1] == '>') {  // <|word|>, e.g. <|"|>
                    out += "<" + s.substr(w0, j - w0) + ">";
                    i = j + 2;
                    continue;
                }
                if (!open_form && j + 1 < s.size() && s[j] == '|' && s[j + 1] == '>') {   // <word|>
                    out += "<" + s.substr(w0, j - w0) + ">";
                    i = j + 2;
                    continue;
                }
            }
        }
        out += s[i++];
    }
    return out;
}

std::string native_user_turn(const std::string& msg) {
    return "<turn|>\n<|turn>user\n" + strip_control_tokens(msg) + "<turn|>\n<|turn>model\n";
}

void LoopGuards::on_tool(const std::string& tool, const std::string& bash_command, bool ok, const std::string& result) {
    if (!ok || result.compare(0, 6, "error:") == 0) return;
    if (tool == "write_file" || tool == "edit_file") {
        edited_since_check_ = true;
        verify_nudged_ = false;          // new unchecked edits earn a new reminder
    } else if (tool == "bash") {
        bool is_check = cfg_.check_patterns.empty();
        for (const auto& pat : cfg_.check_patterns)
            if (bash_command.find(pat) != std::string::npos) { is_check = true; break; }
        if (is_check) edited_since_check_ = false;
    }
}

std::string LoopGuards::on_no_call(const NativeStep& step, bool* give_up) {
    if (give_up) *give_up = false;
    if (step.kind == NativeStepKind::CutNoCall)
        return "Your reply was cut off (token limit) with no tool call and no summary, and was discarded. "
               "Make one tool call now, or reply with a short summary if the task is done and verified.";
    if (step.kind == NativeStepKind::JunkFinal) {
        if (junk_nudges_ >= cfg_.max_junk_nudges) { if (give_up) *give_up = true; return ""; }
        ++junk_nudges_;
        return "Your last turn had no tool call and no summary. Continue the task: make the next tool call, "
               "or, if everything is done and verified, reply with a short summary.";
    }
    if (step.kind == NativeStepKind::Final && edited_since_check_ && !verify_nudged_) {
        verify_nudged_ = true;
        return "You changed files after your last check. Before finishing, run the checks the goal asks for "
               "and fix any failures; then reply with a short summary.";
    }
    return "";
}

std::string LoopGuards::steps_left_note(int steps_left) const {
    if (cfg_.steps_left_warning <= 0 || steps_left != cfg_.steps_left_warning) return "";
    return " NOTE: only " + std::to_string(steps_left) + " steps left. Run your checks now and finish "
           "(reply with a short summary once they pass).";
}

uint64_t native_step_seed(uint64_t base, int step) {
    uint64_t z = base + 0x9E3779B97F4A7C15ULL * (uint64_t)(step + 1);   // splitmix64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return z ? z : 1;
}

const char* agent_stop_name(AgentStop s) {
    switch (s) {
        case AgentStop::Done:      return "done";
        case AgentStop::MaxSteps:  return "max_steps";
        case AgentStop::ToolError: return "tool_error";
        case AgentStop::Empty:     return "empty";
        case AgentStop::Loop:      return "loop";
    }
    return "done";
}

// ============================================================================
// AgentResult
// ============================================================================
int AgentResult::tool_calls() const {
    int n = 0;
    for (const auto& s : steps)
        if (s.had_tool_call) ++n;
    return n;
}

// Collapse newlines and cap to `n` chars (with an ellipsis) so each step renders
// on a single readable line in the report.
static std::string truncate_line(const std::string& s, size_t n) {
    std::string t;
    t.reserve(s.size());
    for (char c : s) t += (c == '\n' || c == '\r') ? ' ' : c;
    if (t.size() > n) return t.substr(0, n) + "...";
    return t;
}

double AgentResult::total_s() const {
    double t = 0;
    for (const auto& s : steps) {
        if (s.gen_s > 0) t += s.gen_s;
        if (s.tool_s > 0) t += s.tool_s;
    }
    return t;
}

std::string AgentResult::report() const {
    std::ostringstream os;
    os << "goal: " << goal << "\n";
    os << "stop=" << agent_stop_name(stop) << "\n";
    os << "steps=" << steps_taken() << "\n";
    os << "tool_calls=" << tool_calls() << "\n";
    for (const auto& s : steps) {
        const std::string label = s.had_tool_call ? s.tool_name
                                : s.malformed     ? std::string("malformed")
                                : s.nudged        ? std::string("nudge")
                                                  : std::string("final");
        os << "#" << s.index << " " << label << " ok=" << (s.tool_ok ? 1 : 0);
        if (s.gen_s >= 0 || s.tool_s >= 0) {
            char buf[64];
            std::snprintf(buf, sizeof buf, " [gen %.1fs tool %.1fs]",
                          s.gen_s < 0 ? 0.0 : s.gen_s, s.tool_s < 0 ? 0.0 : s.tool_s);
            os << buf;
        }
        const bool fed_back = s.had_tool_call || s.malformed || s.nudged;
        os << " :: " << truncate_line(fed_back ? s.tool_result : s.model_output, 120)
           << "\n";
    }
    if (total_s() > 0) {
        char buf[48];
        std::snprintf(buf, sizeof buf, "wall=%.1fs\n", total_s());
        os << buf;
    }
    os << "final: " << final_text;
    return os.str();
}

// ============================================================================
// Nishachar
// ============================================================================
void Nishachar::add_tool(ToolDef def, ToolHandler handler) {
    // Store the handler first (keyed by name) before `def` is moved away; a
    // repeat name overwrites both, matching ToolRegistry semantics.
    handlers_[def.name] = std::move(handler);
    registry_.register_tool(std::move(def));
}

bool Nishachar::has_tool(const std::string& name) const {
    return registry_.find(name) != nullptr;
}

std::string Nishachar::dispatch(const ToolCall& call, bool& ok) const {
    if (!has_tool(call.name)) {
        ok = false;
        return "error: unknown tool '" + call.name + "'";
    }
    auto it = handlers_.find(call.name);
    if (it == handlers_.end()) {
        ok = false;
        return "error: unknown tool '" + call.name + "'";
    }
    try {
        std::string result = it->second(call);
        ok = true;
        return result;
    } catch (const std::exception& e) {
        ok = false;
        return std::string("error: ") + e.what();
    }
}

// ============================================================================
// Lenient tool-call detection (hand-written scanners; no regex, no JSON lib)
// ============================================================================
namespace {

const char* kHint =
    "Emit exactly <tool_call>{\"name\":...,\"arguments\":{...}}</tool_call> with "
    "valid JSON (escape newlines as \\n and quotes as \\\").";

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
}

std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(" \t\r\n") - b + 1);
}

void json_escape_into(std::string& o, const std::string& v) {
    for (char c : v) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\t': o += "\\t"; break;
            case '\r': o += "\\r"; break;
            default:   o += c;
        }
    }
}

// Balanced-object end (string-aware). npos when incomplete.
size_t match_object(const std::string& s, size_t open) {
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = open; i < s.size(); ++i) {
        const char c = s[i];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{') ++depth;
        else if (c == '}' && --depth == 0) return i + 1;
    }
    return std::string::npos;
}

// Minimal strict-ish JSON validator (recursive descent). Raw control chars in
// strings are tolerated (models emit literal newlines; the parser copes).
// Returns "" on success, else a short reason.
struct JsonCheck {
    const std::string& s;
    size_t i = 0;
    std::string err;
    explicit JsonCheck(const std::string& src) : s(src) {}
    bool fail(const std::string& m) {
        if (err.empty()) err = m + " at offset " + std::to_string(i);
        return false;
    }
    bool str() {
        ++i;  // opening quote
        for (; i < s.size(); ++i) {
            if (s[i] == '\\') { ++i; continue; }
            if (s[i] == '"') { ++i; return true; }
        }
        return fail("unterminated string");
    }
    bool value(int depth) {
        if (depth > 64) return fail("nesting too deep");
        skip_ws(s, i);
        if (i >= s.size()) return fail("unexpected end");
        const char c = s[i];
        if (c == '"') return str();
        if (c == '{') {
            ++i; skip_ws(s, i);
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            for (;;) {
                skip_ws(s, i);
                if (i >= s.size() || s[i] != '"') return fail("expected quoted key");
                if (!str()) return false;
                skip_ws(s, i);
                if (i >= s.size() || s[i] != ':') return fail("expected ':' after key");
                ++i;
                if (!value(depth + 1)) return false;
                skip_ws(s, i);
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return true; }
                return fail("expected ',' or '}' (unescaped quote inside a string?)");
            }
        }
        if (c == '[') {
            ++i; skip_ws(s, i);
            if (i < s.size() && s[i] == ']') { ++i; return true; }
            for (;;) {
                if (!value(depth + 1)) return false;
                skip_ws(s, i);
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return true; }
                return fail("expected ',' or ']'");
            }
        }
        // number / true / false / null
        const size_t b = i;
        while (i < s.size() && (std::isalnum((unsigned char)s[i]) || s[i] == '-' ||
                                s[i] == '+' || s[i] == '.'))
            ++i;
        if (i == b) return fail(std::string("unexpected character '") + c + "'");
        const std::string tok = s.substr(b, i - b);
        if (tok == "true" || tok == "false" || tok == "null") return true;
        if (std::isdigit((unsigned char)tok[0]) || tok[0] == '-') return true;
        return fail("invalid literal '" + tok + "'");
    }
};

std::string json_error(const std::string& obj) {
    JsonCheck jc(obj);
    if (!jc.value(0)) return jc.err;
    skip_ws(obj, jc.i);
    if (jc.i != obj.size()) return "trailing characters after the JSON object";
    return "";
}

// Decode the JSON string literal starting at s[i]=='"'. Advances i past it.
bool decode_string(const std::string& s, size_t& i, std::string& out) {
    out.clear();
    for (++i; i < s.size(); ++i) {
        char c = s[i];
        if (c == '"') { ++i; return true; }
        if (c != '\\') { out += c; continue; }
        if (++i >= s.size()) return false;
        c = s[i];
        switch (c) {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'u': {
                if (i + 4 >= s.size()) return false;
                unsigned cp = (unsigned)std::strtoul(s.substr(i + 1, 4).c_str(), nullptr, 16);
                i += 4;
                if (cp < 0x80) out += (char)cp;
                else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
                else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F));
                       out += (char)(0x80 | (cp & 0x3F)); }
                break;
            }
            default: out += c;  // \" \\ \/
        }
    }
    return false;
}

// Find the value of top-level key `key` in object `obj` (obj[0]=='{').
// Returns the index of the value's first char, or npos.
size_t find_top_key(const std::string& obj, const std::string& key) {
    int depth = 0;
    for (size_t i = 0; i < obj.size(); ++i) {
        const char c = obj[i];
        if (c == '{' || c == '[') { ++depth; continue; }
        if (c == '}' || c == ']') { --depth; continue; }
        if (c != '"') continue;
        size_t j = i;
        std::string k;
        if (!decode_string(obj, j, k)) return std::string::npos;
        size_t m = j;
        skip_ws(obj, m);
        if (depth == 1 && m < obj.size() && obj[m] == ':' && k == key) {
            ++m;
            skip_ws(obj, m);
            return m;
        }
        i = j - 1;
    }
    return std::string::npos;
}

// Gemma 4 native call body ("call:NAME{k:<|"|>v<|"|>,n:1}") -> canonical JSON.
bool gemma_native_to_json(const std::string& body, std::string& json, std::string& err) {
    static const std::string Q = "<|\"|>";
    std::string b = trim(body);
    if (b.compare(0, 5, "call:") == 0) b = b.substr(5);
    const size_t brace = b.find('{');
    if (brace == std::string::npos) { err = "missing '{' after the tool name"; return false; }
    const std::string name = trim(b.substr(0, brace));
    if (name.empty()) { err = "missing tool name"; return false; }
    std::string args;
    std::vector<char> stack;   // '{' or '['
    bool expect_key = false;
    size_t i = brace;
    for (; i < b.size(); ++i) {
        const char c = b[i];
        if (b.compare(i, Q.size(), Q) == 0) {
            const size_t e = b.find(Q, i + Q.size());
            if (e == std::string::npos) { err = "unterminated <|\"|> string"; return false; }
            args += '"';
            json_escape_into(args, b.substr(i + Q.size(), e - i - Q.size()));
            args += '"';
            i = e + Q.size() - 1;
            expect_key = false;
            continue;
        }
        if (c == '"') {  // a regular JSON string mixed in: copy verbatim
            size_t j = i + 1;
            for (; j < b.size(); ++j) {
                if (b[j] == '\\') { ++j; continue; }
                if (b[j] == '"') break;
            }
            if (j >= b.size()) { err = "unterminated string"; return false; }
            args += b.substr(i, j - i + 1);
            i = j;
            expect_key = false;
            continue;
        }
        if (c == '{' || c == '[') {
            stack.push_back(c); args += c; expect_key = (c == '{'); continue;
        }
        if (c == '}' || c == ']') {
            if (stack.empty()) { err = "unbalanced brackets"; return false; }
            stack.pop_back(); args += c; expect_key = false;
            if (stack.empty()) break;
            continue;
        }
        if (c == ',') { args += c; expect_key = !stack.empty() && stack.back() == '{'; continue; }
        if (expect_key && !std::isspace((unsigned char)c)) {
            const size_t colon = b.find(':', i);
            if (colon == std::string::npos) { err = "missing ':' after key"; return false; }
            args += '"';
            json_escape_into(args, trim(b.substr(i, colon - i)));
            args += "\":";
            i = colon;
            expect_key = false;
            continue;
        }
        args += c;
    }
    if (!stack.empty()) { err = "unterminated argument object"; return false; }
    json = "{\"name\":\"";
    json_escape_into(json, name);
    json += "\",\"arguments\":" + args + "}";
    return true;
}

std::string available(const ToolRegistry& reg, const std::vector<std::string>& names) {
    std::string out;
    for (const auto& n : names)
        if (reg.find(n)) out += (out.empty() ? "" : ", ") + n;
    return out;
}

// Canonicalize + validate one JSON object text into a ToolCall.
// Returns "" on success; else the reason.
std::string parse_object(const std::string& obj_in, const ToolRegistry& reg,
                         const std::vector<std::string>& tool_names, ToolCall& call) {
    const std::string obj = trim(obj_in);
    std::string err = json_error(obj);
    if (!err.empty()) return "invalid JSON: " + err;

    size_t np = find_top_key(obj, "name");
    if (np == std::string::npos || obj[np] != '"') return "missing \"name\" string";
    std::string name;
    if (!decode_string(obj, np, name)) return "bad \"name\" string";
    name = trim(name);
    if (!reg.find(name))
        return "unknown tool '" + name + "' (available: " + available(reg, tool_names) + ")";

    std::string args = "{}";
    size_t ap = std::string::npos;
    for (const char* k : {"arguments", "parameters", "args", "input"})
        if ((ap = find_top_key(obj, k)) != std::string::npos) break;
    if (ap != std::string::npos) {
        if (obj[ap] == '{') {
            args = obj.substr(ap, match_object(obj, ap) - ap);
        } else if (obj[ap] == '"') {
            std::string dec;
            if (!decode_string(obj, ap, dec)) return "bad \"arguments\" string";
            dec = trim(dec);
            if (dec.empty()) dec = "{}";
            if (dec[0] != '{') return "\"arguments\" must be a JSON object";
            err = json_error(dec);
            if (!err.empty()) return "invalid JSON in string-encoded \"arguments\": " + err;
            args = dec;
        } else if (obj.compare(ap, 4, "null") != 0) {
            return "\"arguments\" must be a JSON object";
        }
    }

    std::string canon = "{\"name\":\"";
    json_escape_into(canon, name);
    canon += "\",\"arguments\":" + args + "}";

    ToolParser parser(reg);
    parser.feed(canon);
    if (parser.state() != ToolParser::State::Done) return "could not parse the call";
    call = parser.parsed_call();
    call.raw_json = canon;

    const ToolDef* def = reg.find(name);
    for (const auto& p : def->params)
        if (p.required && p.default_value.empty() && !call.has(p.name))
            return "missing required argument '" + p.name + "' for tool '" + name + "'";
    return "";
}

}  // namespace

AgentCallParse parse_agent_tool_call(const std::string& text, const ToolRegistry& reg) {
    AgentCallParse r;
    // Tool names for the "available:" hint. ToolRegistry has no iterator, so
    // scan its schema block for identifiers that are registered tool names.
    std::vector<std::string> names;
    {
        const std::string blk = reg.system_prompt_block();
        std::string tok;
        for (char c : blk) {
            if (std::isalnum((unsigned char)c) || c == '_' || c == '-') { tok += c; continue; }
            if (!tok.empty() && reg.find(tok)) {
                bool dup = false;
                for (auto& n : names) dup |= (n == tok);
                if (!dup) names.push_back(tok);
            }
            tok.clear();
        }
    }
    auto malformed = [&](const std::string& why) {
        r.kind = AgentCallParse::Kind::Malformed;
        r.error = "error: could not parse your tool call (" + why + "). " + kHint;
        return r;
    };
    auto finish = [&](const std::string& obj) {
        std::string why = parse_object(obj, reg, names, r.call);
        if (!why.empty()) return malformed(why);
        r.kind = AgentCallParse::Kind::Call;
        return r;
    };

    // 1. Gemma 4 native: <|tool_call>call:NAME{...}<tool_call|>
    const size_t g = text.find("<|tool_call>");
    if (g != std::string::npos) {
        std::string body = text.substr(g + 12);
        const size_t e = body.find("<tool_call|>");
        if (e != std::string::npos) body = body.substr(0, e);
        std::string json, err;
        if (!gemma_native_to_json(body, json, err)) return malformed(err);
        return finish(json);
    }

    // 2. <tool_call> ... </tool_call> (JSON, possibly ```json fenced)
    const size_t m = text.find("<tool_call>");
    if (m != std::string::npos) {
        std::string inner = text.substr(m + 11);
        const size_t e = inner.find("</tool_call>");
        const bool closed = e != std::string::npos;
        if (closed) inner = inner.substr(0, e);
        const size_t open = inner.find('{');
        if (open == std::string::npos) return malformed("no JSON object after <tool_call>");
        const size_t close = match_object(inner, open);
        if (close == std::string::npos)
            return malformed(closed ? "unbalanced braces or an unterminated string"
                                    : "unterminated JSON object, missing </tool_call>"
                                      " (output may have been cut off)");
        return finish(inner.substr(open, close - open));
    }

    // 3. No marker: accept a bare JSON object naming a registered tool, else final.
    const size_t open = text.find('{');
    if (open != std::string::npos) {
        const size_t close = match_object(text, open);
        if (close != std::string::npos) {
            ToolCall c;
            if (parse_object(text.substr(open, close - open), reg, names, c).empty()) {
                r.kind = AgentCallParse::Kind::Call;
                r.call = std::move(c);
            }
        }
    }
    return r;
}

// ============================================================================
// The loop
// ============================================================================
AgentResult Nishachar::run(const std::string& goal, const AgentGenerator& gen,
                           const AgentConfig& cfg) const {
    using clock = std::chrono::steady_clock;
    auto secs = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    AgentResult result;
    result.goal = goal;
    // Default outcome: if every step calls a tool and the bound is exhausted, we
    // stopped for MaxSteps. Overridden on any Done/Empty/ToolError/Loop break.
    result.stop = AgentStop::MaxSteps;

    std::vector<ChatMessage> messages;
    // render_chat() folds the tool-schema block into the system message itself;
    // adding it here too would advertise every tool twice.
    messages.push_back({ChatMessage::Role::System, cfg.system_preamble, ""});
    messages.push_back({ChatMessage::Role::User, goal, ""});

    auto record = [&](AgentStep s) {
        result.steps.push_back(std::move(s));
        if (cfg.on_step) cfg.on_step(result.steps.back());
    };

    std::string last_sig;
    int repeat = 0;
    bool nudged = false;

    for (int step = 0; step < cfg.max_steps; ++step) {
        std::string prompt = render_chat(messages, registry_, cfg.style,
                                         /*add_gen_prompt=*/true);
        const auto t0 = clock::now();
        std::string out = gen(prompt);
        const double gen_s = secs(t0, clock::now());

        if (out.empty()) {
            result.stop = AgentStop::Empty;
            break;
        }

        AgentStep s;
        s.index = step;
        s.model_output = out;
        s.gen_s = gen_s;

        const AgentCallParse parsed = parse_agent_tool_call(out, registry_);

        if (parsed.kind == AgentCallParse::Kind::Malformed) {
            s.malformed = true;
            s.tool_ok = false;
            s.tool_result = parsed.error;
            last_sig.clear();
            repeat = 0;
            record(std::move(s));
            messages.push_back({ChatMessage::Role::Assistant, out, ""});
            messages.push_back({ChatMessage::Role::Tool, parsed.error, ""});
            continue;
        }

        if (parsed.kind == AgentCallParse::Kind::Call) {
            const ToolCall& call = parsed.call;
            const std::string sig = call.raw_json;  // canonical name + args
            repeat = (sig == last_sig) ? repeat + 1 : 1;
            last_sig = sig;

            s.had_tool_call = true;
            s.tool_name = call.name;
            s.tool_args_json = call.raw_json;

            const int lim = cfg.loop_repeat_limit;
            if (lim > 0 && repeat > lim) {
                s.tool_ok = false;
                s.tool_result = "error: repeated identical tool call; stopping (loop guard)";
                record(std::move(s));
                result.stop = AgentStop::Loop;
                break;
            }
            bool ok = true;
            std::string tool_result;
            if (lim > 0 && repeat == lim) {
                ok = false;
                tool_result = "error: you are looping -- this exact '" + call.name +
                              "' call was made " + std::to_string(repeat) +
                              " times in a row and was NOT run again. Its result will "
                              "not change; try a different action or give the final answer.";
            } else {
                const auto t1 = clock::now();
                tool_result = dispatch(call, ok);
                s.tool_s = secs(t1, clock::now());
            }
            s.tool_result = tool_result;
            s.tool_ok = ok;
            record(std::move(s));

            messages.push_back({ChatMessage::Role::Assistant, out, ""});
            messages.push_back({ChatMessage::Role::Tool, tool_result, ""});

            if (!ok && cfg.stop_on_tool_error) {
                result.stop = AgentStop::ToolError;
                break;
            }
            continue;
        }

        last_sig.clear();
        repeat = 0;

        // No tool call. Optionally nudge once if nothing has been done yet.
        if (cfg.require_tool_before_done && !nudged && result.tool_calls() == 0) {
            nudged = true;
            static const char* kNudge =
                "You have not used any tool yet; act with a tool call.";
            s.nudged = true;
            s.tool_result = kNudge;
            record(std::move(s));
            messages.push_back({ChatMessage::Role::Assistant, out, ""});
            messages.push_back({ChatMessage::Role::User, kNudge, ""});
            continue;
        }

        // Final answer.
        record(std::move(s));
        result.final_text = out;
        result.stop = AgentStop::Done;
        break;
    }

    return result;
}

} // namespace llm
