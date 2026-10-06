// tools.cpp — tool-calling + chat-template implementation (see tools.h).
//
// The self-contained, zero-dependency tool registry, incremental tool-call
// JSON parser (hand-written state machine, no JSON library), tool-schema
// rendering, and chat-template rendering for every supported model family.
// Pure C++17 + the standard library.
#include "llm/tools.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace llm {

// ============================================================================
// ToolCall accessors
// ============================================================================
std::string ToolCall::get(const std::string& key,
                          const std::string& fallback) const {
    for (const ToolCallArg& a : args)
        if (a.key == key) return a.value;
    return fallback;
}
bool ToolCall::has(const std::string& key) const {
    for (const ToolCallArg& a : args)
        if (a.key == key) return true;
    return false;
}

// ============================================================================
// ToolDef / ToolRegistry
// ============================================================================
static const char* param_type_name(ToolParamType t) {
    switch (t) {
        case ToolParamType::String:      return "string";
        case ToolParamType::Int:         return "integer";
        case ToolParamType::Float:       return "number";
        case ToolParamType::Bool:        return "boolean";
        case ToolParamType::StringArray: return "array<string>";
    }
    return "string";
}

std::string ToolDef::schema_text() const {
    // A compact, self-generated function signature the model can follow. We do
    // not emit real JSON Schema (avoids a serializer dependency and is easier
    // for small models to imitate); the shape mirrors what the tool-call parser
    // below accepts: {"name": "...", "arguments": {...}}.
    std::ostringstream o;
    o << name << "(";
    for (size_t i = 0; i < params.size(); ++i) {
        const ToolParam& p = params[i];
        if (i) o << ", ";
        o << p.name << ": " << param_type_name(p.type);
        if (!p.required) o << "?";
    }
    o << ")";
    if (!description.empty()) o << " — " << description;
    for (const ToolParam& p : params) {
        if (!p.description.empty())
            o << "\n    - " << p.name << ": " << p.description
              << (p.required ? " (required)" : " (optional)");
    }
    return o.str();
}

void ToolRegistry::register_tool(ToolDef def) {
    for (ToolDef& t : tools_)
        if (t.name == def.name) { t = std::move(def); return; }
    tools_.push_back(std::move(def));
}
void ToolRegistry::unregister_tool(const std::string& name) {
    for (size_t i = 0; i < tools_.size(); ++i)
        if (tools_[i].name == name) { tools_.erase(tools_.begin() + i); return; }
}
const ToolDef* ToolRegistry::find(const std::string& name) const {
    for (const ToolDef& t : tools_)
        if (t.name == name) return &t;
    return nullptr;
}
std::string ToolRegistry::system_prompt_block() const {
    if (tools_.empty()) return "";
    std::ostringstream o;
    o << "You have access to the following tools. To call one, emit a tool "
         "call as JSON: {\"name\": \"<tool>\", \"arguments\": {<args>}}.\n\n"
         "Available tools:\n";
    for (const ToolDef& t : tools_) o << "- " << t.schema_text() << "\n";
    return o.str();
}

// ============================================================================
// ToolParser — incremental tool-call JSON state machine
// ============================================================================
ToolParser::ToolParser(const ToolRegistry& reg) : reg_(reg) {}

void ToolParser::reset() {
    state_ = State::Idle;
    current_ = ToolCall{};
    buf_.clear();
}

// Find the balanced object starting at the '{' at src[open]. Respects string
// literals and escapes so braces inside strings don't count. Returns the index
// just past the matching '}', or std::string::npos if not yet complete.
static size_t match_object(const std::string& src, size_t open) {
    int depth = 0;
    bool in_str = false, esc = false;
    for (size_t i = open; i < src.size(); ++i) {
        char c = src[i];
        if (in_str) {
            if (esc) esc = false;
            else if (c == '\\') esc = true;
            else if (c == '"') in_str = false;
            continue;
        }
        if (c == '"') in_str = true;
        else if (c == '{') ++depth;
        else if (c == '}') { if (--depth == 0) return i + 1; }
    }
    return std::string::npos;   // incomplete
}

// Extract the string value of "<key>" in src at/after `start`. Handles escaped
// quotes. Sets out to the unescaped value and end_pos past the closing quote.
bool ToolParser::extract_string(const std::string& src, const std::string& key,
                                size_t start, std::string& out,
                                size_t& end_pos) const {
    const std::string needle = "\"" + key + "\"";
    size_t k = src.find(needle, start);
    if (k == std::string::npos) return false;
    size_t colon = src.find(':', k + needle.size());
    if (colon == std::string::npos) return false;
    size_t q = src.find('"', colon);
    if (q == std::string::npos) return false;
    std::string v;
    bool esc = false;
    size_t i = q + 1;
    for (; i < src.size(); ++i) {
        char c = src[i];
        if (esc) {
            switch (c) { case 'n': v += '\n'; break; case 't': v += '\t'; break;
                         case 'r': v += '\r'; break; default: v += c; }
            esc = false;
        } else if (c == '\\') esc = true;
        else if (c == '"') { end_pos = i + 1; out = v; return true; }
        else v += c;
    }
    return false;   // unterminated
}

// Parse the "arguments" object's top-level key:value pairs into args. Values are
// captured as strings: quoted strings unescaped; numbers/bools/null verbatim;
// nested objects/arrays captured as their raw substring.
static void parse_arguments(const std::string& obj, std::vector<ToolCallArg>& out) {
    size_t i = 0;
    auto skip_ws = [&] { while (i < obj.size() && std::isspace((unsigned char)obj[i])) ++i; };
    if (i < obj.size() && obj[i] == '{') ++i;
    while (i < obj.size()) {
        skip_ws();
        if (i >= obj.size() || obj[i] == '}') break;
        if (obj[i] != '"') { ++i; continue; }
        // key
        std::string key; bool esc = false; ++i;
        for (; i < obj.size(); ++i) {
            char c = obj[i];
            if (esc) { key += c; esc = false; }
            else if (c == '\\') esc = true;
            else if (c == '"') { ++i; break; }
            else key += c;
        }
        skip_ws();
        if (i < obj.size() && obj[i] == ':') ++i;
        skip_ws();
        if (i >= obj.size()) break;
        std::string val;
        char c = obj[i];
        if (c == '"') {                              // string value
            esc = false; ++i;
            for (; i < obj.size(); ++i) {
                char d = obj[i];
                if (esc) { switch (d) { case 'n': val += '\n'; break; case 't': val += '\t'; break;
                                        case 'r': val += '\r'; break; default: val += d; } esc = false; }
                else if (d == '\\') esc = true;
                else if (d == '"') { ++i; break; }
                else val += d;
            }
        } else if (c == '{' || c == '[') {           // nested object/array: raw
            char open = c, close = (c == '{') ? '}' : ']';
            int depth = 0; bool in_str = false; esc = false;
            for (; i < obj.size(); ++i) {
                char d = obj[i]; val += d;
                if (in_str) { if (esc) esc = false; else if (d == '\\') esc = true; else if (d == '"') in_str = false; }
                else if (d == '"') in_str = true;
                else if (d == open) ++depth;
                else if (d == close) { if (--depth == 0) { ++i; break; } }
            }
        } else {                                     // number / bool / null
            for (; i < obj.size(); ++i) {
                char d = obj[i];
                if (d == ',' || d == '}' || std::isspace((unsigned char)d)) break;
                val += d;
            }
        }
        out.push_back({key, val});
        skip_ws();
        if (i < obj.size() && obj[i] == ',') ++i;
    }
}

bool ToolParser::parse_buf() {
    size_t open = buf_.find('{');
    if (open == std::string::npos) return false;
    size_t close = match_object(buf_, open);
    if (close == std::string::npos) return false;    // object not yet complete
    std::string obj = buf_.substr(open, close - open);

    ToolCall call;
    call.raw_json = obj;
    size_t ep = 0;
    if (!extract_string(obj, "name", 0, call.name, ep)) { state_ = State::Error; return false; }

    // arguments object (optional — a no-arg tool call is valid).
    size_t ak = obj.find("\"arguments\"");
    if (ak != std::string::npos) {
        size_t ao = obj.find('{', ak);
        if (ao != std::string::npos) {
            size_t ac = match_object(obj, ao);
            if (ac != std::string::npos)
                parse_arguments(obj.substr(ao, ac - ao), call.args);
        }
    }

    // Only accept calls to a registered tool (guards against the model echoing
    // JSON that isn't a real call).
    if (!reg_.find(call.name)) { state_ = State::Error; return false; }
    current_ = std::move(call);
    state_ = State::Done;
    return true;
}

bool ToolParser::feed(const std::string& text_chunk) {
    if (state_ == State::Done || state_ == State::Error) reset();
    buf_ += text_chunk;

    // Marker mode: wait for call_start, then capture until call_end (or parse
    // greedily once the object completes if call_end never arrives).
    if (!markers_.call_start.empty()) {
        size_t s = buf_.find(markers_.call_start);
        if (s == std::string::npos) { state_ = State::Scanning; return false; }
        std::string inner = buf_.substr(s + markers_.call_start.size());
        if (!markers_.call_end.empty()) {
            size_t e = inner.find(markers_.call_end);
            if (e != std::string::npos) inner = inner.substr(0, e);
            else { state_ = State::InObject; /* fall through: try partial */ }
        }
        std::string saved = buf_;
        buf_ = inner;
        bool ok = parse_buf();
        if (!ok && state_ != State::Error) buf_ = saved;   // keep accumulating
        return ok;
    }

    // Raw-JSON mode (Qwen2.5 / Mistral-Nemo): brace-match the first '{'.
    state_ = State::InObject;
    return parse_buf();
}

// ============================================================================
// Chat templates
// ============================================================================
ChatTemplateStyle style_from_model(const ModelConfig& cfg) {
    switch (cfg.arch_kind) {
        case Arch::Llama:   return ChatTemplateStyle::Llama3;
        case Arch::Mistral: return ChatTemplateStyle::Mistral;
        case Arch::Qwen2:   return ChatTemplateStyle::Qwen2;
        case Arch::Gemma2:
        case Arch::Gemma3:  return ChatTemplateStyle::Gemma;
        case Arch::Gemma4:  return ChatTemplateStyle::Gemma4;
        case Arch::Phi3:    return ChatTemplateStyle::Phi3;
        case Arch::Phi2:
        case Arch::GPT2:    return ChatTemplateStyle::GPT2;
        case Arch::Unknown:
        default:            return ChatTemplateStyle::ChatML;
    }
}

static const char* role_word(ChatMessage::Role r) {
    switch (r) {
        case ChatMessage::Role::System:    return "system";
        case ChatMessage::Role::User:      return "user";
        case ChatMessage::Role::Assistant: return "assistant";
        case ChatMessage::Role::Tool:      return "tool";
    }
    return "user";
}

std::string render_chat(const std::vector<ChatMessage>& messages,
                        const ToolRegistry& tools, ChatTemplateStyle style,
                        bool add_gen_prompt) {
    // Fold the tool schema block into the (first) system message so any template
    // carries it. If there is no system message we synthesize one.
    std::vector<ChatMessage> msgs = messages;
    std::string tool_block = tools.system_prompt_block();
    if (!tool_block.empty()) {
        bool placed = false;
        for (ChatMessage& m : msgs)
            if (m.role == ChatMessage::Role::System) {
                m.content += (m.content.empty() ? "" : "\n\n") + tool_block;
                placed = true; break;
            }
        if (!placed)
            msgs.insert(msgs.begin(), {ChatMessage::Role::System, tool_block, ""});
    }

    std::ostringstream o;
    switch (style) {
        case ChatTemplateStyle::Llama3:
            o << "<|begin_of_text|>";
            for (const ChatMessage& m : msgs)
                o << "<|start_header_id|>" << role_word(m.role) << "<|end_header_id|>\n\n"
                  << m.content << "<|eot_id|>";
            if (add_gen_prompt)
                o << "<|start_header_id|>assistant<|end_header_id|>\n\n";
            break;

        case ChatTemplateStyle::Qwen2:
        case ChatTemplateStyle::ChatML:
            for (const ChatMessage& m : msgs)
                o << "<|im_start|>" << role_word(m.role) << "\n" << m.content << "<|im_end|>\n";
            if (add_gen_prompt) o << "<|im_start|>assistant\n";
            break;

        case ChatTemplateStyle::Gemma:
            // Gemma has no system role; fold system into the first user turn.
            for (size_t i = 0; i < msgs.size(); ++i) {
                const ChatMessage& m = msgs[i];
                const char* who = (m.role == ChatMessage::Role::Assistant) ? "model" : "user";
                o << "<start_of_turn>" << who << "\n" << m.content << "<end_of_turn>\n";
            }
            if (add_gen_prompt) o << "<start_of_turn>model\n";
            break;

        case ChatTemplateStyle::Gemma4:
            // Follows the GGUF's jinja template with thinking disabled: user and
            // system content is trimmed, the model turn is "model", and the
            // generation prompt opens an empty thought channel so the reply
            // starts directly. Past model turns are rendered exactly as they
            // were generated (empty thought channel, untrimmed) rather than
            // stripped like the jinja does: the history then re-renders as a
            // verbatim extension of the tokens already in the KV cache, so a
            // multi-turn chat only ever prefills the new turn.
            for (const ChatMessage& m : msgs) {
                if (m.role == ChatMessage::Role::Assistant) {
                    o << "<|turn>model\n<|channel>thought\n<channel|>" << m.content << "<turn|>\n";
                    continue;
                }
                const char* who = m.role == ChatMessage::Role::System ? "system" : "user";
                std::string c = m.content;
                const size_t b0 = c.find_first_not_of(" \t\r\n");
                c = b0 == std::string::npos ? "" : c.substr(b0, c.find_last_not_of(" \t\r\n") - b0 + 1);
                o << "<|turn>" << who << "\n" << c << "<turn|>\n";
            }
            if (add_gen_prompt) o << "<|turn>model\n<|channel>thought\n<channel|>";
            break;

        case ChatTemplateStyle::Mistral:
            for (const ChatMessage& m : msgs) {
                if (m.role == ChatMessage::Role::Assistant) o << " " << m.content << "</s>";
                else o << "[INST] " << m.content << " [/INST]";
            }
            break;

        case ChatTemplateStyle::Phi3:
            for (const ChatMessage& m : msgs)
                o << "<|" << role_word(m.role) << "|>\n" << m.content << "<|end|>\n";
            if (add_gen_prompt) o << "<|assistant|>\n";
            break;

        case ChatTemplateStyle::GPT2:
        case ChatTemplateStyle::Raw:
            for (const ChatMessage& m : msgs)
                o << m.content << "\n";
            break;
    }
    return o.str();
}

// ============================================================================
// Production System Tools (read_file, write_file, list_dir, bash, grep_search)
// ============================================================================

// Resolve a model-supplied path against the agent's workdir and refuse anything
// that lands outside it ("../" escapes, absolute paths elsewhere, symlinks
// pointing out). weakly_canonical resolves symlinks of the existing prefix and
// normalizes the rest, so a not-yet-existing file (write_file) still resolves.
// Returns false and sets `err` (never throws).
// Component-wise prefix check (a string prefix would let /work2 pass for /work).
// Both paths must already be canonical.
static bool path_within(const std::filesystem::path& base, const std::filesystem::path& target) {
    auto bi = base.begin(), ti = target.begin();
    for (; bi != base.end(); ++bi, ++ti) {
        if (bi->empty()) continue;   // trailing-separator artifact
        if (ti == target.end() || *bi != *ti) return false;
    }
    return true;
}

static bool resolve_tool_path(const std::string& raw_path, const std::string& workdir,
                              std::filesystem::path& out, std::string& err) {
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path base = fs::absolute(workdir.empty() ? fs::path(".") : fs::path(workdir), ec);
    if (ec) { err = "error: invalid workdir: " + ec.message(); return false; }
    base = fs::weakly_canonical(base, ec);
    if (ec) { err = "error: invalid workdir: " + ec.message(); return false; }
    fs::path p(raw_path);
    fs::path target = p.is_absolute() ? p : base / p;
    target = fs::weakly_canonical(target, ec);
    if (ec) { err = "error: cannot resolve path '" + raw_path + "': " + ec.message(); return false; }
    if (!path_within(base, target)) {
        err = "error: path '" + raw_path + "' is outside the working directory "
              "(use a path relative to " + base.string() + ")";
        return false;
    }
    out = target;
    return true;
}

// Parse a model-supplied integer argument ("10", "10.0", "\"10\""); -1 if absent/bad.
static long long tool_int_arg(const ToolCall& call, const std::string& key) {
    if (!call.has(key)) return -1;
    std::string v = call.get(key);
    try { size_t used = 0; long long n = std::stoll(v, &used); return used ? n : -1; }
    catch (...) { return -1; }
}

static bool tool_bool_arg(const ToolCall& call, const std::string& key) {
    std::string v = call.get(key);
    for (char& c : v) c = (char)std::tolower((unsigned char)c);
    return v == "true" || v == "1" || v == "yes";
}

static bool has_nul_byte(const std::string& s, size_t probe = 8192) {
    return s.find('\0', 0) < std::min(probe, s.size());
}

static bool slurp_file(const std::filesystem::path& p, std::string& out) {
    std::ifstream f(p, std::ios::binary);
    if (!f.is_open()) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// 1-based line number of byte offset `pos` in `s`.
static size_t line_of(const std::string& s, size_t pos) {
    return 1 + (size_t)std::count(s.begin(), s.begin() + (std::ptrdiff_t)std::min(pos, s.size()), '\n');
}

// Lines [first, last] (1-based, inclusive) of `s`, cat -n style ("N\t...").
static std::string numbered_lines(const std::string& s, size_t first, size_t last,
                                  size_t max_line_chars = 300) {
    std::ostringstream o;
    std::istringstream in(s);
    std::string line;
    size_t n = 0;
    while (std::getline(in, line) && n < last) {
        if (++n < first) continue;
        if (line.size() > max_line_chars) line = line.substr(0, max_line_chars) + "...";
        o << n << "\t" << line << "\n";
    }
    return o.str();
}

ToolDef make_read_file_tool() {
    ToolDef d;
    d.name = "read_file";
    d.description = "Read a text file. Output lines are prefixed 'N<tab>' (line numbers, not file content).";
    d.params.push_back({"path", ToolParamType::String, true, "", ""});
    d.params.push_back({"offset", ToolParamType::Int, false, "first line to read (1-based)", ""});
    d.params.push_back({"limit", ToolParamType::Int, false, "max lines (default 200)", ""});
    return d;
}

ToolHandler make_read_file_handler(const std::string& workdir) {
    return [workdir](const ToolCall& call) -> std::string {
        std::string raw_path = call.get("path", call.get("file_path", ""));
        if (raw_path.empty()) {
            return "error: 'path' argument is required";
        }
        std::filesystem::path p;
        std::string err;
        if (!resolve_tool_path(raw_path, workdir, p, err)) return err;
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) {
            return "error: file not found: " + raw_path;
        }
        if (std::filesystem::is_directory(p, ec)) {
            return "error: path is a directory (use list_dir): " + raw_path;
        }
        std::ifstream f(p, std::ios::binary);
        if (!f.is_open()) {
            return "error: failed to open file: " + raw_path;
        }
        // Kept under the Nishachar CLI's ~6000-char tool-result cap so a page
        // is never middle-truncated and the paging note always survives.
        constexpr long long DEFAULT_LIMIT = 200;
        constexpr size_t MAX_OUT_BYTES = 5000;   // per-call output cap
        constexpr size_t MAX_LINE_CHARS = 2000;
        long long offset = tool_int_arg(call, "offset");
        long long limit = tool_int_arg(call, "limit");
        if (offset < 1) offset = 1;
        if (limit < 1) limit = DEFAULT_LIMIT;

        std::ostringstream o;
        std::string line;
        long long n = 0, last_shown = 0;
        bool byte_capped = false, checked_binary = false;
        size_t out_bytes = 0;
        while (std::getline(f, line)) {
            ++n;
            if (!checked_binary) {
                checked_binary = true;
                if (line.find('\0') != std::string::npos)
                    return "error: binary file, not shown: " + raw_path;
            }
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (n < offset || n >= offset + limit || byte_capped) continue;
            if (line.size() > MAX_LINE_CHARS)
                line = line.substr(0, MAX_LINE_CHARS) + "... [line truncated]";
            std::string row = std::to_string(n) + "\t" + line + "\n";
            if (out_bytes + row.size() > MAX_OUT_BYTES && last_shown > 0) { byte_capped = true; continue; }
            o << row;
            out_bytes += row.size();
            last_shown = n;
        }
        const long long total = n;
        if (total == 0) return "(empty file)";
        if (offset > total)
            return "error: offset " + std::to_string(offset) + " is past end of file (" +
                   std::to_string(total) + " lines)";
        std::string out = o.str();
        if (last_shown < total)
            out += "[showing lines " + std::to_string(offset) + "-" + std::to_string(last_shown) +
                   " of " + std::to_string(total) + "; use offset=" + std::to_string(last_shown + 1) +
                   " to read more]";
        return out;
    };
}

ToolDef make_write_file_tool() {
    ToolDef d;
    d.name = "write_file";
    d.description = "Create or overwrite a file (makes parent dirs). Prefer edit_file for changes to existing files.";
    d.params.push_back({"path", ToolParamType::String, true, "", ""});
    d.params.push_back({"content", ToolParamType::String, true, "", ""});
    return d;
}

static size_t count_lines(const std::string& s) {
    if (s.empty()) return 0;
    size_t n = (size_t)std::count(s.begin(), s.end(), '\n');
    return s.back() == '\n' ? n : n + 1;
}

ToolHandler make_write_file_handler(const std::string& workdir) {
    return [workdir](const ToolCall& call) -> std::string {
        std::string raw_path = call.get("path", call.get("file_path", ""));
        if (raw_path.empty()) {
            return "error: 'path' argument is required";
        }
        if (!call.has("content")) {
            return "error: 'content' argument is required";
        }
        std::string content = call.get("content");
        std::filesystem::path p;
        std::string err;
        if (!resolve_tool_path(raw_path, workdir, p, err)) return err;
        std::error_code ec;
        if (std::filesystem::is_directory(p, ec)) {
            return "error: path is a directory: " + raw_path;
        }
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path(), ec);
            if (ec) {
                return "error: failed to create parent directory: " + ec.message();
            }
        }
        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) {
            return "error: failed to open file for writing: " + raw_path;
        }
        f.write(content.data(), (std::streamsize)content.size());
        f.close();
        if (f.fail()) {
            return "error: failed writing content to: " + raw_path;
        }
        return "ok: wrote " + std::to_string(content.size()) + " bytes (" +
               std::to_string(count_lines(content)) + " lines) to " + raw_path;
    };
}

ToolDef make_edit_file_tool() {
    ToolDef d;
    d.name = "edit_file";
    d.description = "Replace exact text in a file. old_string must match exactly once (include surrounding lines to make it unique) unless replace_all=true.";
    d.params.push_back({"path", ToolParamType::String, true, "", ""});
    d.params.push_back({"old_string", ToolParamType::String, true, "", ""});
    d.params.push_back({"new_string", ToolParamType::String, true, "", ""});
    d.params.push_back({"replace_all", ToolParamType::Bool, false, "", ""});
    return d;
}

ToolHandler make_edit_file_handler(const std::string& workdir) {
    return [workdir](const ToolCall& call) -> std::string {
        std::string raw_path = call.get("path", call.get("file_path", ""));
        if (raw_path.empty()) return "error: 'path' argument is required";
        if (!call.has("old_string")) return "error: 'old_string' argument is required";
        if (!call.has("new_string")) return "error: 'new_string' argument is required";
        const std::string old_s = call.get("old_string");
        const std::string new_s = call.get("new_string");
        const bool replace_all = tool_bool_arg(call, "replace_all");
        if (old_s.empty()) return "error: old_string is empty (use write_file to create a file)";
        if (old_s == new_s) return "error: old_string and new_string are identical; nothing to do";

        std::filesystem::path p;
        std::string err;
        if (!resolve_tool_path(raw_path, workdir, p, err)) return err;
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) return "error: file not found: " + raw_path;
        if (std::filesystem::is_directory(p, ec)) return "error: path is a directory: " + raw_path;
        std::string text;
        if (!slurp_file(p, text)) return "error: failed to open file: " + raw_path;
        if (has_nul_byte(text)) return "error: binary file, cannot edit: " + raw_path;

        std::vector<size_t> hits;
        for (size_t pos = text.find(old_s); pos != std::string::npos;
             pos = text.find(old_s, pos + old_s.size()))
            hits.push_back(pos);

        if (hits.empty()) {
            std::string msg = "error: old_string not found in " + raw_path +
                              ". It must match the file exactly (whitespace, indentation; no line-number prefixes).";
            // Point at where the first meaningful line of old_string does occur.
            std::istringstream in(old_s);
            std::string first;
            while (std::getline(in, first)) {
                size_t b = first.find_first_not_of(" \t\r");
                if (b == std::string::npos) continue;
                first = first.substr(b, first.find_last_not_of(" \t\r") - b + 1);
                break;
            }
            if (!first.empty()) {
                std::vector<size_t> near;
                for (size_t pos = text.find(first); pos != std::string::npos && near.size() < 3;
                     pos = text.find(first, pos + first.size()))
                    near.push_back(line_of(text, pos));
                if (!near.empty()) {
                    msg += " Its first line occurs at:\n";
                    for (size_t ln : near)
                        msg += numbered_lines(text, ln, ln + 2);
                    msg += "Re-read those lines and copy them exactly.";
                } else {
                    msg += " Use read_file to see the current content.";
                }
            }
            return msg;
        }
        if (hits.size() > 1 && !replace_all) {
            std::string msg = "error: old_string matches " + std::to_string(hits.size()) +
                              " times in " + raw_path + "; add surrounding lines to make it unique, "
                              "or set replace_all=true. Matches at:\n";
            for (size_t i = 0; i < hits.size() && i < 5; ++i) {
                size_t ln = line_of(text, hits[i]);
                msg += numbered_lines(text, ln, ln);
            }
            if (hits.size() > 5) msg += "...\n";
            return msg;
        }

        std::string out;
        out.reserve(text.size() + hits.size() * (new_s.size() + 1));
        size_t prev = 0;
        for (size_t pos : hits) {
            out.append(text, prev, pos - prev);
            out += new_s;
            prev = pos + old_s.size();
        }
        out.append(text, prev, std::string::npos);

        std::ofstream f(p, std::ios::binary | std::ios::trunc);
        if (!f.is_open()) return "error: failed to open file for writing: " + raw_path;
        f.write(out.data(), (std::streamsize)out.size());
        f.close();
        if (f.fail()) return "error: failed writing to: " + raw_path;

        // Show the edited region (first replacement) so the model can verify
        // without another read_file round-trip.
        size_t first_ln = line_of(out, hits[0]);
        size_t new_lines = std::max<size_t>(1, count_lines(new_s));
        size_t from = first_ln > 2 ? first_ln - 2 : 1;
        size_t to = first_ln + std::min<size_t>(new_lines, 8) + 1;
        return "ok: replaced " + std::to_string(hits.size()) + " occurrence(s) in " + raw_path +
               "\n" + numbered_lines(out, from, to);
    };
}

ToolDef make_list_dir_tool() {
    ToolDef d;
    d.name = "list_dir";
    d.description = "List a directory (default: workdir).";
    d.params.push_back({"path", ToolParamType::String, false, "", "."});
    return d;
}

ToolHandler make_list_dir_handler(const std::string& workdir) {
    return [workdir](const ToolCall& call) -> std::string {
        std::string raw_path = call.get("path", call.get("dir", call.get("directory", ".")));
        if (raw_path.empty()) raw_path = ".";
        std::filesystem::path p;
        std::string err;
        if (!resolve_tool_path(raw_path, workdir, p, err)) return err;
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) {
            return "error: path not found: " + raw_path;
        }
        if (!std::filesystem::is_directory(p, ec)) {
            return "error: path is not a directory: " + raw_path;
        }
        struct Item {
            std::string name;
            bool is_dir = false;
            uintmax_t size = 0;
        };
        std::vector<Item> items;
        for (const auto& entry : std::filesystem::directory_iterator(p, std::filesystem::directory_options::skip_permission_denied, ec)) {
            Item itm;
            itm.name = entry.path().filename().string();
            itm.is_dir = entry.is_directory(ec);
            if (!itm.is_dir) {
                itm.size = entry.file_size(ec);
                if (ec) itm.size = 0;
            }
            items.push_back(itm);
        }
        std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
            if (a.is_dir != b.is_dir) return a.is_dir > b.is_dir;
            return a.name < b.name;
        });
        if (items.empty()) return "(empty directory)";
        std::ostringstream oss;
        for (const auto& itm : items) {
            if (itm.is_dir) {
                oss << "[DIR]  " << itm.name << "/\n";
            } else {
                oss << "[FILE] " << itm.name << " (" << itm.size << " bytes)\n";
            }
        }
        return oss.str();
    };
}

ToolDef make_bash_tool() {
    ToolDef d;
    d.name = "bash";
    d.description = "Run a shell command in the workdir; returns stdout+stderr and [exit code: N].";
    d.params.push_back({"command", ToolParamType::String, true, "", ""});
    return d;
}

ToolHandler make_bash_handler(const std::string& workdir) {
    return [workdir](const ToolCall& call) -> std::string {
        std::string cmd = call.get("command", call.get("cmd", ""));
        if (cmd.empty()) {
            return "error: 'command' argument is required";
        }
        std::string full_cmd;
        if (!workdir.empty() && workdir != ".") {
            full_cmd = "cd \"" + workdir + "\" 2>/dev/null && (" + cmd + ") 2>&1";
        } else {
            full_cmd = "(" + cmd + ") 2>&1";
        }
        FILE* pipe = popen(full_cmd.c_str(), "r");
        if (!pipe) {
            return "error: popen failed to start process";
        }
        std::string output;
        constexpr size_t MAX_BASH_OUTPUT = 64 * 1024; // 64KB cap
        char buffer[4096];
        bool truncated = false;
        while (true) {
            size_t bytes = fread(buffer, 1, sizeof(buffer), pipe);
            if (bytes == 0) break;
            if (output.size() + bytes > MAX_BASH_OUTPUT) {
                size_t take = MAX_BASH_OUTPUT - output.size();
                output.append(buffer, take);
                truncated = true;
                break;
            }
            output.append(buffer, bytes);
        }
        int status = pclose(pipe);
        int exit_code = 0;
#if !defined(_WIN32)
        if (WIFEXITED(status)) {
            exit_code = WEXITSTATUS(status);
        } else if (WIFSIGNALED(status)) {
            exit_code = 128 + WTERMSIG(status);
        } else {
            exit_code = status;
        }
#else
        exit_code = status;
#endif
        std::string result = output;
        if (output.empty()) result = "(no output)";
        if (truncated) {
            result += "\n[... output truncated at 64KB ...]";
        }
        if (!result.empty() && result.back() != '\n') result += "\n";
        result += "[exit code: " + std::to_string(exit_code) + "]";
        if (exit_code == 124) result += " (timed out)";
        else if (exit_code == 137) result += " (killed, likely timed out)";
        return result;
    };
}

ToolDef make_grep_search_tool() {
    ToolDef d;
    d.name = "grep_search";
    d.description = "Search files recursively for a regex (case-insensitive); returns path:line: text.";
    d.params.push_back({"pattern", ToolParamType::String, true, "", ""});
    d.params.push_back({"path", ToolParamType::String, false, "file or dir (default: workdir)", "."});
    return d;
}

static bool is_binary_file_check(const std::filesystem::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in.is_open()) return false;
    char buf[512];
    in.read(buf, sizeof(buf));
    std::streamsize n = in.gcount();
    for (std::streamsize i = 0; i < n; ++i) {
        if (buf[i] == '\0') return true;
    }
    return false;
}

ToolHandler make_grep_search_handler(const std::string& workdir) {
    return [workdir](const ToolCall& call) -> std::string {
        std::string pattern = call.get("pattern");
        if (pattern.empty()) {
            return "error: 'pattern' argument is required";
        }
        std::string raw_path = call.get("path", call.get("directory", call.get("dir", ".")));
        if (raw_path.empty()) raw_path = ".";
        std::filesystem::path p;
        std::string err;
        if (!resolve_tool_path(raw_path, workdir, p, err)) return err;
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) {
            return "error: path not found: " + raw_path;
        }

        bool use_regex = false;
        std::regex re;
        try {
            re = std::regex(pattern, std::regex_constants::ECMAScript | std::regex_constants::icase);
            use_regex = true;
        } catch (...) {
            use_regex = false;
        }

        std::vector<std::filesystem::path> files_to_search;
        if (std::filesystem::is_regular_file(p, ec)) {
            files_to_search.push_back(p);
        } else if (std::filesystem::is_directory(p, ec)) {
            std::filesystem::recursive_directory_iterator it(p, std::filesystem::directory_options::skip_permission_denied, ec);
            std::filesystem::recursive_directory_iterator end;
            while (it != end && !ec) {
                if (it->is_directory(ec)) {
                    std::string dname = it->path().filename().string();
                    if ((!dname.empty() && dname[0] == '.') || dname == "build" || dname == "node_modules") {
                        it.disable_recursion_pending();
                    }
                } else if (it->is_regular_file(ec)) {
                    std::error_code cec;
                    if (it->is_symlink(cec) &&
                        !path_within(p, std::filesystem::weakly_canonical(it->path(), cec))) {
                        it.increment(ec);
                        continue;   // symlink pointing out of the workdir
                    }
                    uintmax_t fsz = it->file_size(ec);
                    if (!ec && fsz <= 2 * 1024 * 1024) { // max 2MB per file
                        files_to_search.push_back(it->path());
                    }
                }
                it.increment(ec);
            }
        }

        std::ostringstream oss;
        int matches = 0;
        constexpr int MAX_MATCHES = 100;

        for (const auto& fpath : files_to_search) {
            if (is_binary_file_check(fpath)) continue;
            std::ifstream in(fpath);
            if (!in.is_open()) continue;
            std::string line;
            int line_no = 0;
            std::string display_path = fpath.string();
            {
                // Paths are canonical-absolute after resolve_tool_path; show
                // them relative to the workdir (shorter, and reusable as-is).
                std::error_code rel_ec;
                std::filesystem::path wd = std::filesystem::weakly_canonical(
                    std::filesystem::absolute(workdir.empty() ? "." : workdir, rel_ec), rel_ec);
                auto rel = std::filesystem::relative(fpath, wd, rel_ec);
                if (!rel_ec && !rel.empty()) display_path = rel.string();
            }
            while (std::getline(in, line)) {
                ++line_no;
                bool match = false;
                if (use_regex) {
                    try {
                        match = std::regex_search(line, re);
                    } catch (...) {
                        match = (line.find(pattern) != std::string::npos);
                    }
                } else {
                    match = (line.find(pattern) != std::string::npos);
                }
                if (match) {
                    oss << display_path << ":" << line_no << ": " << line << "\n";
                    ++matches;
                    if (matches >= MAX_MATCHES) {
                        oss << "[... maximum matches reached (" << MAX_MATCHES << ") ...]\n";
                        break;
                    }
                }
            }
            if (matches >= MAX_MATCHES) break;
        }

        if (matches == 0) {
            return "no matches found for pattern '" + pattern + "'";
        }
        return oss.str();
    };
}

void register_system_tools(std::function<void(ToolDef, ToolHandler)> registrar,
                           const std::string& workdir) {
    registrar(make_read_file_tool(), make_read_file_handler(workdir));
    registrar(make_write_file_tool(), make_write_file_handler(workdir));
    registrar(make_edit_file_tool(), make_edit_file_handler(workdir));
    registrar(make_list_dir_tool(), make_list_dir_handler(workdir));
    registrar(make_bash_tool(), make_bash_handler(workdir));
    registrar(make_grep_search_tool(), make_grep_search_handler(workdir));
}

void register_system_tools(ToolRegistry& reg,
                           std::unordered_map<std::string, ToolHandler>& handlers,
                           const std::string& workdir) {
    register_system_tools([&](ToolDef def, ToolHandler handler) {
        handlers[def.name] = handler;
        reg.register_tool(std::move(def));
    }, workdir);
}

} // namespace llm
