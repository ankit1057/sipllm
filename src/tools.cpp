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

static std::filesystem::path resolve_tool_path(const std::string& raw_path, const std::string& workdir) {
    std::filesystem::path p(raw_path);
    if (p.is_absolute()) return p;
    if (workdir.empty() || workdir == ".") return p;
    return std::filesystem::path(workdir) / p;
}

ToolDef make_read_file_tool() {
    ToolDef d;
    d.name = "read_file";
    d.description = "Read the contents of a file at the specified path safely.";
    d.params.push_back({"path", ToolParamType::String, true, "Path to the file to read", ""});
    return d;
}

ToolHandler make_read_file_handler(const std::string& workdir) {
    return [workdir](const ToolCall& call) -> std::string {
        std::string raw_path = call.get("path", call.get("file_path", ""));
        if (raw_path.empty()) {
            return "error: 'path' argument is required";
        }
        std::filesystem::path p = resolve_tool_path(raw_path, workdir);
        std::error_code ec;
        if (!std::filesystem::exists(p, ec)) {
            return "error: file not found: " + raw_path;
        }
        if (std::filesystem::is_directory(p, ec)) {
            return "error: path is a directory: " + raw_path;
        }
        uintmax_t sz = std::filesystem::file_size(p, ec);
        if (ec) {
            return "error: cannot determine file size: " + ec.message();
        }
        std::ifstream f(p, std::ios::binary);
        if (!f.is_open()) {
            return "error: failed to open file: " + raw_path;
        }
        constexpr uintmax_t MAX_READ_BYTES = 512 * 1024; // 512KB cap
        if (sz > MAX_READ_BYTES) {
            std::string buf(MAX_READ_BYTES, '\0');
            f.read(&buf[0], MAX_READ_BYTES);
            std::streamsize bytes_read = f.gcount();
            buf.resize(bytes_read);
            buf += "\n[... file truncated: read " + std::to_string(bytes_read) +
                   " of " + std::to_string(sz) + " bytes ...]";
            return buf;
        }
        std::string content((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
        return content;
    };
}

ToolDef make_write_file_tool() {
    ToolDef d;
    d.name = "write_file";
    d.description = "Write text content to a file at the specified path (creates parent directories if needed).";
    d.params.push_back({"path", ToolParamType::String, true, "Path to the file to write", ""});
    d.params.push_back({"content", ToolParamType::String, true, "Content to write into the file", ""});
    return d;
}

ToolHandler make_write_file_handler(const std::string& workdir) {
    return [workdir](const ToolCall& call) -> std::string {
        std::string raw_path = call.get("path", call.get("file_path", ""));
        if (raw_path.empty()) {
            return "error: 'path' argument is required";
        }
        std::string content = call.get("content");
        std::filesystem::path p = resolve_tool_path(raw_path, workdir);
        std::error_code ec;
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
        f.write(content.data(), content.size());
        f.close();
        if (f.fail()) {
            return "error: failed writing content to: " + raw_path;
        }
        return "ok: wrote " + std::to_string(content.size()) + " bytes to " + raw_path;
    };
}

ToolDef make_list_dir_tool() {
    ToolDef d;
    d.name = "list_dir";
    d.description = "List files and directories within a specified path.";
    d.params.push_back({"path", ToolParamType::String, false, "Path to directory (default: current working directory)", "."});
    return d;
}

ToolHandler make_list_dir_handler(const std::string& workdir) {
    return [workdir](const ToolCall& call) -> std::string {
        std::string raw_path = call.get("path", call.get("dir", call.get("directory", ".")));
        if (raw_path.empty()) raw_path = ".";
        std::filesystem::path p = resolve_tool_path(raw_path, workdir);
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
    d.description = "Execute a bash shell command and capture combined stdout/stderr.";
    d.params.push_back({"command", ToolParamType::String, true, "The command string to execute", ""});
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
        std::string result;
        if (exit_code != 0) {
            result += "[exit status: " + std::to_string(exit_code) + "]\n";
        }
        if (output.empty() && exit_code == 0) {
            result += "(command finished with no output)";
        } else {
            result += output;
        }
        if (truncated) {
            result += "\n[... output truncated at 64KB ...]";
        }
        return result;
    };
}

ToolDef make_grep_search_tool() {
    ToolDef d;
    d.name = "grep_search";
    d.description = "Search for lines matching a pattern across files in a directory or file.";
    d.params.push_back({"pattern", ToolParamType::String, true, "Substring or regex pattern to search for", ""});
    d.params.push_back({"path", ToolParamType::String, false, "Path to directory or file (default: current working directory)", "."});
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
        std::filesystem::path p = resolve_tool_path(raw_path, workdir);
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
            if (!workdir.empty() && workdir != ".") {
                std::error_code rel_ec;
                auto rel = std::filesystem::relative(fpath, workdir, rel_ec);
                if (!rel_ec) display_path = rel.string();
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
