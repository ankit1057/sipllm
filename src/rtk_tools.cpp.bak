// rtk_tools.cpp — RTK tool-calling + chat-template implementation.
//
// Implements the SELF-CONTAINED, zero-dependency half of rtk.h: tool registry,
// the incremental tool-call JSON parser (hand-written state machine, no JSON
// library), tool-schema rendering, chat-template rendering for every supported
// model family, and template auto-detection. Pure C++17 + the standard library.
//
// Ownership note (see AGENTS.md): this file owns the ToolRegistry / ToolParser /
// ToolDef / ToolCall / render_chat / style_from_model symbols declared in rtk.h.
// The `RTK` orchestrator class and the vision types (VisionEncoder /
// MultimodalProjector) are implemented elsewhere (src/rtk.cpp / vision) — this
// file deliberately does NOT define them, to avoid an ODR clash.
#include "llm/rtk.h"

#include <cctype>
#include <sstream>

namespace llm {

// ============================================================================
// ToolCall accessors
// ============================================================================
const std::string& ToolCall::get(const std::string& key,
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

} // namespace llm
