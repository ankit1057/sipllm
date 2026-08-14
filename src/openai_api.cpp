// openai_api.cpp — OpenAI-compatible /v1/chat/completions parse/build (see
// openai_api.h). Pure, zero-dependency: a small hand-written JSON parser (no
// JSON library, no <regex>) plus request parsing and response rendering.
#include "llm/openai_api.h"

#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace llm {

// ============================================================================
// Minimal zero-dependency JSON parser (objects, arrays, strings w/ escapes,
// numbers, true/false/null). Modeled on the safetensors.cpp parser; kept
// file-local so it links independently of the rest of the engine.
// ============================================================================
namespace {

struct JsonValue {
    enum class T { Null, Bool, Num, Str, Arr, Obj } type = T::Null;
    bool                                           b = false;
    double                                         num = 0;
    std::string                                    str;
    std::vector<JsonValue>                         arr;
    std::vector<std::pair<std::string, JsonValue>> obj;  // preserves order

    const JsonValue* find(const std::string& k) const {
        for (auto& kv : obj)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
    bool        is_bool() const { return type == T::Bool; }
    bool        is_num() const { return type == T::Num; }
    bool        is_str() const { return type == T::Str; }
    bool        is_arr() const { return type == T::Arr; }
    bool        is_obj() const { return type == T::Obj; }
    double      as_num(double d = 0) const { return type == T::Num ? num : d; }
    bool        as_bool(bool d = false) const { return type == T::Bool ? b : d; }
    std::string as_str(const std::string& d = "") const {
        return type == T::Str ? str : d;
    }
};

// Thrown on malformed JSON; caught by parse_chat_request to fill `err`.
struct JsonError : std::runtime_error {
    explicit JsonError(const std::string& m) : std::runtime_error(m) {}
};

class JsonParser {
public:
    explicit JsonParser(const char* p, size_t n) : p_(p), n_(n) {}
    JsonValue parse() {
        ws();
        JsonValue v = value();
        ws();
        if (i_ != n_) fail("trailing characters after value");
        return v;
    }

private:
    const char* p_;
    size_t      n_, i_ = 0;

    [[noreturn]] void fail(const char* m) {
        throw JsonError(std::string("json: ") + m);
    }
    void ws() {
        while (i_ < n_ && std::isspace((unsigned char)p_[i_])) ++i_;
    }
    char peek() { return i_ < n_ ? p_[i_] : '\0'; }
    char get() { return i_ < n_ ? p_[i_++] : '\0'; }
    bool eat(char c) {
        if (peek() == c) {
            ++i_;
            return true;
        }
        return false;
    }

    JsonValue value() {
        ws();
        char c = peek();
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') {
            JsonValue v;
            v.type = JsonValue::T::Str;
            v.str = string();
            return v;
        }
        if (c == 't' || c == 'f') return boolean();
        if (c == 'n') {
            expect("null");
            return JsonValue{};
        }
        return number();
    }
    void expect(const char* lit) {
        for (const char* q = lit; *q; ++q)
            if (get() != *q) fail("bad literal");
    }
    JsonValue boolean() {
        JsonValue v;
        v.type = JsonValue::T::Bool;
        if (peek() == 't') {
            expect("true");
            v.b = true;
        } else {
            expect("false");
            v.b = false;
        }
        return v;
    }
    JsonValue number() {
        size_t s = i_;
        if (peek() == '-' || peek() == '+') ++i_;
        while (i_ < n_ && (std::isdigit((unsigned char)p_[i_]) || p_[i_] == '.' ||
                           p_[i_] == 'e' || p_[i_] == 'E' || p_[i_] == '+' ||
                           p_[i_] == '-'))
            ++i_;
        if (i_ == s) fail("expected number");
        JsonValue v;
        v.type = JsonValue::T::Num;
        v.num = std::strtod(std::string(p_ + s, i_ - s).c_str(), nullptr);
        return v;
    }
    std::string string() {
        if (!eat('"')) fail("expected string");
        std::string out;
        while (i_ < n_) {
            char c = get();
            if (c == '"') return out;
            if (c == '\\') {
                char e = get();
                switch (e) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'b': out += '\b'; break;
                    case 'f': out += '\f'; break;
                    case '/': out += '/'; break;
                    case '"': out += '"'; break;
                    case '\\': out += '\\'; break;
                    case 'u': {  // \uXXXX -> UTF-8 (BMP only; enough here)
                        if (i_ + 4 > n_) fail("bad \\u escape");
                        int cp = (int)std::strtol(
                            std::string(p_ + i_, 4).c_str(), nullptr, 16);
                        i_ += 4;
                        if (cp < 0x80) {
                            out += (char)cp;
                        } else if (cp < 0x800) {
                            out += (char)(0xC0 | (cp >> 6));
                            out += (char)(0x80 | (cp & 0x3F));
                        } else {
                            out += (char)(0xE0 | (cp >> 12));
                            out += (char)(0x80 | ((cp >> 6) & 0x3F));
                            out += (char)(0x80 | (cp & 0x3F));
                        }
                        break;
                    }
                    default: out += e;
                }
            } else {
                out += c;
            }
        }
        fail("unterminated string");
    }
    JsonValue array() {
        JsonValue v;
        v.type = JsonValue::T::Arr;
        eat('[');
        ws();
        if (eat(']')) return v;
        for (;;) {
            v.arr.push_back(value());
            ws();
            if (eat(',')) {
                ws();
                continue;
            }
            if (eat(']')) break;
            fail("expected , or ] in array");
        }
        return v;
    }
    JsonValue object() {
        JsonValue v;
        v.type = JsonValue::T::Obj;
        eat('{');
        ws();
        if (eat('}')) return v;
        for (;;) {
            ws();
            std::string key = string();
            ws();
            if (!eat(':')) fail("expected : in object");
            v.obj.emplace_back(key, value());
            ws();
            if (eat(',')) {
                ws();
                continue;
            }
            if (eat('}')) break;
            fail("expected , or } in object");
        }
        return v;
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
ChatMessage::Role role_from_string(const std::string& s) {
    if (s == "system") return ChatMessage::Role::System;
    if (s == "user") return ChatMessage::Role::User;
    if (s == "assistant") return ChatMessage::Role::Assistant;
    if (s == "tool") return ChatMessage::Role::Tool;
    return ChatMessage::Role::User;  // unknown roles map to User
}

ToolParamType param_type_from_json(const std::string& s) {
    if (s == "integer") return ToolParamType::Int;
    if (s == "number") return ToolParamType::Float;
    if (s == "boolean") return ToolParamType::Bool;
    if (s == "array") return ToolParamType::StringArray;
    return ToolParamType::String;  // "string" and anything unknown
}

// JSON-escape a UTF-8 string for embedding in output.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\t': out += "\\t"; break;
            case '\r': out += "\\r"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    out += "\\u00";
                    out += hex[(c >> 4) & 0xF];
                    out += hex[c & 0xF];
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

// Parse a single OpenAI tools[] entry {type,function:{...}} -> ToolDef.
ToolDef parse_tool_def(const JsonValue& entry) {
    ToolDef def;
    const JsonValue* fn = entry.find("function");
    if (!fn || !fn->is_obj()) return def;  // best-effort: skip malformed

    if (const JsonValue* n = fn->find("name"); n && n->is_str())
        def.name = n->str;
    if (const JsonValue* d = fn->find("description"); d && d->is_str())
        def.description = d->str;

    const JsonValue* params = fn->find("parameters");
    if (!params || !params->is_obj()) return def;

    // Collect the set of required parameter names.
    std::vector<std::string> required;
    if (const JsonValue* req = params->find("required"); req && req->is_arr()) {
        for (const auto& r : req->arr)
            if (r.is_str()) required.push_back(r.str);
    }
    auto is_required = [&](const std::string& name) {
        for (const auto& r : required)
            if (r == name) return true;
        return false;
    };

    const JsonValue* props = params->find("properties");
    if (props && props->is_obj()) {
        for (const auto& kv : props->obj) {
            ToolParam p;
            p.name = kv.first;
            const JsonValue& spec = kv.second;
            if (spec.is_obj()) {
                if (const JsonValue* t = spec.find("type"); t && t->is_str())
                    p.type = param_type_from_json(t->str);
                if (const JsonValue* ds = spec.find("description");
                    ds && ds->is_str())
                    p.description = ds->str;
            }
            p.required = is_required(p.name);
            def.params.push_back(std::move(p));
        }
    }
    return def;
}

}  // namespace

// ============================================================================
// parse_chat_request
// ============================================================================
bool parse_chat_request(const std::string& body, ChatCompletionRequest& out,
                        std::string& err) {
    JsonValue root;
    try {
        JsonParser parser(body.data(), body.size());
        root = parser.parse();
    } catch (const std::exception& e) {
        err = std::string("invalid JSON: ") + e.what();
        return false;
    }

    if (!root.is_obj()) {
        err = "request body must be a JSON object";
        return false;
    }

    if (const JsonValue* m = root.find("model"); m && m->is_str())
        out.model = m->str;

    const JsonValue* messages = root.find("messages");
    if (!messages || !messages->is_arr() || messages->arr.empty()) {
        err = "request is missing a non-empty \"messages\" array";
        return false;
    }
    for (const auto& msg : messages->arr) {
        if (!msg.is_obj()) {
            err = "each entry of \"messages\" must be an object";
            return false;
        }
        ChatMessage cm;
        cm.role = ChatMessage::Role::User;
        if (const JsonValue* r = msg.find("role"); r && r->is_str())
            cm.role = role_from_string(r->str);
        if (const JsonValue* c = msg.find("content"); c && c->is_str())
            cm.content = c->str;
        if (const JsonValue* tid = msg.find("tool_call_id");
            tid && tid->is_str())
            cm.tool_call_id = tid->str;
        out.messages.push_back(std::move(cm));
    }

    if (const JsonValue* tools = root.find("tools"); tools && tools->is_arr()) {
        for (const auto& t : tools->arr) {
            if (t.is_obj()) out.tools.push_back(parse_tool_def(t));
        }
    }

    if (const JsonValue* mt = root.find("max_tokens"); mt && mt->is_num())
        out.max_tokens = (int)mt->num;
    if (const JsonValue* temp = root.find("temperature");
        temp && temp->is_num())
        out.temperature = (float)temp->num;
    if (const JsonValue* s = root.find("stream"); s && s->is_bool())
        out.stream = s->b;
    if (const JsonValue* a = root.find("agent"); a && a->is_bool())
        out.agent = a->b;

    return true;
}

// ============================================================================
// last_user_message
// ============================================================================
std::string last_user_message(const std::vector<ChatMessage>& messages) {
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (it->role == ChatMessage::Role::User) return it->content;
    }
    return "";
}

// ============================================================================
// build_chat_response
// ============================================================================
std::string build_chat_response(const ChatCompletionResponse& r) {
    const bool  has_tools = r.tool_calls.size() > 0;
    std::string finish = has_tools ? "tool_calls" : r.finish_reason;

    std::string out;
    out += "{\"id\":\"" + json_escape(r.id) + "\"";
    out += ",\"object\":\"chat.completion\"";
    out += ",\"created\":" + std::to_string(r.created);
    out += ",\"model\":\"" + json_escape(r.model) + "\"";
    out += ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",";

    if (has_tools) {
        out += "\"content\":null,\"tool_calls\":[";
        for (size_t i = 0; i < r.tool_calls.size(); ++i) {
            const ResponseToolCall& tc = r.tool_calls[i];
            if (i) out += ",";
            out += "{\"id\":\"" + json_escape(tc.id) + "\"";
            out += ",\"type\":\"function\"";
            out += ",\"function\":{\"name\":\"" + json_escape(tc.name) + "\"";
            out += ",\"arguments\":\"" + json_escape(tc.arguments) + "\"}}";
        }
        out += "]";
    } else {
        out += "\"content\":\"" + json_escape(r.content) + "\"";
    }

    out += "},\"finish_reason\":\"" + json_escape(finish) + "\"}]";

    const long total = (long)r.prompt_tokens + (long)r.completion_tokens;
    out += ",\"usage\":{\"prompt_tokens\":" + std::to_string(r.prompt_tokens);
    out += ",\"completion_tokens\":" + std::to_string(r.completion_tokens);
    out += ",\"total_tokens\":" + std::to_string(total) + "}}";
    return out;
}

}  // namespace llm
