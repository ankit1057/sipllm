// tools.h — tool-calling + chat-template core (the agent's "hands").
//
// Self-contained and zero-dependency: a tool registry, an incremental tool-call
// JSON parser (hand-written state machine, no JSON library), compact tool-schema
// rendering, and chat-template rendering for every supported model family with
// auto-detection. Pure C++17 + the standard library — touches no language model,
// so it links/runs independently of the engine.
//
// History: formerly the parked `src/rtk_tools.cpp` + a would-be `rtk.h`. It now
// lives in `tools.h` / `tools.cpp` so the `rtk.h` name is free for the runtime/KV
// plugin (RTK, #53). It is the foundation Nishachar (#58) uses to parse and emit
// tool calls.
#pragma once

#include "llm/model.h"

#include <string>
#include <vector>

namespace llm {

// ============================================================================
// Tool calls
// ============================================================================
struct ToolCallArg {
    std::string key;
    std::string value;
};

struct ToolCall {
    std::string raw_json;
    std::string name;
    std::vector<ToolCallArg> args;

    bool has(const std::string& key) const;
    // Returns by value: safe to call with a temporary/default `fallback`.
    std::string get(const std::string& key, const std::string& fallback = "") const;
};

// ============================================================================
// Tool definitions + registry
// ============================================================================
enum class ToolParamType {
    String,
    Int,
    Float,
    Bool,
    StringArray
};

struct ToolParam {
    std::string name;
    ToolParamType type;
    bool required = true;
    std::string description;
    std::string default_value = "";
};

struct ToolDef {
    std::string name;
    std::string description;
    std::vector<ToolParam> params;

    std::string schema_text() const;
};

class ToolRegistry {
public:
    void register_tool(ToolDef def);
    void unregister_tool(const std::string& name);
    const ToolDef* find(const std::string& name) const;

    std::string system_prompt_block() const;

    size_t size() const { return tools_.size(); }
    bool empty() const { return tools_.empty(); }

private:
    std::vector<ToolDef> tools_;
};

// ============================================================================
// Incremental tool-call parser
// ============================================================================
struct ToolParserMarkers {
    std::string call_start;
    std::string call_end;
};

class ToolParser {
public:
    enum class State { Idle, Scanning, InObject, Done, Error };

    explicit ToolParser(const ToolRegistry& reg);

    void reset();
    bool feed(const std::string& text_chunk);

    void set_markers(ToolParserMarkers m) { markers_ = std::move(m); }
    State state() const { return state_; }
    const ToolCall& parsed_call() const { return current_; }

private:
    bool parse_buf();
    bool extract_string(const std::string& src, const std::string& key,
                        size_t start, std::string& out, size_t& end_pos) const;

    const ToolRegistry& reg_;
    State state_ = State::Idle;
    ToolParserMarkers markers_;
    std::string buf_;
    ToolCall current_;
};

// ============================================================================
// Chat templates
// ============================================================================
enum class ChatTemplateStyle {
    Llama3,
    Mistral,
    Qwen2,
    Gemma,
    Phi3,
    GPT2,
    ChatML,
    Raw
};

struct ChatMessage {
    enum class Role { System, User, Assistant, Tool };
    Role role;
    std::string content;
    std::string tool_call_id;
};

ChatTemplateStyle style_from_model(const ModelConfig& cfg);

std::string render_chat(const std::vector<ChatMessage>& messages,
                        const ToolRegistry& tools,
                        ChatTemplateStyle style,
                        bool add_gen_prompt);

} // namespace llm
