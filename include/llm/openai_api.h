// openai_api.h — OpenAI-compatible /v1/chat/completions request/response shapes.
//
// A small, zero-dependency mapping between the OpenAI chat-completions wire
// format and SipLLM's internal types (tools.h). The server's
// /v1/chat/completions route supports two modes:
//   * passthrough (default): one model turn per request; if the model calls a
//     tool the response carries OpenAI-format `tool_calls` for the CLIENT to
//     execute (standard OpenAI tool protocol).
//   * agent (request "agent": true): the server runs the Nishachar loop with
//     its own registered tools and returns the finished answer.
// The parse/build logic here is pure (no sockets, no Runtime) so it is
// unit-tested without a server or a model.
#pragma once

#include "llm/tools.h"  // ChatMessage, ToolDef

#include <string>
#include <vector>

namespace llm {

// The subset of an OpenAI /v1/chat/completions request we honor.
struct ChatCompletionRequest {
    std::string              model;                  // advisory; the server serves one model
    std::vector<ChatMessage> messages;              // role (system/user/assistant/tool) + content
    std::vector<ToolDef>     tools;                  // OpenAI tools[].function -> ToolDef
    int                      max_tokens  = 256;      // per generation step
    float                    temperature = 0.0f;     // 0 => greedy
    bool                     stream      = false;    // SSE streaming (server may decline)
    bool                     agent       = false;    // opt-in: run the Nishachar loop server-side
};

// Parse a /v1/chat/completions request body (JSON). Returns false and fills
// `err` on malformed JSON or a missing/empty `messages` array; otherwise fills
// `out`. Zero-dependency, hand-written parser (no JSON library). Unknown fields
// are ignored; unknown message roles map to ChatMessage::Role::User. Each entry
// of `tools` is parsed from tools[].function {name, description, parameters:
// {properties{...}, required[...]}} into a ToolDef (best-effort param typing).
bool parse_chat_request(const std::string& body, ChatCompletionRequest& out,
                        std::string& err);

// The goal handed to the agent loop = the content of the LAST user message
// (empty if the request has no user message).
std::string last_user_message(const std::vector<ChatMessage>& messages);

// A tool call returned to the client in passthrough mode (OpenAI format).
struct ResponseToolCall {
    std::string id;         // e.g. "call_1"
    std::string name;       // function name
    std::string arguments;  // JSON object string of arguments
};

// Fields needed to render an OpenAI-compatible chat.completion response.
struct ChatCompletionResponse {
    std::string                   id;                       // e.g. "chatcmpl-1"
    long                          created = 0;              // unix epoch seconds
    std::string                   model;
    std::string                   content;                  // assistant content (empty if tool_calls)
    std::vector<ResponseToolCall> tool_calls;               // non-empty => passthrough tool turn
    std::string                   finish_reason = "stop";   // "stop" | "length" | "tool_calls"
    int                           prompt_tokens = 0;
    int                           completion_tokens = 0;
};

// Render `r` as an OpenAI-compatible chat.completion JSON string:
//   {"id","object":"chat.completion","created","model",
//    "choices":[{"index":0,"message":{"role":"assistant","content":...
//                [,"tool_calls":[{"id","type":"function",
//                                 "function":{"name","arguments"}}]]},
//                "finish_reason":...}],
//    "usage":{"prompt_tokens","completion_tokens","total_tokens"}}
// When tool_calls is non-empty, "content" is JSON null and finish_reason is
// forced to "tool_calls"; otherwise the content string is JSON-escaped.
std::string build_chat_response(const ChatCompletionResponse& r);

} // namespace llm
