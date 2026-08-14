// test_openai_api.cpp — OpenAI /v1/chat/completions parse/build (pure).
//
// Exercises the zero-dependency request parser (parse_chat_request),
// last_user_message, and the response builder (build_chat_response). None of
// these touch a socket, a Runtime, or a model, so this links and runs on its
// own. Assertions use substring checks so minor JSON spacing differences do not
// make the tests brittle.
#include "llm/openai_api.h"
#include "llm/tools.h"
#include "tests/test_util.h"

#include <string>

using namespace llm;

// --------------------------------------------------------------------------
// parse_chat_request
// --------------------------------------------------------------------------
TEST(parse_basic) {
    const std::string body =
        "{\"model\":\"m\",\"messages\":["
        "{\"role\":\"system\",\"content\":\"s\"},"
        "{\"role\":\"user\",\"content\":\"hi\"}],"
        "\"max_tokens\":32,\"temperature\":0.5}";
    ChatCompletionRequest out;
    std::string err;
    CHECK(parse_chat_request(body, out, err));
    CHECK(out.model == "m");
    CHECK(out.messages.size() == 2);
    CHECK(out.messages[0].role == ChatMessage::Role::System);
    CHECK(out.messages[1].role == ChatMessage::Role::User);
    CHECK(out.messages[1].content == "hi");
    CHECK(out.max_tokens == 32);
    APPROX(out.temperature, 0.5, 1e-6);
    CHECK(out.agent == false);
}

TEST(parse_agent_flag) {
    const std::string body =
        "{\"model\":\"m\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"hi\"}],\"agent\":true}";
    ChatCompletionRequest out;
    std::string err;
    CHECK(parse_chat_request(body, out, err));
    CHECK(out.agent == true);
}

TEST(parse_tools) {
    const std::string body =
        "{\"model\":\"m\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"hi\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
        "\"name\":\"get_weather\",\"description\":\"d\","
        "\"parameters\":{\"type\":\"object\",\"properties\":{"
        "\"city\":{\"type\":\"string\"},"
        "\"days\":{\"type\":\"integer\"}},"
        "\"required\":[\"city\"]}}}]}";
    ChatCompletionRequest out;
    std::string err;
    CHECK(parse_chat_request(body, out, err));
    CHECK(out.tools.size() == 1);
    CHECK(out.tools[0].name == "get_weather");

    bool city_ok = false;
    bool days_ok = false;
    for (const auto& p : out.tools[0].params) {
        if (p.name == "city") {
            city_ok = (p.type == ToolParamType::String) && p.required;
        } else if (p.name == "days") {
            days_ok = (p.type == ToolParamType::Int);
        }
    }
    CHECK_MSG(city_ok, "param 'city' should be a required String");
    CHECK_MSG(days_ok, "param 'days' should be an Int");
}

TEST(parse_unknown_role_is_user) {
    const std::string body =
        "{\"model\":\"m\",\"messages\":["
        "{\"role\":\"function\",\"content\":\"a\"},"
        "{\"role\":\"weird\",\"content\":\"b\"}]}";
    ChatCompletionRequest out;
    std::string err;
    CHECK(parse_chat_request(body, out, err));
    CHECK(out.messages.size() == 2);
    CHECK(out.messages[0].role == ChatMessage::Role::User);
    CHECK(out.messages[1].role == ChatMessage::Role::User);
}

TEST(parse_missing_messages) {
    const std::string body = "{\"model\":\"m\"}";
    ChatCompletionRequest out;
    std::string err;
    CHECK(!parse_chat_request(body, out, err));
    CHECK(!err.empty());
}

TEST(parse_malformed) {
    const std::string body = "{not json";
    ChatCompletionRequest out;
    std::string err;
    CHECK(!parse_chat_request(body, out, err));
    CHECK(!err.empty());
}

TEST(parse_string_with_braces) {
    // The user content itself contains JSON-ish punctuation: {"a":1} and ]}.
    // A correct parser must not be fooled by braces/brackets inside a string.
    const std::string body =
        "{\"model\":\"m\",\"messages\":["
        "{\"role\":\"user\",\"content\":\"{\\\"a\\\":1}]}\"}]}";
    ChatCompletionRequest out;
    std::string err;
    CHECK(parse_chat_request(body, out, err));
    CHECK(out.messages.size() == 1);
    CHECK(out.messages[0].role == ChatMessage::Role::User);
    CHECK(out.messages[0].content == "{\"a\":1}]}");
}

// --------------------------------------------------------------------------
// last_user_message
// --------------------------------------------------------------------------
TEST(last_user) {
    std::vector<ChatMessage> msgs = {
        {ChatMessage::Role::System, "sys", ""},
        {ChatMessage::Role::User, "a", ""},
        {ChatMessage::Role::Assistant, "resp", ""},
        {ChatMessage::Role::User, "b", ""},
    };
    CHECK(last_user_message(msgs) == "b");

    std::vector<ChatMessage> empty;
    CHECK(last_user_message(empty) == "");

    std::vector<ChatMessage> no_user = {
        {ChatMessage::Role::System, "sys", ""},
        {ChatMessage::Role::Assistant, "resp", ""},
    };
    CHECK(last_user_message(no_user) == "");
}

// --------------------------------------------------------------------------
// build_chat_response
// --------------------------------------------------------------------------
TEST(build_content) {
    ChatCompletionResponse r;
    r.id = "chatcmpl-1";
    r.model = "m";
    r.content = "hello \"world\"";
    r.finish_reason = "stop";
    r.prompt_tokens = 3;
    r.completion_tokens = 4;

    std::string out = build_chat_response(r);
    CHECK(out.find("\"object\"") != std::string::npos);
    CHECK(out.find("chat.completion") != std::string::npos);
    // Quotes inside content must be backslash-escaped in the JSON output.
    CHECK(out.find("hello \\\"world\\\"") != std::string::npos);
    CHECK(out.find("\"finish_reason\"") != std::string::npos);
    CHECK(out.find("stop") != std::string::npos);
    // total_tokens = prompt + completion = 7.
    CHECK(out.find("\"total_tokens\"") != std::string::npos);
    CHECK(out.find("7") != std::string::npos);
}

TEST(build_tool_calls) {
    ChatCompletionResponse r;
    r.id = "chatcmpl-2";
    r.model = "m";
    r.tool_calls.push_back(
        ResponseToolCall{"call_1", "get_weather", "{\"city\":\"Paris\"}"});

    std::string out = build_chat_response(r);
    CHECK(out.find("\"tool_calls\"") != std::string::npos);
    CHECK(out.find("get_weather") != std::string::npos);
    CHECK(out.find("\"finish_reason\"") != std::string::npos);
    CHECK(out.find("tool_calls") != std::string::npos);
    // content must be JSON null when tool_calls is present.
    CHECK(out.find("\"content\": null") != std::string::npos ||
          out.find("\"content\":null") != std::string::npos);
}

int main() {
    printf("== test_openai_api ==\n");
    return llmtest::run_all();
}
