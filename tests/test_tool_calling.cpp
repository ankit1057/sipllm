// test_tool_calling.cpp — tool-calling + chat templates (self-contained).
//
// Exercises the zero-dependency tool-call parser, the tool registry / schema
// rendering, and chat-template rendering — none of which touch the language
// model, so this links and runs independently of the rest of the engine.
#include "llm/tools.h"
#include "tests/test_util.h"

#include <string>

using namespace llm;

static ToolRegistry make_registry() {
    ToolRegistry reg;
    ToolDef weather;
    weather.name = "get_weather";
    weather.description = "Get the weather for a city";
    weather.params = {
        {"city", ToolParamType::String, true, "City name", ""},
        {"units", ToolParamType::String, false, "metric or imperial", "metric"},
    };
    reg.register_tool(weather);
    return reg;
}

TEST(registry_register_find_unregister) {
    ToolRegistry reg = make_registry();
    CHECK(reg.size() == 1);
    CHECK(reg.find("get_weather") != nullptr);
    CHECK(reg.find("nope") == nullptr);
    // Re-registering the same name overwrites, does not duplicate.
    ToolDef again; again.name = "get_weather"; again.description = "v2";
    reg.register_tool(again);
    CHECK(reg.size() == 1);
    CHECK(reg.find("get_weather")->description == "v2");
    reg.unregister_tool("get_weather");
    CHECK(reg.empty());
}

TEST(schema_text_mentions_params) {
    ToolRegistry reg = make_registry();
    std::string s = reg.find("get_weather")->schema_text();
    CHECK(s.find("get_weather") != std::string::npos);
    CHECK(s.find("city") != std::string::npos);
    CHECK(s.find("units") != std::string::npos);
    // The full system-prompt block references the tool.
    std::string block = reg.system_prompt_block();
    CHECK(block.find("get_weather") != std::string::npos);
    CHECK(block.find("arguments") != std::string::npos);
}

TEST(parser_raw_json_single_call) {
    ToolRegistry reg = make_registry();
    ToolParser p(reg);
    p.set_markers({"", ""});   // raw-JSON mode
    bool done = p.feed("Sure! {\"name\": \"get_weather\", "
                       "\"arguments\": {\"city\": \"Paris\", \"units\": \"metric\"}}");
    CHECK(done);
    const ToolCall& c = p.parsed_call();
    CHECK(c.name == "get_weather");
    CHECK(c.has("city") && c.get("city") == "Paris");
    CHECK(c.get("units") == "metric");
    CHECK(c.get("missing", "fallback") == "fallback");
}

TEST(parser_marker_mode) {
    ToolRegistry reg = make_registry();
    ToolParser p(reg);   // default markers <tool_call>...</tool_call>
    bool done = p.feed("<tool_call>{\"name\":\"get_weather\","
                       "\"arguments\":{\"city\":\"Tokyo\"}}</tool_call>");
    CHECK(done);
    CHECK(p.parsed_call().name == "get_weather");
    CHECK(p.parsed_call().get("city") == "Tokyo");
}

TEST(parser_incremental_chunks) {
    // A streamed call arriving in pieces must parse once the object completes.
    ToolRegistry reg = make_registry();
    ToolParser p(reg);
    p.set_markers({"", ""});
    CHECK(!p.feed("{\"name\": \"get_weather\", \"argum"));
    CHECK(!p.feed("ents\": {\"city\": \"Berl"));
    bool done = p.feed("in\"}}");
    CHECK(done);
    CHECK(p.parsed_call().get("city") == "Berlin");
}

TEST(parser_rejects_unregistered_tool) {
    ToolRegistry reg = make_registry();
    ToolParser p(reg);
    p.set_markers({"", ""});
    bool done = p.feed("{\"name\": \"launch_missiles\", \"arguments\": {}}");
    CHECK(!done);   // not a registered tool -> not accepted
}

TEST(parser_escaped_and_nested_values) {
    ToolRegistry reg;
    ToolDef t; t.name = "run"; reg.register_tool(t);
    ToolParser p(reg);
    p.set_markers({"", ""});
    bool done = p.feed("{\"name\":\"run\",\"arguments\":{"
                       "\"cmd\":\"echo \\\"hi\\\"\",\"opts\":{\"n\":3},\"flag\":true}}");
    CHECK(done);
    const ToolCall& c = p.parsed_call();
    CHECK(c.get("cmd") == "echo \"hi\"");        // unescaped quotes
    CHECK(c.get("opts") == "{\"n\":3}");          // nested object captured raw
    CHECK(c.get("flag") == "true");               // bool verbatim
}

TEST(chat_template_llama3) {
    ToolRegistry empty;
    std::vector<ChatMessage> msgs = {
        {ChatMessage::Role::System, "You are helpful.", ""},
        {ChatMessage::Role::User, "Hi", ""},
    };
    std::string s = render_chat(msgs, empty, ChatTemplateStyle::Llama3, true);
    CHECK(s.find("<|begin_of_text|>") != std::string::npos);
    CHECK(s.find("<|start_header_id|>system<|end_header_id|>") != std::string::npos);
    CHECK(s.find("You are helpful.") != std::string::npos);
    CHECK(s.find("<|start_header_id|>assistant<|end_header_id|>") != std::string::npos); // gen prompt
}

TEST(chat_template_qwen_and_gemma) {
    ToolRegistry empty;
    std::vector<ChatMessage> msgs = { {ChatMessage::Role::User, "Hello", ""} };
    std::string q = render_chat(msgs, empty, ChatTemplateStyle::Qwen2, true);
    CHECK(q.find("<|im_start|>user") != std::string::npos);
    CHECK(q.find("<|im_start|>assistant") != std::string::npos);
    std::string g = render_chat(msgs, empty, ChatTemplateStyle::Gemma, true);
    CHECK(g.find("<start_of_turn>user") != std::string::npos);
    CHECK(g.find("<start_of_turn>model") != std::string::npos);
}

TEST(chat_template_injects_tools) {
    ToolRegistry reg = make_registry();
    std::vector<ChatMessage> msgs = { {ChatMessage::Role::User, "weather?", ""} };
    // No system message present -> render_chat must synthesize one with tools.
    std::string s = render_chat(msgs, reg, ChatTemplateStyle::Qwen2, true);
    CHECK(s.find("get_weather") != std::string::npos);
    CHECK(s.find("<|im_start|>system") != std::string::npos);
}

TEST(style_from_model_maps_arch) {
    ModelConfig c;
    c.arch_kind = Arch::Gemma2;  CHECK(style_from_model(c) == ChatTemplateStyle::Gemma);
    c.arch_kind = Arch::Qwen2;   CHECK(style_from_model(c) == ChatTemplateStyle::Qwen2);
    c.arch_kind = Arch::Mistral; CHECK(style_from_model(c) == ChatTemplateStyle::Mistral);
    c.arch_kind = Arch::Phi3;    CHECK(style_from_model(c) == ChatTemplateStyle::Phi3);
    c.arch_kind = Arch::Llama;   CHECK(style_from_model(c) == ChatTemplateStyle::Llama3);
    c.arch_kind = Arch::Gemma4;  CHECK(style_from_model(c) == ChatTemplateStyle::Gemma4);   // <|turn> markers, not <start_of_turn>
}

TEST(style_from_model_prefers_gguf_template) {
    // Arch "llama" covers TinyLlama (Zephyr), SmolLM2 (ChatML), Llama-2 ([INST])
    // and Llama-3: the GGUF's own template decides.
    ModelConfig c;
    c.arch_kind = Arch::Llama;
    c.chat_template = "{{ '<|user|>\\n' + message['content'] + eos_token }}{{ '<|assistant|>' }}";
    CHECK(style_from_model(c) == ChatTemplateStyle::Zephyr);
    c.chat_template = "{{'<|im_start|>' + message['role'] + '\\n'}}";
    CHECK(style_from_model(c) == ChatTemplateStyle::ChatML);
    c.chat_template = "{{ bos_token + '[INST] ' + message['content'] + ' [/INST]' }}";
    CHECK(style_from_model(c) == ChatTemplateStyle::Mistral);
    c.chat_template = "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>' }}";
    CHECK(style_from_model(c) == ChatTemplateStyle::Llama3);
    c.arch_kind = Arch::Phi3;
    c.chat_template = "{{'<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>'}}{{'<|assistant|>'}}";
    CHECK(style_from_model(c) == ChatTemplateStyle::Phi3);
}

TEST(chat_template_zephyr) {
    ToolRegistry empty;
    std::vector<ChatMessage> msgs = {
        {ChatMessage::Role::System, "Be brief.", ""},
        {ChatMessage::Role::User, "Hello", ""},
    };
    CHECK(render_chat(msgs, empty, ChatTemplateStyle::Zephyr, true) ==
          "<|system|>\nBe brief.</s>\n<|user|>\nHello</s>\n<|assistant|>\n");
}

TEST(tool_json_schema) {
    ToolDef t = *make_registry().find("get_weather");
    CHECK(t.json_schema() ==
          "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", "
          "\"description\": \"Get the weather for a city\", \"parameters\": {\"type\": \"object\", "
          "\"properties\": {\"city\": {\"type\": \"string\", \"description\": \"City name\"}, "
          "\"units\": {\"type\": \"string\", \"description\": \"metric or imperial\"}}, "
          "\"required\": [\"city\"]}}}");
    ToolDef q;
    q.name = "say \"hi\"";
    q.description = "line1\nline2";
    q.params = {{"tags", ToolParamType::StringArray, false, "", ""}};
    CHECK(q.json_schema() ==
          "{\"type\": \"function\", \"function\": {\"name\": \"say \\\"hi\\\"\", "
          "\"description\": \"line1\\nline2\", \"parameters\": {\"type\": \"object\", "
          "\"properties\": {\"tags\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}}}}}}");
}

// Golden strings rendered by jinja2 from Qwen's official chat template
// (Qwen/Qwen2.5-1.5B-Instruct tokenizer_config.json; HF's tojson: no key
// sorting, no HTML escaping). The copy embedded in some GGUFs differs only in
// writing the example call with doubled braces.
static const char* kQwenToolHeader =
    "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n"
    "{\"type\": \"function\", \"function\": {\"name\": \"get_weather\", \"description\": \"Get the weather for a city\", "
    "\"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"City name\"}, "
    "\"units\": {\"type\": \"string\", \"description\": \"metric or imperial\"}}, \"required\": [\"city\"]}}}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";

TEST(chat_template_qwen_tools_matches_gguf_template) {
    ToolRegistry reg = make_registry();
    const std::string call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Pune\"}}\n</tool_call>";
    std::vector<ChatMessage> msgs = {
        {ChatMessage::Role::System, "Be brief.", ""},
        {ChatMessage::Role::User, "weather in Pune?", ""},
        {ChatMessage::Role::Assistant, call, ""},
        {ChatMessage::Role::Tool, "31C, clear", ""},
    };
    const std::string want = std::string("<|im_start|>system\nBe brief.") + kQwenToolHeader +
        "<|im_start|>user\nweather in Pune?<|im_end|>\n<|im_start|>assistant\n" + call +
        "<|im_end|>\n<|im_start|>user\n<tool_response>\n31C, clear\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n";
    CHECK(render_chat(msgs, reg, ChatTemplateStyle::Qwen2, true, ToolPromptFormat::Qwen) == want);
    CHECK(render_chat(msgs, reg, ChatTemplateStyle::ChatML, true, ToolPromptFormat::Qwen) == want);

    // No system message: the template's default system line.
    std::vector<ChatMessage> bare = { {ChatMessage::Role::User, "weather in Pune?", ""} };
    CHECK(render_chat(bare, reg, ChatTemplateStyle::Qwen2, true, ToolPromptFormat::Qwen) ==
          std::string("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.") +
          kQwenToolHeader + "<|im_start|>user\nweather in Pune?<|im_end|>\n<|im_start|>assistant\n");

    // Consecutive tool results share one user turn.
    msgs.push_back({ChatMessage::Role::Tool, "humid", ""});
    std::string two = render_chat(msgs, reg, ChatTemplateStyle::Qwen2, false, ToolPromptFormat::Qwen);
    CHECK(two.find("<|im_start|>user\n<tool_response>\n31C, clear\n</tool_response>\n<tool_response>\n"
                   "humid\n</tool_response><|im_end|>\n") != std::string::npos);
}

TEST(chat_template_qwen_tools_falls_back_to_compact) {
    // Only ChatML-family styles have the Qwen tool format; no tools means no
    // tool block. Both cases render exactly as the default (compact) format.
    ToolRegistry reg = make_registry(), empty;
    std::vector<ChatMessage> msgs = { {ChatMessage::Role::System, "Be brief.", ""},
                                      {ChatMessage::Role::User, "Hi", ""} };
    for (ChatTemplateStyle st : {ChatTemplateStyle::Llama3, ChatTemplateStyle::Gemma, ChatTemplateStyle::Zephyr})
        CHECK(render_chat(msgs, reg, st, true, ToolPromptFormat::Qwen) == render_chat(msgs, reg, st, true));
    CHECK(render_chat(msgs, empty, ChatTemplateStyle::Qwen2, true, ToolPromptFormat::Qwen) ==
          render_chat(msgs, empty, ChatTemplateStyle::Qwen2, true));
    CHECK(render_chat(msgs, reg, ChatTemplateStyle::Qwen2, true, ToolPromptFormat::Compact) ==
          render_chat(msgs, reg, ChatTemplateStyle::Qwen2, true));
}

TEST(tool_format_from_model_reads_gguf_template) {
    ModelConfig c;
    CHECK(tool_format_from_model(c) == ToolPromptFormat::Compact);   // no template
    c.chat_template = "{{'<|im_start|>' + message['role'] + '\\n'}}";
    CHECK(tool_format_from_model(c) == ToolPromptFormat::Compact);   // ChatML, no tools
    c.chat_template = "{%- if tools %}<|im_start|>system\\n<tools></tools><tool_call>\\n</tool_call>{%- endif %}";
    CHECK(tool_format_from_model(c) == ToolPromptFormat::Qwen);
}

int main() {
    printf("== test_tool_calling ==\n");
    return llmtest::run_all();
}
