// test_nishachar.cpp — Nishachar Path A loop (scripted-generator unit tests).
//
// Exercises the autonomous goal->plan->act->verify loop deterministically with
// a scripted AgentGenerator (no language model): registers real tool handlers
// on a Nishachar and drives run() through every stop reason and dispatch path.
#include "llm/nishachar.h"
#include "llm/tools.h"
#include "tests/test_util.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

// A scripted generator: yields `outputs` in order across successive calls, then
// empty once exhausted. The shared index makes the counter survive the copy the
// std::function wrapper performs.
static AgentGenerator scripted(std::vector<std::string> outputs) {
    auto idx = std::make_shared<size_t>(0);
    auto outs = std::make_shared<std::vector<std::string>>(std::move(outputs));
    return [idx, outs](const std::string&) -> std::string {
        if (*idx >= outs->size()) return "";
        return (*outs)[(*idx)++];
    };
}

// A generator that always returns the same output regardless of step.
static AgentGenerator always(std::string output) {
    auto out = std::make_shared<std::string>(std::move(output));
    return [out](const std::string&) -> std::string { return *out; };
}

// Build a Nishachar with echo / add / boom tools registered.
static Nishachar make_agent() {
    Nishachar agent;

    ToolDef echo;
    echo.name = "echo";
    echo.description = "Echo the text argument back";
    echo.params = {{"text", ToolParamType::String, true, "text to echo", ""}};
    agent.add_tool(echo, [](const ToolCall& c) -> std::string {
        return c.get("text");
    });

    ToolDef add;
    add.name = "add";
    add.description = "Add two integers";
    add.params = {
        {"a", ToolParamType::Int, true, "first addend", ""},
        {"b", ToolParamType::Int, true, "second addend", ""},
    };
    agent.add_tool(add, [](const ToolCall& c) -> std::string {
        int a = std::stoi(c.get("a", "0"));
        int b = std::stoi(c.get("b", "0"));
        return std::to_string(a + b);
    });

    ToolDef boom;
    boom.name = "boom";
    boom.description = "Always fails";
    agent.add_tool(boom, [](const ToolCall&) -> std::string {
        throw std::runtime_error("boom");
    });

    return agent;
}

static const char* kEchoHi =
    "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"hi\"}}</tool_call>";
static const char* kAdd23 =
    "<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":\"2\",\"b\":\"3\"}}</tool_call>";
static const char* kBoom =
    "<tool_call>{\"name\":\"boom\",\"arguments\":{}}</tool_call>";
static const char* kUnknown =
    "<tool_call>{\"name\":\"nope\",\"arguments\":{}}</tool_call>";

TEST(immediate_final) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("say hello", scripted({"Hello there!"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.final_text == "Hello there!");
    CHECK(r.steps_taken() == 1);
    CHECK(r.tool_calls() == 0);
    CHECK(!r.steps[0].had_tool_call);
}

TEST(single_tool_then_final) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("echo hi", scripted({kEchoHi, "all done"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 1);
    CHECK(r.steps_taken() == 2);
    CHECK(r.steps[0].had_tool_call);
    CHECK(r.steps[0].tool_name == "echo");
    CHECK(r.steps[0].tool_result == "hi");
    CHECK(r.steps[0].tool_ok);
    CHECK(!r.steps[1].had_tool_call);
    CHECK(r.final_text == "all done");
}

TEST(multi_step_chain) {
    Nishachar agent = make_agent();
    AgentResult r =
        agent.run("chain", scripted({kEchoHi, kAdd23, "finished"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 2);
    CHECK(r.steps_taken() == 3);
    CHECK(r.steps[0].tool_name == "echo");
    CHECK(r.steps[0].tool_result == "hi");
    CHECK(r.steps[0].tool_ok);
    CHECK(r.steps[1].tool_name == "add");
    CHECK(r.steps[1].tool_result == "5");
    CHECK(r.steps[1].tool_ok);
    CHECK(!r.steps[2].had_tool_call);
    CHECK(r.final_text == "finished");
}

TEST(max_steps_bound) {
    Nishachar agent = make_agent();
    AgentConfig cfg;
    cfg.max_steps = 3;
    AgentResult r = agent.run("loop forever", always(kEchoHi), cfg);
    CHECK(r.stop == AgentStop::MaxSteps);
    CHECK(r.steps_taken() == 3);
    CHECK(r.tool_calls() == 3);
}

TEST(unknown_tool_is_final) {
    // The parser is constructed with the registry and rejects unregistered
    // tools, so a call to "nope" is never detected -> the output is treated as
    // the final answer.
    Nishachar agent = make_agent();
    AgentResult r = agent.run("call unknown", scripted({kUnknown}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 0);
    CHECK(r.steps_taken() == 1);
    CHECK(!r.steps[0].had_tool_call);
    CHECK(r.final_text == kUnknown);
}

TEST(tool_error_continue) {
    Nishachar agent = make_agent();
    AgentConfig cfg;
    cfg.stop_on_tool_error = false;
    AgentResult r = agent.run("recover", scripted({kBoom, "recovered"}), cfg);
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.steps_taken() == 2);
    CHECK(r.steps[0].had_tool_call);
    CHECK(!r.steps[0].tool_ok);
    CHECK(r.steps[0].tool_result.rfind("error:", 0) == 0);
    CHECK(r.final_text == "recovered");
}

TEST(tool_error_stop) {
    Nishachar agent = make_agent();
    AgentConfig cfg;
    cfg.stop_on_tool_error = true;
    AgentResult r = agent.run("fail hard", scripted({kBoom}), cfg);
    CHECK(r.stop == AgentStop::ToolError);
    CHECK(r.tool_calls() == 1);
    CHECK(r.steps_taken() == 1);
    CHECK(!r.steps[0].tool_ok);
}

TEST(empty_generation) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("nothing", scripted({""}));
    CHECK(r.stop == AgentStop::Empty);
    CHECK(r.tool_calls() == 0);
}

TEST(tool_schema_rendered_once) {
    // The tool block is prefill paid on every step. run() used to add it to the
    // system message and render_chat() added it again: two copies, about a
    // third of a real 433-token Qwen2.5 step-0 prompt.
    Nishachar agent = make_agent();
    auto prompts = std::make_shared<std::vector<std::string>>();
    AgentGenerator gen = [prompts](const std::string& p) -> std::string {
        prompts->push_back(p);
        return prompts->size() == 1 ? kEchoHi : "done";
    };
    AgentResult r = agent.run("echo hi", gen);
    CHECK(r.stop == AgentStop::Done);
    CHECK(prompts->size() == 2);
    const std::string block = agent.tools().system_prompt_block();
    for (const std::string& p : *prompts) {
        size_t first = p.find(block);
        CHECK_MSG(first != std::string::npos, "the tool block is missing from the prompt");
        CHECK_MSG(p.find(block, first + 1) == std::string::npos,
                  "the tool block appears more than once in the prompt");
        CHECK(p.find(AgentConfig{}.system_preamble) != std::string::npos);
    }
}

TEST(qwen_tool_format_feeds_results_as_tool_response) {
    // Opt-in Qwen format: schemas in <tools>, the result back as <tool_response>
    // in a user turn, which is what Qwen2.5 was trained on.
    Nishachar agent = make_agent();
    auto prompts = std::make_shared<std::vector<std::string>>();
    AgentGenerator gen = [prompts](const std::string& p) -> std::string {
        prompts->push_back(p);
        return prompts->size() == 1 ? kEchoHi : "done";
    };
    AgentConfig cfg;
    cfg.tool_format = ToolPromptFormat::Qwen;
    AgentResult r = agent.run("echo hi", gen, cfg);
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.final_text == "done");
    CHECK(prompts->size() == 2);
    CHECK((*prompts)[0].find("<tools>\n{\"type\": \"function\"") != std::string::npos);
    CHECK((*prompts)[1].find(std::string(kEchoHi) + "<|im_end|>\n<|im_start|>user\n<tool_response>\nhi\n"
                             "</tool_response><|im_end|>\n<|im_start|>assistant\n") != std::string::npos);
    // The default stays the compact format.
    CHECK(AgentConfig{}.tool_format == ToolPromptFormat::Compact);
}

TEST(session_steps_like_run) {
    // The caller-stepped session: it hands back the parsed call, waits for the
    // result, and ends exactly as run() does on the same model outputs.
    Nishachar agent = make_agent();
    AgentSession s(agent, "echo hi", AgentConfig{});
    CHECK(!s.finished() && !s.awaiting_result());
    CHECK(s.submit_output(kEchoHi) == AgentSession::Event::ToolCall);
    CHECK(s.awaiting_result());
    CHECK(s.pending_call().name == "echo");
    CHECK(s.pending_call().get("text") == "hi");
    s.submit_result("hi from the app", true);
    CHECK(!s.awaiting_result() && !s.finished());
    CHECK(s.prompt().find("hi from the app") != std::string::npos);   // fed back
    CHECK(s.submit_output("all done") == AgentSession::Event::Final);
    CHECK(s.finished());

    const AgentResult& r = s.result();
    AgentResult ran = agent.run("echo hi", scripted({kEchoHi, "all done"}));
    CHECK(r.stop == AgentStop::Done && ran.stop == AgentStop::Done);
    CHECK(r.steps_taken() == 2 && ran.steps_taken() == 2);
    CHECK(r.final_text == "all done" && ran.final_text == "all done");
    CHECK(r.steps[0].tool_name == "echo" && r.steps[0].tool_result == "hi from the app");
    CHECK(r.steps[0].tool_args_json == ran.steps[0].tool_args_json);
}

TEST(session_bounds_and_errors) {
    Nishachar agent = make_agent();
    AgentConfig cfg;
    cfg.max_steps = 2;
    AgentSession s(agent, "loop", cfg);
    for (int i = 0; i < 2; ++i) {
        CHECK(!s.finished());
        CHECK(s.submit_output(kEchoHi) == AgentSession::Event::ToolCall);
        s.submit_result("hi", true);
    }
    CHECK(s.finished());
    CHECK(s.result().stop == AgentStop::MaxSteps);
    CHECK(s.result().steps_taken() == 2);

    cfg.max_steps = 8;
    cfg.stop_on_tool_error = true;
    AgentSession e(agent, "fail", cfg);
    CHECK(e.submit_output(kBoom) == AgentSession::Event::ToolCall);
    e.submit_result("error: the app refused", false);
    CHECK(e.finished());
    CHECK(e.result().stop == AgentStop::ToolError);
    CHECK(!e.result().steps[0].tool_ok);

    AgentSession empty(agent, "nothing", AgentConfig{});
    CHECK(empty.submit_output("") == AgentSession::Event::Empty);
    CHECK(empty.finished() && empty.result().stop == AgentStop::Empty);

    cfg.max_steps = 0;
    AgentSession none(agent, "zero", cfg);
    CHECK(none.finished());
    CHECK(none.result().stop == AgentStop::MaxSteps && none.result().steps_taken() == 0);
}

TEST(malformed_calls_never_dispatch) {
    // Model outputs that look like calls but are not usable ones must end the
    // run inside the bound with nothing dispatched, through run() and through
    // the stepped session alike.
    const std::vector<std::string> bad = {
        "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"hi",   // cut off by max_new
        "<tool_call>{\"arguments\":{\"text\":\"hi\"}}</tool_call>",      // no name
        "<tool_call>{\"name\":\"nope\",\"arguments\":{}}</tool_call>",   // unregistered
        "{\"name\": 42, \"arguments\": {}}",                             // name not a string
    };
    for (const std::string& out : bad) {
        Nishachar agent = make_agent();
        AgentResult r = agent.run("x", always(out));
        CHECK_MSG(r.tool_calls() == 0, ("dispatched: " + out).c_str());
        CHECK_MSG(r.stop == AgentStop::Done && r.steps_taken() == 1, ("did not end: " + out).c_str());
        AgentSession s(agent, "x", AgentConfig{});
        CHECK_MSG(s.submit_output(out) == AgentSession::Event::Final, ("session: " + out).c_str());
        CHECK(s.finished() && !s.awaiting_result());
    }
    // A string where the arguments object should be: the call is dispatched
    // with no arguments, and the handler sees none (echo returns "").
    Nishachar agent = make_agent();
    AgentResult r = agent.run("x", scripted({"<tool_call>{\"name\":\"echo\",\"arguments\":\"hi\"}</tool_call>", "done"}));
    CHECK(r.tool_calls() == 1 && r.steps[0].tool_result.empty());
    CHECK(r.stop == AgentStop::Done);
    // The doubled braces some GGUF templates teach still parse to the call.
    AgentSession s(agent, "x", AgentConfig{});
    CHECK(s.submit_output("<tool_call>\n{{\"name\": \"echo\", \"arguments\": {\"text\": \"hi\"}}}}\n</tool_call>") ==
          AgentSession::Event::ToolCall);
    CHECK(s.pending_call().name == "echo" && s.pending_call().get("text") == "hi");
}

TEST(report_mentions) {
    Nishachar agent = make_agent();
    AgentResult r =
        agent.run("please echo hi", scripted({kEchoHi, "all done"}));
    std::string rep = r.report();
    CHECK(rep.find("please echo hi") != std::string::npos);
    CHECK(rep.find("echo") != std::string::npos);
}

int main() {
    printf("== test_nishachar ==\n");
    return llmtest::run_all();
}
