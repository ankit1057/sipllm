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
