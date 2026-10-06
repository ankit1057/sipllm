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

TEST(unknown_tool_is_fed_back) {
    // A marked call to an unregistered tool is an attempted call: the loop
    // feeds an error back (listing the real tools) and keeps going.
    Nishachar agent = make_agent();
    AgentResult r = agent.run("call unknown", scripted({kUnknown, "ok then"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 0);
    CHECK(r.steps_taken() == 2);
    CHECK(r.steps[0].malformed);
    CHECK(r.steps[0].tool_result.find("unknown tool 'nope'") != std::string::npos);
    CHECK(r.steps[0].tool_result.find("echo") != std::string::npos);
    CHECK(r.final_text == "ok then");
}

TEST(malformed_call_recovery) {
    Nishachar agent = make_agent();
    // Unterminated (cut off), broken JSON (unescaped quote), then a good call.
    std::vector<std::string> prompts;
    auto inner = scripted({
        "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"hi",
        "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"say \"hi\" now\"}}</tool_call>",
        kEchoHi,
        "done",
    });
    AgentGenerator gen = [&](const std::string& p) { prompts.push_back(p); return inner(p); };
    AgentResult r = agent.run("echo hi", gen);
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.steps_taken() == 4);
    CHECK(r.steps[0].malformed && !r.steps[0].had_tool_call);
    CHECK(r.steps[0].tool_result.rfind("error: could not parse your tool call (", 0) == 0);
    CHECK(r.steps[0].tool_result.find("escape newlines") != std::string::npos);
    CHECK(r.steps[1].malformed);
    CHECK(r.steps[1].tool_result.find("invalid JSON") != std::string::npos);
    CHECK(r.steps[2].had_tool_call && r.steps[2].tool_result == "hi");
    CHECK(r.tool_calls() == 1);
    CHECK(r.final_text == "done");
    // The error text was fed back into the next prompt.
    CHECK(prompts.size() == 4);
    CHECK(prompts[1].find("could not parse your tool call") != std::string::npos);
    CHECK(r.report().find("malformed") != std::string::npos);
}

TEST(missing_required_arg_is_malformed) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("x", scripted({
        "<tool_call>{\"name\":\"echo\",\"arguments\":{}}</tool_call>", "fine"}));
    CHECK(r.steps[0].malformed);
    CHECK(r.steps[0].tool_result.find("missing required argument 'text'") != std::string::npos);
    CHECK(r.stop == AgentStop::Done);
}

TEST(fenced_json_call) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("echo", scripted({
        "Sure.\n<tool_call>\n```json\n{\"name\": \"echo\", \"arguments\": {\"text\": \"fenced\"}}\n```\n</tool_call>",
        "done"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 1);
    CHECK(r.steps[0].tool_result == "fenced");
}

TEST(string_encoded_arguments) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("add", scripted({
        "<tool_call>{\"name\":\"add\",\"arguments\":\"{\\\"a\\\": 4, \\\"b\\\": 5}\"}</tool_call>",
        "<tool_call>{\"name\":\"echo\",\"parameters\":{\"text\":\"line1\\nline2\"}}</tool_call>",
        "done"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 2);
    CHECK(r.steps[0].tool_result == "9");
    CHECK(r.steps[1].tool_result == "line1\nline2");
}

TEST(gemma4_native_call) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("g4", scripted({
        "<|tool_call>call:echo{text:<|\"|>he said \"yo\"\nok<|\"|>}<tool_call|>",
        "<|tool_call>call:add{a:2,b:40}<tool_call|>",
        "<|tool_call>call:echo{text:<|\"|>never closed",
        "done"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 2);
    CHECK(r.steps[0].tool_result == "he said \"yo\"\nok");
    CHECK(r.steps[1].tool_result == "42");
    CHECK(r.steps[2].malformed);
}

TEST(parse_agent_tool_call_kinds) {
    Nishachar agent = make_agent();
    const ToolRegistry& reg = agent.tools();
    CHECK(parse_agent_tool_call("just prose, no call", reg).kind == AgentCallParse::Kind::None);
    CHECK(parse_agent_tool_call("code: int f() { return 1; }", reg).kind == AgentCallParse::Kind::None);
    // Bare JSON naming a registered tool still counts (legacy behaviour).
    auto b = parse_agent_tool_call("{\"name\":\"echo\",\"arguments\":{\"text\":\"x\"}}", reg);
    CHECK(b.kind == AgentCallParse::Kind::Call && b.call.get("text") == "x");
    CHECK(parse_agent_tool_call("<tool_call>oops</tool_call>", reg).kind ==
          AgentCallParse::Kind::Malformed);
}

TEST(loop_guard) {
    Nishachar agent = make_agent();
    AgentConfig cfg;
    cfg.max_steps = 10;
    int runs = 0;
    agent.add_tool(ToolDef{"tick", "count", {}}, [&](const ToolCall&) {
        ++runs;
        return std::string("tock");
    });
    AgentResult r = agent.run("loop",
        always("<tool_call>{\"name\":\"tick\",\"arguments\":{}}</tool_call>"), cfg);
    CHECK(r.stop == AgentStop::Loop);
    CHECK(r.steps_taken() == 4);
    CHECK(runs == 2);  // 3rd identical call is refused, 4th stops the run
    CHECK(!r.steps[2].tool_ok);
    CHECK(r.steps[2].tool_result.find("looping") != std::string::npos);
    CHECK(std::string(agent_stop_name(r.stop)) == "loop");

    // A different call in between resets the streak.
    AgentResult r2 = agent.run("alt", scripted({kEchoHi, kEchoHi, kAdd23, kEchoHi, kEchoHi, "done"}), cfg);
    CHECK(r2.stop == AgentStop::Done);
    for (int i = 0; i < 5; ++i) CHECK(r2.steps[i].tool_ok);
}

TEST(require_tool_nudge) {
    Nishachar agent = make_agent();
    // Default: a tool-less first answer is final.
    AgentResult r0 = agent.run("write code", scripted({"Here is the code: ...", "x"}));
    CHECK(r0.stop == AgentStop::Done && r0.steps_taken() == 1);

    AgentConfig cfg;
    cfg.require_tool_before_done = true;
    std::vector<std::string> prompts;
    auto inner = scripted({"Here is the code: ...", kEchoHi, "done"});
    AgentGenerator gen = [&](const std::string& p) { prompts.push_back(p); return inner(p); };
    AgentResult r = agent.run("write code", gen, cfg);
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.steps_taken() == 3);
    CHECK(r.steps[0].nudged);
    CHECK(prompts[1].find("You have not used any tool yet") != std::string::npos);
    CHECK(r.final_text == "done");

    // Nudges only once: a second tool-less answer is accepted.
    AgentResult r2 = agent.run("w", scripted({"talk", "more talk"}), cfg);
    CHECK(r2.stop == AgentStop::Done && r2.steps_taken() == 2);
    CHECK(r2.final_text == "more talk");
}

TEST(on_step_and_timing) {
    Nishachar agent = make_agent();
    AgentConfig cfg;
    std::vector<int> seen;
    cfg.on_step = [&](const AgentStep& s) { seen.push_back(s.index); };
    AgentResult r = agent.run("t", scripted({kEchoHi, "done"}), cfg);
    CHECK(seen.size() == 2 && seen[0] == 0 && seen[1] == 1);
    CHECK(r.steps[0].gen_s >= 0 && r.steps[0].tool_s >= 0);
    CHECK(r.steps[1].tool_s < 0);
    CHECK(r.report().find("[gen ") != std::string::npos);
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
