// nishachar.h — Nishachar Path A: the autonomous goal->plan->act->verify loop.
//
// Nishachar (#58) is the runtime's reference autonomous consumer: input a GOAL,
// output completed work. Path A is the C++ *loop foundation* built on the
// tool-calling core (tools.h): it renders a chat, asks a text generator to act,
// parses any tool call, dispatches it to a registered handler, feeds the result
// back, and repeats until the model answers without calling a tool (or a step
// bound is hit). The generator is an abstraction (AgentGenerator) so the loop is
// exercised deterministically in tests with a scripted generator and bound to a
// real SipLLM model (Runtime::generate) in production.
//
// Scope note (honest): Path A is the loop mechanism + tool dispatch + a
// structured run report -- one slice of the #58 deliverable. The richer
// compile/test/benchmark verification and the fs/shell/git/compiler capability
// layer are the Plugin phase (#51) and later; this header deliberately does not
// pretend to be the full autonomous coding agent. It is model-agnostic: it never
// includes runtime.h, so the loop links without the engine.
#pragma once

#include "llm/tools.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llm {

// Text-completion backend: given a fully-rendered prompt, return the model's
// generated continuation. Production binds this to Runtime::generate; tests
// inject a scripted function.
using AgentGenerator = std::function<std::string(const std::string& prompt)>;

// Executes one tool call and returns the result text fed back to the model.
// A handler MAY throw std::exception to signal failure; the loop catches it and
// records a failed step (see AgentConfig::stop_on_tool_error).
using ToolHandler = std::function<std::string(const ToolCall&)>;

// Why the loop stopped.
enum class AgentStop {
    Done,       // model produced a final answer (no tool call)
    MaxSteps,   // hit AgentConfig::max_steps without a final answer
    ToolError,  // a handler failed and stop_on_tool_error was set
    Empty,      // generator returned empty text -> no progress possible
};

const char* agent_stop_name(AgentStop s);

// One iteration of the loop.
struct AgentStep {
    int         index = 0;
    std::string model_output;    // raw generation this step
    bool        had_tool_call = false;
    std::string tool_name;       // set when had_tool_call
    std::string tool_args_json;  // the call's raw JSON (for the report)
    std::string tool_result;     // handler output, or the error text if !tool_ok
    bool        tool_ok = true;  // false if the handler failed
};

// Structured run report (a slice of the #58 "structured run report" deliverable).
struct AgentResult {
    std::string            goal;
    std::string            final_text;  // final answer; empty unless stop == Done
    std::vector<AgentStep> steps;
    AgentStop              stop = AgentStop::Done;

    int steps_taken() const { return static_cast<int>(steps.size()); }
    int tool_calls() const;      // number of steps that dispatched a tool
    std::string report() const;  // human-readable structured summary
};

struct AgentConfig {
    int max_steps      = 8;    // hard bound on loop iterations
    int max_new_tokens = 256;  // advisory; forwarded to a Runtime-backed generator

    // Agent preamble prepended (before the tool-schema block) to the system
    // message. Instructs the model on the <tool_call>{...}</tool_call> protocol.
    std::string system_preamble =
        "You are Nishachar, an autonomous worker. Achieve the user's goal step "
        "by step. To use a tool, emit exactly one "
        "<tool_call>{\"name\":\"<tool>\",\"arguments\":{...}}</tool_call> and "
        "stop. When the goal is complete, reply with the final answer and no "
        "tool call.";

    ChatTemplateStyle style = ChatTemplateStyle::ChatML;
    bool stop_on_tool_error = false;  // false => feed the error back and continue
};

// The loop. Model-agnostic: register tools (advertised schema + C++ executor),
// then run(goal, generator). Re-usable across runs; holds no per-run state.
class Nishachar {
public:
    // Bind a tool's advertised schema to its executor. Re-adding a name
    // overwrites both (matching ToolRegistry semantics).
    void add_tool(ToolDef def, ToolHandler handler);

    const ToolRegistry& tools() const { return registry_; }
    bool has_tool(const std::string& name) const;

    // Run goal -> plan -> act -> verify until a final answer or max_steps.
    AgentResult run(const std::string& goal, const AgentGenerator& gen,
                    const AgentConfig& cfg = AgentConfig{}) const;

private:
    // Look up and invoke the handler for `call`; sets ok=false (and returns an
    // error string) when the tool is unknown or the handler throws.
    std::string dispatch(const ToolCall& call, bool& ok) const;

    ToolRegistry registry_;
    std::unordered_map<std::string, ToolHandler> handlers_;
};

} // namespace llm
