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
    // How tools and results are rendered (tools.h). Compact keeps the prompt
    // byte-identical to earlier releases; tool_format_from_model() picks the
    // model's own format (Qwen) when its GGUF template carries one.
    ToolPromptFormat tool_format = ToolPromptFormat::Compact;
    bool stop_on_tool_error = false;  // false => feed the error back and continue
};

class Nishachar;

// One run of the loop, stepped by the caller instead of run(): the caller
// generates the model's turn for prompt() and runs each tool itself, so tools
// may be asynchronous (an app's UI actions) and nothing executes inside the
// loop. run() is this session driven synchronously. Holds a reference to the
// agent's tool registry: the Nishachar must outlive the session and must not
// gain or lose tools while it runs.
class AgentSession {
public:
    enum class Event { ToolCall, Final, Empty };

    AgentSession(const Nishachar& agent, const std::string& goal, const AgentConfig& cfg);

    // The rendered prompt for the next model turn. Only while !finished() and
    // !awaiting_result().
    std::string prompt() const;

    // The model's output for prompt(). ToolCall: run pending_call(), then
    // submit_result(). Final and Empty end the session.
    Event submit_output(const std::string& out);

    // The call to run; valid while awaiting_result().
    const ToolCall& pending_call() const { return pending_; }

    // The tool's result text (or error text with ok=false), fed back to the
    // model. May end the session (max_steps, or stop_on_tool_error).
    void submit_result(const std::string& result, bool ok);

    bool awaiting_result() const { return awaiting_; }
    bool finished() const { return finished_; }
    int  steps_taken() const { return result_.steps_taken(); }

    // The report so far; final once finished().
    const AgentResult& result() const { return result_; }

private:
    const ToolRegistry&      registry_;
    AgentConfig              cfg_;
    std::vector<ChatMessage> messages_;
    AgentResult              result_;
    AgentStep                step_;      // the tool step awaiting its result
    ToolCall                 pending_;
    bool awaiting_ = false;
    bool finished_ = false;
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
