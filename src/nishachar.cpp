// nishachar.cpp — Nishachar Path A: the autonomous goal->plan->act->verify loop.
//
// Implements the loop foundation declared in nishachar.h over the tool-calling
// core (tools.h): render chat -> generate -> parse tool call -> dispatch handler
// -> feed result back -> repeat until a final (no-tool) answer or a step bound.
// Model-agnostic: never touches the engine (runtime.h), so it links standalone.
#include "llm/nishachar.h"

#include <exception>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace llm {

const char* agent_stop_name(AgentStop s) {
    switch (s) {
        case AgentStop::Done:      return "done";
        case AgentStop::MaxSteps:  return "max_steps";
        case AgentStop::ToolError: return "tool_error";
        case AgentStop::Empty:     return "empty";
    }
    return "done";
}

// ============================================================================
// AgentResult
// ============================================================================
int AgentResult::tool_calls() const {
    int n = 0;
    for (const auto& s : steps)
        if (s.had_tool_call) ++n;
    return n;
}

// Collapse newlines and cap to `n` chars (with an ellipsis) so each step renders
// on a single readable line in the report.
static std::string truncate_line(const std::string& s, size_t n) {
    std::string t;
    t.reserve(s.size());
    for (char c : s) t += (c == '\n' || c == '\r') ? ' ' : c;
    if (t.size() > n) return t.substr(0, n) + "...";
    return t;
}

std::string AgentResult::report() const {
    std::ostringstream os;
    os << "goal: " << goal << "\n";
    os << "stop=" << agent_stop_name(stop) << "\n";
    os << "steps=" << steps_taken() << "\n";
    os << "tool_calls=" << tool_calls() << "\n";
    for (const auto& s : steps) {
        os << "#" << s.index << " "
           << (s.had_tool_call ? s.tool_name : std::string("final"))
           << " ok=" << (s.tool_ok ? 1 : 0) << " :: "
           << truncate_line(s.had_tool_call ? s.tool_result : s.model_output, 120)
           << "\n";
    }
    os << "final: " << final_text;
    return os.str();
}

// ============================================================================
// Nishachar
// ============================================================================
void Nishachar::add_tool(ToolDef def, ToolHandler handler) {
    // Store the handler first (keyed by name) before `def` is moved away; a
    // repeat name overwrites both, matching ToolRegistry semantics.
    handlers_[def.name] = std::move(handler);
    registry_.register_tool(std::move(def));
}

bool Nishachar::has_tool(const std::string& name) const {
    return registry_.find(name) != nullptr;
}

std::string Nishachar::dispatch(const ToolCall& call, bool& ok) const {
    if (!has_tool(call.name)) {
        ok = false;
        return "error: unknown tool '" + call.name + "'";
    }
    auto it = handlers_.find(call.name);
    if (it == handlers_.end()) {
        ok = false;
        return "error: unknown tool '" + call.name + "'";
    }
    try {
        std::string result = it->second(call);
        ok = true;
        return result;
    } catch (const std::exception& e) {
        ok = false;
        return std::string("error: ") + e.what();
    }
}

// ============================================================================
// AgentSession
// ============================================================================
AgentSession::AgentSession(const Nishachar& agent, const std::string& goal,
                           const AgentConfig& cfg)
    : registry_(agent.tools()), cfg_(cfg) {
    result_.goal = goal;
    // Default outcome: if every step calls a tool and the bound is exhausted, we
    // stopped for MaxSteps. Overridden on any Done/Empty/ToolError end below.
    result_.stop = AgentStop::MaxSteps;
    // render_chat() appends the registry's tool block to the system message, so
    // the preamble goes in alone (adding the block here rendered it twice).
    messages_.push_back({ChatMessage::Role::System, cfg_.system_preamble, ""});
    messages_.push_back({ChatMessage::Role::User, goal, ""});
    finished_ = cfg_.max_steps <= 0;
}

std::string AgentSession::prompt() const {
    return render_chat(messages_, registry_, cfg_.style, /*add_gen_prompt=*/true,
                       cfg_.tool_format);
}

AgentSession::Event AgentSession::submit_output(const std::string& out) {
    const int index = result_.steps_taken();
    if (out.empty()) {
        result_.stop = AgentStop::Empty;
        finished_ = true;
        return Event::Empty;
    }

    // Fresh parser per step, default markers (do NOT call set_markers). It
    // rejects unregistered tools -> such output is treated as a final answer.
    ToolParser parser(registry_);
    parser.feed(out);

    if (parser.state() == ToolParser::State::Done) {
        pending_ = parser.parsed_call();
        step_ = AgentStep{};
        step_.index = index;
        step_.model_output = out;
        step_.had_tool_call = true;
        step_.tool_name = pending_.name;
        step_.tool_args_json = pending_.raw_json;
        awaiting_ = true;
        return Event::ToolCall;
    }

    // No tool call -> this is the final answer.
    AgentStep s;
    s.index = index;
    s.model_output = out;
    s.had_tool_call = false;
    result_.steps.push_back(std::move(s));
    result_.final_text = out;
    result_.stop = AgentStop::Done;
    finished_ = true;
    return Event::Final;
}

void AgentSession::submit_result(const std::string& tool_result, bool ok) {
    step_.tool_result = tool_result;
    step_.tool_ok = ok;
    messages_.push_back({ChatMessage::Role::Assistant, step_.model_output, ""});
    messages_.push_back({ChatMessage::Role::Tool, tool_result, ""});
    result_.steps.push_back(std::move(step_));
    step_ = AgentStep{};
    pending_ = ToolCall{};
    awaiting_ = false;
    if (!ok && cfg_.stop_on_tool_error) {
        result_.stop = AgentStop::ToolError;
        finished_ = true;
    } else if (result_.steps_taken() >= cfg_.max_steps) {
        finished_ = true;   // stop stays MaxSteps
    }
}

AgentResult Nishachar::run(const std::string& goal, const AgentGenerator& gen,
                           const AgentConfig& cfg) const {
    AgentSession session(*this, goal, cfg);
    while (!session.finished()) {
        if (session.submit_output(gen(session.prompt())) != AgentSession::Event::ToolCall)
            continue;   // Final or Empty: the session has finished
        bool ok = true;
        std::string tool_result = dispatch(session.pending_call(), ok);
        session.submit_result(tool_result, ok);
    }
    return session.result();
}

} // namespace llm
