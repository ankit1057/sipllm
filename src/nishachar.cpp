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

AgentResult Nishachar::run(const std::string& goal, const AgentGenerator& gen,
                           const AgentConfig& cfg) const {
    AgentResult result;
    result.goal = goal;
    // Default outcome: if every step calls a tool and the bound is exhausted, we
    // stopped for MaxSteps. Overridden on any Done/Empty/ToolError break below.
    result.stop = AgentStop::MaxSteps;

    std::vector<ChatMessage> messages;
    messages.push_back({ChatMessage::Role::System,
                        cfg.system_preamble + "\n\n" + registry_.system_prompt_block(),
                        ""});
    messages.push_back({ChatMessage::Role::User, goal, ""});

    for (int step = 0; step < cfg.max_steps; ++step) {
        std::string prompt = render_chat(messages, registry_, cfg.style,
                                         /*add_gen_prompt=*/true);
        std::string out = gen(prompt);

        if (out.empty()) {
            result.stop = AgentStop::Empty;
            break;
        }

        // Fresh parser per step, default markers (do NOT call set_markers). It
        // rejects unregistered tools -> such output is treated as a final answer.
        ToolParser parser(registry_);
        parser.feed(out);

        if (parser.state() == ToolParser::State::Done) {
            const ToolCall call = parser.parsed_call();
            bool ok = true;
            std::string tool_result = dispatch(call, ok);

            AgentStep s;
            s.index = step;
            s.model_output = out;
            s.had_tool_call = true;
            s.tool_name = call.name;
            s.tool_args_json = call.raw_json;
            s.tool_result = tool_result;
            s.tool_ok = ok;
            result.steps.push_back(std::move(s));

            messages.push_back({ChatMessage::Role::Assistant, out, ""});
            messages.push_back({ChatMessage::Role::Tool, tool_result, ""});

            if (!ok && cfg.stop_on_tool_error) {
                result.stop = AgentStop::ToolError;
                break;
            }
            continue;
        }

        // No tool call -> this is the final answer.
        AgentStep s;
        s.index = step;
        s.model_output = out;
        s.had_tool_call = false;
        result.steps.push_back(std::move(s));
        result.final_text = out;
        result.stop = AgentStop::Done;
        break;
    }

    return result;
}

} // namespace llm
