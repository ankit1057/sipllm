// nishachar_demo — drive the Nishachar autonomous loop on a real SipLLM model.
//
//   nishachar_demo <model_path> "<goal>" [--max-steps N]
//
// Loads a model, registers three deterministic in-process tools (calc/echo/
// upper), and runs the goal->plan->act->verify loop with a greedy generator.
// The loop mechanism is exercised regardless of model quality: a trained model
// will emit <tool_call> blocks that the loop dispatches; a toy model simply
// answers immediately, which still verifies the plumbing end to end.
#include "llm/nishachar.h"
#include "llm/runtime.h"
#include "llm/sampler.h"
#include "llm/tools.h"

#include <cctype>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace llm;

// Parse a simple "<int> <op> <int>" expression (ops: + - * /), integer math.
// Throws std::runtime_error on a malformed expression or divide-by-zero.
static std::string eval_calc(const std::string& expr) {
    std::istringstream in(expr);
    long a = 0, b = 0;
    std::string op;
    if (!(in >> a >> op >> b))
        throw std::runtime_error("bad expression: expected '<int> <op> <int>'");
    if (op.size() != 1 || std::string("+-*/").find(op) == std::string::npos)
        throw std::runtime_error("unknown operator '" + op + "'");
    long r = 0;
    switch (op[0]) {
        case '+': r = a + b; break;
        case '-': r = a - b; break;
        case '*': r = a * b; break;
        case '/':
            if (b == 0) throw std::runtime_error("division by zero");
            r = a / b;
            break;
    }
    return std::to_string(r);
}

static std::string to_upper_ascii(const std::string& s) {
    std::string out = s;
    for (char& c : out) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: nishachar_demo <model_path> \"<goal>\" [--max-steps N]\n");
        return 2;
    }
    std::string model_path = argv[1];
    std::string goal = argv[2];
    int max_steps = 8;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--max-steps" && i + 1 < argc) {
            max_steps = std::stoi(argv[++i]);
        } else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            fprintf(stderr, "usage: nishachar_demo <model_path> \"<goal>\" [--max-steps N]\n");
            return 2;
        }
    }

    try {
        // Build the runtime, mirroring dump_logits.cpp / bench.cpp.
        auto src = open_model(model_path);
        LayerLoader::Options opt;
        opt.residency = Residency::Quantized;
        Runtime rt(std::move(src), opt);

        // Greedy / deterministic sampling: temperature 0 => argmax (sampler.h).
        SamplerConfig scfg{0.f};

        // Register the three deterministic tools with advertised schemas so
        // registry.system_prompt_block() describes them to the model.
        Nishachar agent;

        ToolDef calc_def;
        calc_def.name = "calc";
        calc_def.description = "Evaluate a simple integer arithmetic expression of the "
                              "form '<int> <op> <int>' where op is one of + - * /.";
        calc_def.params.push_back(ToolParam{"expr", ToolParamType::String, true,
                                            "the expression, e.g. '6 * 7'", ""});
        agent.add_tool(calc_def, [](const ToolCall& call) -> std::string {
            return eval_calc(call.get("expr"));
        });

        ToolDef echo_def;
        echo_def.name = "echo";
        echo_def.description = "Return the given text verbatim.";
        echo_def.params.push_back(ToolParam{"text", ToolParamType::String, true,
                                            "the text to echo back", ""});
        agent.add_tool(echo_def, [](const ToolCall& call) -> std::string {
            return call.get("text");
        });

        ToolDef upper_def;
        upper_def.name = "upper";
        upper_def.description = "Return the given text uppercased (ASCII).";
        upper_def.params.push_back(ToolParam{"text", ToolParamType::String, true,
                                             "the text to uppercase", ""});
        agent.add_tool(upper_def, [](const ToolCall& call) -> std::string {
            return to_upper_ascii(call.get("text"));
        });

        AgentConfig cfg;
        cfg.max_steps = max_steps;

        // The generator: reset conversation KV each call (the loop rebuilds the
        // full chat prompt every step), then greedily continue the prompt.
        AgentGenerator gen = [&](const std::string& prompt) -> std::string {
            rt.reset();
            GenStats stats;
            return rt.generate(prompt, cfg.max_new_tokens, scfg, nullptr, &stats);
        };

        printf("# model: %s\n# goal:  %s\n# max_steps: %d\n\n",
               model_path.c_str(), goal.c_str(), cfg.max_steps);

        AgentResult result = agent.run(goal, gen, cfg);

        fputs(result.report().c_str(), stdout);
        printf("\n================ FINAL ANSWER ================\n%s\n",
               result.final_text.c_str());
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
