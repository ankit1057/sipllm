// nishachar.cpp — Nishachar autonomous agent CLI.
//
// usage: nishachar <model_path> "<goal>" [--max-steps N] [--workdir DIR]
//
// Nishachar (#58) is SipLLM's reference autonomous worker: given a goal, it
// iteratively plans, acts via system tools (read_file, write_file, list_dir,
// bash, grep_search), observes results, and verifies completion until a final
// answer is produced.
#include "llm/nishachar.h"
#include "llm/runtime.h"
#include "llm/sampler.h"
#include "llm/tools.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace llm;

static void print_usage(const char* prog) {
    fprintf(stderr,
            "usage: %s <model_path> \"<goal>\" [options]\n\n"
            "arguments:\n"
            "  <model_path>     Path to model file (.gguf, .sipr, safetensors)\n"
            "  \"<goal>\"         High-level goal for Nishachar to achieve\n\n"
            "options:\n"
            "  --max-steps N    Maximum autonomous loop steps (default: 10)\n"
            "  --workdir DIR    Working directory for tools (default: .)\n"
            "  --threads N      Inference thread count (default: auto)\n"
            "  --ctx N          Context length limit (default: model default)\n"
            "  --temp T         Sampling temperature (default: 0.0 for deterministic tools)\n"
            "  --help, -h       Show this help message\n",
            prog);
}

int main(int argc, char** argv) {
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        print_usage(argv[0]);
        return 0;
    }
    if (argc < 3) {
        print_usage(argv[0]);
        return 2;
    }

    std::string model_path = argv[1];
    std::string goal = argv[2];
    int max_steps = 10;
    std::string workdir = ".";
    int threads = 0;
    int ctx = 0;
    float temp = 0.0f;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--help" || a == "-h")) {
            print_usage(argv[0]);
            return 0;
        } else if (a == "--max-steps" && i + 1 < argc) {
            max_steps = std::stoi(argv[++i]);
        } else if (a == "--workdir" && i + 1 < argc) {
            workdir = argv[++i];
        } else if (a == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (a == "--ctx" && i + 1 < argc) {
            ctx = std::stoi(argv[++i]);
        } else if (a == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    std::error_code ec;
    if (!std::filesystem::exists(model_path, ec)) {
        fprintf(stderr, "error: model file '%s' does not exist\n", model_path.c_str());
        return 1;
    }

    if (!std::filesystem::exists(workdir, ec)) {
        fprintf(stderr, "error: workdir '%s' does not exist\n", workdir.c_str());
        return 1;
    }

    printf("\n============================================================\n");
    printf("         NISHACHAR — Autonomous Agent Platform             \n");
    printf("============================================================\n");
    printf("  Model:     %s\n", model_path.c_str());
    printf("  Goal:      %s\n", goal.c_str());
    printf("  Workdir:   %s\n", workdir.c_str());
    printf("  Max Steps: %d\n", max_steps);
    printf("============================================================\n\n");

    try {
        // Load model and initialize runtime
        auto src = open_model(model_path);
        LayerLoader::Options opt;
        opt.residency = Residency::Quantized;
        Runtime rt(std::move(src), opt, ctx, threads);

        printf("[Model Loaded] Arch: %s | Layers: %lld | Dim: %lld\n",
               rt.config().arch.c_str(),
               (long long)rt.config().n_layers,
               (long long)rt.config().dim);

        // Configure agent and tools
        Nishachar agent;

        // Register real system tools with live progress logging
        register_system_tools([&](ToolDef def, ToolHandler handler) {
            std::string tool_name = def.name;
            agent.add_tool(std::move(def), [handler = std::move(handler), tool_name](const ToolCall& call) -> std::string {
                printf("\n\033[1;36m>>> [Tool Execution: %s]\033[0m\n", tool_name.c_str());
                for (const auto& arg : call.args) {
                    std::string v = arg.value;
                    if (v.size() > 100) v = v.substr(0, 97) + "...";
                    for (char& c : v) if (c == '\n' || c == '\r') c = ' ';
                    printf("    \033[33m%s\033[0m: %s\n", arg.key.c_str(), v.c_str());
                }
                std::string result = handler(call);
                std::string preview = result;
                if (preview.size() > 240) {
                    preview = preview.substr(0, 237) + "...";
                }
                for (char& c : preview) if (c == '\n' || c == '\r') c = ' ';
                printf("    \033[1;32m[Result (%zu bytes)]\033[0m %s\n\n", result.size(), preview.c_str());
                return result;
            });
        }, workdir);

        AgentConfig cfg;
        cfg.max_steps = max_steps;
        cfg.max_new_tokens = 512;
        cfg.style = style_from_model(rt.config());

        SamplerConfig scfg;
        scfg.temperature = temp;

        int step_count = 0;
        AgentGenerator gen = [&](const std::string& prompt) -> std::string {
            ++step_count;
            printf("\033[1;35m--- [Step %d / %d: Generating Action] ---\033[0m\n", step_count, cfg.max_steps);
            rt.reset();
            GenStats stats;
            std::string out = rt.generate(prompt, cfg.max_new_tokens, scfg,
                                          [](const std::string& piece, int64_t) {
                                              fputs(piece.c_str(), stdout);
                                              fflush(stdout);
                                              return true;
                                          }, &stats);
            printf("\n");
            return out;
        };

        printf("\033[1;32m[Starting Nishachar Execution Loop]\033[0m\n\n");
        AgentResult result = agent.run(goal, gen, cfg);

        printf("\n============================================================\n");
        printf("                 NISHACHAR EXECUTION REPORT                 \n");
        printf("============================================================\n");
        printf("%s\n", result.report().c_str());

        printf("============================================================\n");
        printf("                        FINAL OUTCOME                       \n");
        printf("============================================================\n");
        printf("Status: %s\n", agent_stop_name(result.stop));
        if (!result.final_text.empty()) {
            printf("\n%s\n", result.final_text.c_str());
        } else {
            printf("(No final textual response generated)\n");
        }
        printf("============================================================\n");

        return (result.stop == AgentStop::Done) ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "\nfatal error: %s\n", e.what());
        return 1;
    }
}
