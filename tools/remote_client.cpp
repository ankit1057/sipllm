// remote_client.cpp — Production streaming layer inference client.
//
// usage: remote_client <host> <port> [options]
//
// Connects to a remote weight server (remote_server), lazily streams layers over
// TCP under a local bounded RAM budget, and runs local compute inference.
#include "llm/remote_weight_source.h"
#include "llm/runtime.h"
#include "llm/sampler.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

using namespace llm;

static void print_usage(const char* prog) {
    fprintf(stderr,
            "usage: %s <host> <port> [options]\n\n"
            "arguments:\n"
            "  <host>              Remote server hostname or IP\n"
            "  <port>              Remote server port\n\n"
            "options:\n"
            "  --prompt, -p TEXT   Prompt to run (default: interactive chat REPL)\n"
            "  --tokens, -n N      Maximum tokens to generate (default: 128)\n"
            "  --temp T            Sampling temperature (default: 0.7, 0 for greedy)\n"
            "  --budget-mb N       RAM budget in MB for layer streaming cache (default: 512)\n"
            "  --threads N         Worker thread count (default: 4)\n"
            "  --ctx N             Context length limit (default: 512)\n"
            "  --interactive, -i   Force interactive multi-turn REPL mode\n"
            "  --help, -h          Show this help message\n",
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

    std::string host = argv[1];
    int port = 0;
    try {
        port = std::stoi(argv[2]);
    } catch (...) {
        fprintf(stderr, "error: invalid port '%s'\n", argv[2]);
        return 2;
    }

    std::string prompt;
    int max_tokens = 128;
    float temp = 0.7f;
    int budget_mb = 512;
    int threads = 4;
    int ctx = 512;
    bool interactive = false;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if ((a == "--prompt" || a == "-p") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((a == "--tokens" || a == "-n") && i + 1 < argc) {
            max_tokens = std::stoi(argv[++i]);
        } else if (a == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else if (a == "--budget-mb" && i + 1 < argc) {
            budget_mb = std::stoi(argv[++i]);
        } else if (a == "--threads" && i + 1 < argc) {
            threads = std::stoi(argv[++i]);
        } else if (a == "--ctx" && i + 1 < argc) {
            ctx = std::stoi(argv[++i]);
        } else if (a == "--interactive" || a == "-i") {
            interactive = true;
        } else {
            fprintf(stderr, "unknown or incomplete argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return 2;
        }
    }

    if (prompt.empty()) {
        interactive = true;
    }

    printf("\n============================================================\n");
    printf("         SipLLM — Production Remote Streaming Client        \n");
    printf("============================================================\n");
    printf("  Target Server:  %s:%d\n", host.c_str(), port);
    printf("  Local Budget:   %d MB\n", budget_mb);
    printf("  Worker Threads: %d\n", threads);
    printf("  Context Limit:  %d tokens\n", ctx);
    printf("============================================================\n\n");

    try {
        printf("Connecting to remote server at %s:%d...\n", host.c_str(), port);
        auto src = std::make_unique<RemoteWeightSource>(host, port);

        LayerLoader::Options opt;
        opt.residency = Residency::Quantized;
        uint64_t ram_budget_bytes = static_cast<uint64_t>(budget_mb) * 1024 * 1024;

        Runtime rt(std::move(src), opt, ctx, threads, ram_budget_bytes, /*force_budget=*/true);

        const auto& c = rt.config();
        printf("[Connected] Model: %s | Layers: %lld | Dim: %lld | Vocab: %lld\n\n",
               c.arch.c_str(), (long long)c.n_layers, (long long)c.dim, (long long)c.vocab_size);

        SamplerConfig scfg;
        scfg.temperature = temp;

        auto on_token = [](const std::string& piece, int64_t) {
            std::cout << piece << std::flush;
            return true;
        };

        if (!interactive) {
            printf("\033[1;34m[Prompt]:\033[0m %s\n", prompt.c_str());
            printf("\033[1;32m[Generating]:\033[0m ");
            GenStats stats;
            rt.generate(prompt, max_tokens, scfg, on_token, &stats);
            std::cout << "\n\n";

            printf("------------------------------------------------------------\n");
            printf("  Prompt Tokens:       %d\n", stats.prompt_tokens);
            printf("  Generated Tokens:    %d\n", stats.gen_tokens);
            printf("  Prefill Speed:       %.2f tok/s (TTFT: %.3fs)\n", stats.prefill_tok_s, stats.ttft_s);
            printf("  Decode Speed:        %.2f tok/s\n", stats.decode_tok_s);
            printf("  Remote Transferred:  %.2f MB\n", stats.bytes_read / (1024.0 * 1024.0));
            printf("  Local RAM Budget:    %d MB\n", budget_mb);
            printf("------------------------------------------------------------\n");
            return 0;
        }

        // Interactive REPL Mode
        printf("Entering interactive session. Type '/exit' or '/quit' to leave, '/reset' to clear context.\n\n");

        std::string current_history;
        while (true) {
            std::cout << "\033[1;36m>>> \033[0m" << std::flush;
            std::string line;
            if (!std::getline(std::cin, line)) {
                std::cout << "\n";
                break;
            }

            if (line == "/exit" || line == "/quit") {
                break;
            }
            if (line == "/reset") {
                rt.reset();
                current_history.clear();
                printf("[Context reset]\n\n");
                continue;
            }
            if (line.empty()) {
                continue;
            }

            std::string cur_prompt = current_history.empty() ? line : current_history + "\nUser: " + line + "\nAssistant:";
            rt.reset();
            GenStats stats;
            std::string reply = rt.generate(cur_prompt, max_tokens, scfg, on_token, &stats);
            std::cout << "\n";
            printf("\033[90m[%d tok, %.1f tok/s, %.1f MB streamed]\033[0m\n\n",
                   stats.gen_tokens, stats.decode_tok_s, stats.bytes_read / (1024.0 * 1024.0));

            current_history = cur_prompt + " " + reply;
        }

        printf("Goodbye!\n");
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "\nremote client error: %s\n", e.what());
        return 1;
    }
}
