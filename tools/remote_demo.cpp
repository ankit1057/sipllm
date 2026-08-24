#include "llm/remote_weight_source.h"
#include "llm/runtime.h"
#include <iostream>

using namespace llm;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <host> <port> <prompt>\n";
        return 1;
    }
    std::string host = argv[1];
    int port = std::stoi(argv[2]);
    std::string prompt = argv[3];

    try {
        std::cout << "Connecting to " << host << ":" << port << "...\n";
        auto src = std::make_unique<RemoteWeightSource>(host, port);
        
        // Pass to Runtime with a ram budget to limit memory (e.g. 512MB limit)
        // Set ram_budget_total = 512*1024*1024
        LayerLoader::Options opt;
        // force streaming for remote layers
        Runtime rt(std::move(src), opt, /*max_ctx*/ 512, /*threads*/ 4, /*ram_budget_total*/ 512*1024*1024, /*force_budget*/ true);
        
        std::cout << "Generating...\n";
        SamplerConfig scfg;
        GenStats stats;
        
        auto on_token = [](const std::string& piece, int64_t id) {
            std::cout << piece << std::flush;
            return true;
        };
        
        rt.generate(prompt, 64, scfg, on_token, &stats);
        std::cout << "\n\nStats:\n"
                  << "Tokens/sec: " << stats.decode_tok_s << "\n"
                  << "Bytes read: " << stats.bytes_read << "\n"
                  << "Peak RSS budget: 512MB\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
