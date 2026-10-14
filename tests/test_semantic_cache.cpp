#include "llm/runtime.h"
#include "llm/semantic_cache.h"
#include "llm/toy_model.h"
#include <iostream>

using namespace llm;

int main() {
    try {
        ToyConfig tc; tc.n_layers=2; tc.dim=32; tc.n_heads=2; tc.n_kv_heads=2; tc.ffn_dim=64; tc.vocab_size=256;
        write_toy_model("st_kosh.llmw", tc);
        
        auto src = open_model("st_kosh.llmw", false);
        LayerLoader::Options opt;
        Runtime rt(std::move(src), opt, 1024, 1, 0, false);
        
        rt.enable_semantic_cache(256 * 1024 * 1024);
        
        SamplerConfig scfg;
        scfg.temperature = 0.0f; // greedy for determinism
        
        std::string prompt = "Hello";
        
        std::cout << "--- RUN 1 (Cold Cache) ---\n";
        GenStats s1;
        std::string out1 = rt.generate(prompt, 5, scfg, nullptr, &s1);
        std::cout << "TTFT: " << s1.ttft_s << " s\n";
        std::cout << "Kosh Hits: " << s1.semantic_cache_hits << "\n";
        std::cout << "Output: " << out1 << "\n\n";
        
        rt.reset(); // Simulate a new HTTP request
        
        std::cout << "--- RUN 2 (Warm Cache) ---\n";
        GenStats s2;
        std::string out2 = rt.generate(prompt, 5, scfg, nullptr, &s2);
        std::cout << "TTFT: " << s2.ttft_s << " s\n";
        std::cout << "Kosh Hits: " << s2.semantic_cache_hits << "\n";
        std::cout << "Output: " << out2 << "\n\n";
        
        if (s2.semantic_cache_hits > 0) {
            std::cout << "SUCCESS! Kosh semantic cache worked perfectly! Hit " << s2.semantic_cache_hits << " tokens.\n";
            return 0;
        } else {
            std::cout << "FAILED! Hit tokens: " << s2.semantic_cache_hits << "\n";
            return 1;
        }
        
    } catch(std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
        return 1;
    }
}
