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
#include <map>
#include <string>
#include <vector>

using namespace llm;

// Minimal JSON string literal encoder for the transcript (zero-dep).
static std::string json_str(const std::string& v) {
    std::string o = "\"";
    for (unsigned char c : v) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
                else o += (char)c;
        }
    }
    return o + "\"";
}

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
            "  --fast           int8-activation SIMD kernels (Q8_0, Q4_K/Q5_K/Q6_K)\n"
            "  --ram-budget B   total peak-RSS target, e.g. 14G (pins layers resident)\n"
            "  --max-new N      max tokens generated per step (default: 2048)\n"
            "  --tool-timeout S kill a bash command after S seconds (default: 600)\n"
            "  --tool-cap N     max chars of tool output fed back (default: 6000)\n"
            "  --log FILE       append one JSON line of stats per step\n"
            "  --transcript F   write the full conversation as JSON lines (one per step)\n"
            "  --require-tool   nudge once if the model answers before using any tool\n"
            "  --help, -h       Show this help message\n",
            prog);
}


static void print_result(const AgentResult& result) {
    printf("\n============================================================\n");
    printf("                 NISHACHAR EXECUTION REPORT                 \n");
    printf("============================================================\n");
    printf("%s\n", result.report().c_str());
    printf("============================================================\n");
    printf("                        FINAL OUTCOME                       \n");
    printf("============================================================\n");
    printf("Status: %s\n", agent_stop_name(result.stop));
    if (!result.final_text.empty()) printf("\n%s\n", result.final_text.c_str());
    else printf("(No final textual response generated)\n");
    printf("============================================================\n");
}

// ---- Gemma 4 native tool calling --------------------------------------------
// Gemma 4 is trained on its own call DSL and ignores a <tool_call>{json}
// instruction. Native protocol (from the GGUF chat template):
//   system turn:  <|tool>declaration:NAME{description:<|"|>..<|"|>,parameters:{..}}<tool|>
//   model call:   <|tool_call>call:NAME{key:<|"|>value<|"|>,flag:true}<tool_call|>
//   tool result:  <|tool_response>response:NAME{value:<|"|>..<|"|>}<tool_response|>
// and the model keeps generating in the SAME turn after a response. <|"|> is a
// string delimiter token, so code needs no JSON escaping — far more reliable
// for write_file payloads. The whole run is one append-only transcript: each
// step prefills only the tool response.
static const char* kQ = "<|\"|>";

static std::string native_declaration(const ToolDef& d) {
    std::string o = "<|tool>declaration:" + d.name + "{description:" + kQ + d.description + kQ;
    if (!d.params.empty()) {
        o += ",parameters:{properties:{";
        for (size_t i = 0; i < d.params.size(); ++i) {
            const ToolParam& p = d.params[i];
            const char* ty = p.type == ToolParamType::Int ? "INTEGER" : p.type == ToolParamType::Float ? "NUMBER"
                           : p.type == ToolParamType::Bool ? "BOOLEAN" : p.type == ToolParamType::StringArray ? "ARRAY" : "STRING";
            if (i) o += ",";
            o += p.name + ":{description:" + kQ + p.description + kQ + ",type:" + kQ + ty + kQ + "}";
        }
        o += "},required:[";
        bool first = true;
        for (const ToolParam& p : d.params)
            if (p.required) { if (!first) o += ","; o += kQ + p.name + kQ; first = false; }
        o += std::string("],type:") + kQ + "OBJECT" + kQ + "}";
    }
    return o + "}<tool|>";
}

// Parse "call:NAME{k:<|"|>v<|"|>,n:3,b:true}" (text after <|tool_call>).
static bool parse_native_call(const std::string& s, ToolCall& call, std::string& err) {
    size_t p = s.find("call:");
    if (p == std::string::npos) { err = "missing 'call:'"; return false; }
    p += 5;
    size_t brace = s.find('{', p);
    if (brace == std::string::npos) { err = "missing '{'"; return false; }
    call.name = s.substr(p, brace - p);
    while (!call.name.empty() && std::isspace((unsigned char)call.name.back())) call.name.pop_back();
    const std::string q = kQ;
    size_t i = brace + 1;
    auto skip = [&] { while (i < s.size() && (std::isspace((unsigned char)s[i]) || s[i] == ',')) ++i; };
    for (;;) {
        skip();
        if (i >= s.size()) { err = "unterminated arguments"; return false; }
        if (s[i] == '}') break;
        size_t colon = s.find(':', i);
        if (colon == std::string::npos) { err = "missing ':' after key"; return false; }
        std::string key = s.substr(i, colon - i);
        if (key.size() >= 2 * q.size() && key.compare(0, q.size(), q) == 0) key = key.substr(q.size(), key.size() - 2 * q.size());
        while (!key.empty() && std::isspace((unsigned char)key.back())) key.pop_back();
        i = colon + 1;
        while (i < s.size() && std::isspace((unsigned char)s[i])) ++i;
        std::string val;
        if (s.compare(i, q.size(), q) == 0) {
            size_t end = s.find(q, i + q.size());
            if (end == std::string::npos) { err = "unterminated string for '" + key + "'"; return false; }
            val = s.substr(i + q.size(), end - i - q.size());
            i = end + q.size();
        } else {
            int depth = 0; size_t j = i;
            for (; j < s.size(); ++j) {
                if (s[j] == '{' || s[j] == '[') ++depth;
                else if (s[j] == '}' || s[j] == ']') { if (depth == 0) break; --depth; }
                else if (s[j] == ',' && depth == 0) break;
            }
            val = s.substr(i, j - i);
            while (!val.empty() && std::isspace((unsigned char)val.back())) val.pop_back();
            i = j;
        }
        call.args.push_back({key, val});
    }
    call.raw_json = s.substr(brace, i + 1 - brace);
    return true;
}

// Degeneration ("rut") detector: the tail of the output is one short unit
// repeated many times (e.g. "\x2d\x2e\x20" forever under greedy decoding).
// Returns the repeating unit, or "" if the tail looks healthy.
static std::string rut_unit(const std::string& acc) {
    for (size_t p = 1; p <= 24; ++p) {
        const size_t need = std::max<size_t>(96, p * 10);
        if (acc.size() < need) continue;
        const char* t = acc.data() + acc.size() - need;
        bool periodic = true;
        for (size_t i = p; i < need && periodic; ++i) periodic = t[i] == t[i - p];
        if (periodic) {
            std::string unit(t, p);
            // a long run of one whitespace/punctuation char is legitimate (padding, rules)
            if (p == 1 && (unit == " " || unit == "\n" || unit == "-" || unit == "=" || unit == "/")) continue;
            return unit;
        }
    }
    return "";
}

static AgentResult run_gemma4_native(Runtime& rt, const SamplerConfig& scfg, const AgentConfig& cfg,
                                     const std::string& goal, const std::vector<ToolDef>& defs,
                                     const std::map<std::string, ToolHandler>& handlers, FILE* logf) {
    AgentResult result;
    result.goal = goal;
    result.stop = AgentStop::MaxSteps;
    std::string sys = "You are Nishachar, an autonomous coding agent running on the user's machine. "
                      "Achieve the goal by calling tools, one call at a time, and use each result before "
                      "the next call. Write complete files (no placeholders). When the goal is fully done "
                      "and verified, reply with a short summary and no tool call.";
    std::string decl;
    for (const ToolDef& d : defs) decl += native_declaration(d);
    // Pending text to feed next; the model's own tokens are already in the KV.
    std::string feed = "<|turn>system\n" + sys + decl + "<turn|>\n<|turn>user\n" + goal +
                       "<turn|>\n<|turn>model\n";
    rt.reset();
    const double t0 = now_sec();
    for (int step = 0; step < cfg.max_steps; ++step) {
        printf("\033[1;35m--- [Step %d / %d] ---\033[0m\n", step + 1, cfg.max_steps);
        std::string acc, last_piece, rut;
        bool stopped = false;
        GenStats st;
        std::string out;
        try {
            out = rt.generate(feed, cfg.max_new_tokens, scfg, [&](const std::string& piece, int64_t) {
                fputs(piece.c_str(), stdout); fflush(stdout);
                acc += piece; last_piece = piece;
                if (acc.find("<tool_call|>") != std::string::npos) { stopped = true; return false; }
                if (!(rut = rut_unit(acc)).empty()) return false;
                return true;
            }, &st);
        } catch (const std::exception& e) {
            printf("\n[generate failed: %s]\n", e.what());
            result.stop = AgentStop::Empty;
            break;
        }
        printf("\n");
        fprintf(stderr, "\033[2m[step %d] prefill %d tok in %.1fs | gen %d tok @ %.2f tok/s | ctx %d/%d | t=%.0fs\033[0m\n",
                step + 1, st.processed_tokens, st.prefill_s, st.gen_tokens, st.decode_tok_s, st.ctx_used, st.ctx_max,
                now_sec() - t0);
        if (logf) {
            fprintf(logf, "{\"step\":%d,\"t\":%.1f,\"prefill_tokens\":%d,\"prefill_s\":%.2f,\"gen_tokens\":%d,"
                          "\"decode_tok_s\":%.3f,\"ctx_used\":%d,\"tool_stop\":%s}\n",
                    step + 1, now_sec() - t0, st.processed_tokens, st.prefill_s, st.gen_tokens, st.decode_tok_s,
                    st.ctx_used, stopped ? "true" : "false");
            fflush(logf);
        }
        AgentStep s;
        s.index = step;
        s.model_output = out;
        const size_t tc = out.find("<|tool_call>");
        if (tc == std::string::npos && rut.empty()) {
            // No call: the turn ended (EOG) — final answer. Strip channel noise.
            std::string fin = out;
            const std::string ch = "<|channel>thought\n<channel|>";
            if (fin.compare(0, ch.size(), ch) == 0) fin = fin.substr(ch.size());
            result.final_text = fin;
            result.stop = out.empty() ? AgentStop::Empty : AgentStop::Done;
            result.steps.push_back(std::move(s));
            break;
        }
        ToolCall call;
        std::string err, tool_result;
        bool ok = false;
        if (!rut.empty()) {
            printf("\n\033[1;31m[rut detected: output repeats '%s' — step aborted]\033[0m\n", rut.c_str());
            err = "your output got stuck repeating '" + rut + "' and the call was discarded. "
                  "Make the call again from the start, writing simpler code at that point";
        } else if (!stopped) err = "tool call was not closed with <tool_call|> (output too long?)";
        else if (parse_native_call(out.substr(tc), call, err)) {
            auto it = handlers.find(call.name);
            if (it == handlers.end()) err = "unknown tool '" + call.name + "'";
            else {
                try { tool_result = it->second(call); ok = true; }
                catch (const std::exception& e) { tool_result = std::string("error: ") + e.what(); }
            }
        }
        if (!err.empty()) tool_result = "error: " + err + ". Call exactly one tool as <|tool_call>call:NAME{arg:<|\"|>text<|\"|>}<tool_call|>.";
        s.had_tool_call = true;
        s.tool_name = call.name.empty() ? "?" : call.name;
        s.tool_args_json = call.raw_json;
        s.tool_result = tool_result;
        s.tool_ok = ok;
        result.steps.push_back(std::move(s));
        // The stop token (<tool_call|>) was sampled but not fed: re-feed it,
        // then the response; the model continues in the same turn.
        std::string body = tool_result;
        for (size_t k; (k = body.find(kQ)) != std::string::npos;) body.replace(k, 5, "\"");
        feed = (stopped ? last_piece : std::string("<tool_call|>")) + "<|tool_response>response:" +
               (call.name.empty() ? std::string("error") : call.name) + "{value:" + kQ + body + kQ + "}<tool_response|>";
    }
    return result;
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
    bool fast = false;
    size_t ram_budget = 0;
    int max_new = 2048, tool_timeout = 600;
    size_t tool_cap = 6000;
    std::string log_path, transcript_path;
    bool require_tool = false;

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
        } else if (a == "--fast") {
            fast = true;
        } else if (a == "--ram-budget" && i + 1 < argc) {
            std::string v = argv[++i];
            double x = std::stod(v);
            char u = (char)std::toupper((unsigned char)v.back());
            ram_budget = (size_t)(x * (u == 'G' ? 1e9 : u == 'M' ? 1e6 : u == 'K' ? 1e3 : 1.0));
        } else if (a == "--max-new" && i + 1 < argc) {
            max_new = std::stoi(argv[++i]);
        } else if (a == "--tool-timeout" && i + 1 < argc) {
            tool_timeout = std::stoi(argv[++i]);
        } else if (a == "--tool-cap" && i + 1 < argc) {
            tool_cap = (size_t)std::stoll(argv[++i]);
        } else if (a == "--log" && i + 1 < argc) {
            log_path = argv[++i];
        } else if (a == "--transcript" && i + 1 < argc) {
            transcript_path = argv[++i];
        } else if (a == "--require-tool") {
            require_tool = true;
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
        opt.fast_quant = fast;
        Runtime rt(std::move(src), opt, ctx, threads, ram_budget);

        printf("[Model Loaded] Arch: %s | Layers: %lld | Dim: %lld\n",
               rt.config().arch.c_str(),
               (long long)rt.config().n_layers,
               (long long)rt.config().dim);

        // Configure agent and tools
        Nishachar agent;
        std::vector<ToolDef> native_defs;
        std::map<std::string, ToolHandler> native_handlers;

        // Register real system tools with live progress logging
        register_system_tools([&](ToolDef def, ToolHandler handler) {
            std::string tool_name = def.name;
            ToolDef def_copy = def;
            ToolHandler wrapped = [handler = std::move(handler), tool_name, tool_timeout, tool_cap](const ToolCall& in) -> std::string {
                // bash: bound every command so a hung process (a dev server, an
                // interactive prompt) can't stall the agent forever.
                ToolCall call = in;
                if (tool_name == "bash")
                    for (auto& arg : call.args)
                        if (arg.key == "command" || arg.key == "cmd") {
                            std::string q = "'";
                            for (char c : arg.value) q += (c == '\'') ? std::string("'\\''") : std::string(1, c);
                            q += "'";
                            arg.value = "timeout -k 10 " + std::to_string(tool_timeout) + " bash -c " + q +
                                        " < /dev/null";
                        }
                printf("\n\033[1;36m>>> [Tool Execution: %s]\033[0m\n", tool_name.c_str());
                for (const auto& arg : call.args) {
                    std::string v = arg.value;
                    if (v.size() > 100) v = v.substr(0, 97) + "...";
                    for (char& c : v) if (c == '\n' || c == '\r') c = ' ';
                    printf("    \033[33m%s\033[0m: %s\n", arg.key.c_str(), v.c_str());
                }
                std::string result = handler(call);
                // Keep the model's context bounded: head + tail (errors and
                // summaries live at the end of build/test output).
                if (result.size() > tool_cap) {
                    const size_t head = tool_cap / 4, tail = tool_cap - head;
                    result = result.substr(0, head) + "\n[... " +
                             std::to_string(result.size() - head - tail) + " chars omitted ...]\n" +
                             result.substr(result.size() - tail);
                }
                std::string preview = result;
                if (preview.size() > 240) {
                    preview = preview.substr(0, 237) + "...";
                }
                for (char& c : preview) if (c == '\n' || c == '\r') c = ' ';
                printf("    \033[1;32m[Result (%zu bytes)]\033[0m %s\n\n", result.size(), preview.c_str());
                return result;
            };
            native_defs.push_back(def_copy);
            native_handlers[tool_name] = wrapped;
            agent.add_tool(std::move(def), std::move(wrapped));
        }, workdir);

        AgentConfig cfg;
        cfg.max_steps = max_steps;
        cfg.max_new_tokens = max_new;
        cfg.style = style_from_model(rt.config());
        cfg.require_tool_before_done = require_tool;

        // Transcript: one JSON object per line -- a header, every step as it
        // completes (flushed, so a crashed/killed run keeps its history), and
        // a closing summary line.
        FILE* trf = nullptr;
        if (!transcript_path.empty()) {
            trf = std::fopen(transcript_path.c_str(), "w");
            if (!trf) fprintf(stderr, "warning: cannot open transcript '%s'\n", transcript_path.c_str());
        }
        if (trf) {
            fprintf(trf, "{\"type\":\"start\",\"goal\":%s,\"model\":%s,\"system\":%s}\n",
                    json_str(goal).c_str(), json_str(model_path).c_str(),
                    json_str(cfg.system_preamble + "\n\n" + agent.tools().system_prompt_block()).c_str());
            fflush(trf);
            cfg.on_step = [trf](const AgentStep& s) {
                const char* kind = s.had_tool_call ? "tool" : s.malformed ? "malformed"
                                 : s.nudged ? "nudge" : "final";
                fprintf(trf, "{\"type\":\"step\",\"index\":%d,\"kind\":\"%s\",\"model_output\":%s,"
                             "\"tool_name\":%s,\"tool_args\":%s,\"tool_result\":%s,\"ok\":%s,"
                             "\"gen_s\":%.2f,\"tool_s\":%.2f}\n",
                        s.index, kind, json_str(s.model_output).c_str(), json_str(s.tool_name).c_str(),
                        json_str(s.tool_args_json).c_str(), json_str(s.tool_result).c_str(),
                        s.tool_ok ? "true" : "false", s.gen_s, s.tool_s);
                fflush(trf);
            };
        }

        SamplerConfig scfg;
        scfg.temperature = temp;

        // Append-only generation: `kv_text` is exactly the text whose tokens are
        // in the KV cache. Each step's prompt re-renders the whole conversation;
        // when it extends kv_text verbatim only the new suffix (the tool result
        // + the next turn header) is prefilled — O(new text) per step instead
        // of re-prefilling the full, growing transcript. A step stops as soon
        // as a complete </tool_call> has been emitted.
        int step_count = 0;
        std::string kv_text;
        FILE* logf = log_path.empty() ? nullptr : std::fopen(log_path.c_str(), "a");
        const double t_run0 = now_sec();
        AgentGenerator gen = [&](const std::string& prompt) -> std::string {
            ++step_count;
            printf("\033[1;35m--- [Step %d / %d: Generating Action] ---\033[0m\n", step_count, cfg.max_steps);
            const bool append = !kv_text.empty() && prompt.size() > kv_text.size() &&
                                prompt.compare(0, kv_text.size(), kv_text) == 0;
            if (!append) rt.reset();
            const std::string feed = append ? prompt.substr(kv_text.size()) : prompt;
            GenStats stats;
            std::string last_piece;
            bool stopped = false;
            std::string acc;
            std::string out;
            try {
                out = rt.generate(feed, cfg.max_new_tokens, scfg,
                    [&](const std::string& piece, int64_t) {
                        fputs(piece.c_str(), stdout);
                        fflush(stdout);
                        acc += piece;
                        last_piece = piece;
                        if (acc.find("</tool_call>") != std::string::npos ||
                            acc.find("<tool_call|>") != std::string::npos) { stopped = true; return false; }
                        return true;
                    }, &stats);
            } catch (const std::exception& e) {
                printf("\n[generate failed: %s]\n", e.what());
                kv_text.clear();
                return std::string();
            }
            printf("\n");
            // A callback stop returns before the final token is fed forward, so
            // the KV holds `out` minus its last piece; the next render re-feeds it.
            kv_text = prompt + (stopped ? out.substr(0, out.size() - last_piece.size()) : out);
            fprintf(stderr, "\033[2m[step %d] %s prefill %d tok in %.1fs (%.1f tok/s) | gen %d tok @ %.2f tok/s | ctx %d/%d\033[0m\n",
                    step_count, append ? "append" : "full", stats.processed_tokens, stats.prefill_s,
                    stats.prefill_s > 0 ? stats.processed_tokens / stats.prefill_s : 0.0,
                    stats.gen_tokens, stats.decode_tok_s, stats.ctx_used, stats.ctx_max);
            if (logf) {
                fprintf(logf, "{\"step\":%d,\"t\":%.1f,\"mode\":\"%s\",\"prefill_tokens\":%d,\"prefill_s\":%.2f,"
                              "\"gen_tokens\":%d,\"decode_tok_s\":%.3f,\"ctx_used\":%d,\"tool_stop\":%s}\n",
                        step_count, now_sec() - t_run0, append ? "append" : "full", stats.processed_tokens,
                        stats.prefill_s, stats.gen_tokens, stats.decode_tok_s, stats.ctx_used,
                        stopped ? "true" : "false");
                fflush(logf);
            }
            return out;
        };

        printf("\033[1;32m[Starting Nishachar Execution Loop]\033[0m\n\n");
        if (cfg.style == ChatTemplateStyle::Gemma4 && !getenv("NISHACHAR_JSON_TOOLS")) {
            AgentResult r = run_gemma4_native(rt, scfg, cfg, goal, native_defs, native_handlers, logf);
            print_result(r);
            return (r.stop == AgentStop::Done) ? 0 : 1;
        }
        AgentResult result = agent.run(goal, gen, cfg);
        if (trf) {
            fprintf(trf, "{\"type\":\"end\",\"stop\":\"%s\",\"steps\":%d,\"tool_calls\":%d,"
                         "\"wall_s\":%.1f,\"final_text\":%s}\n",
                    agent_stop_name(result.stop), result.steps_taken(), result.tool_calls(),
                    now_sec() - t_run0, json_str(result.final_text).c_str());
            std::fclose(trf);
        }
        if (logf) std::fclose(logf);

        print_result(result);
        return (result.stop == AgentStop::Done) ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "\nfatal error: %s\n", e.what());
        return 1;
    }
}
