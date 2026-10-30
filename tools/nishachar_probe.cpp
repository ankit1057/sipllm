// nishachar_probe — prove (or disprove) that a real model drives the Nishachar
// tool loop end to end, and record the evidence.
//
//   nishachar_probe <model_path> [--case ID|all] [--format compact|qwen|auto]
//                   [--max-steps N] [--max-new N] [--temp T] [--threads N]
//                   [--ram-budget MB] [--fast] [--reuse] [--no-stop-at-call]
//                   [--out FILE]
//
// Each case gives the model one goal and three harmless tools (calc, echo, and a
// read_file confined to a fresh temp directory), runs the real Nishachar loop on
// the real Runtime, and checks:
//   1. the model emitted tool calls the parser accepted,
//   2. the intended tools were dispatched, in order, with the intended arguments,
//   3. each tool's result was fed back in the next prompt,
//   4. the loop ended with a final answer that carries the result,
//   5. it stopped inside the step bound.
// A control case expects no tool call; an error case expects the failed read to
// be reported back to the model and the loop to end.
//
// --format auto (default) uses the model's own tool format when its GGUF chat
// template has one (tool_format_from_model). --ram-budget defaults to the C
// ABI's AUTO (half of available RAM); 0 streams every layer on every token.
// --reuse keeps the KV cache across steps and cases and prefills only the new
// tail of each prompt. Generation stops once "</tool_call>" appears unless
// --no-stop-at-call. Every step's rendered prompt, raw output, token counts,
// timings and peak RSS go to the transcript (--out, default stdout). Exit 0
// only if every check of every case passed.
#include "llm/common.h"
#include "llm/device_profile.h"
#include "llm/nishachar.h"
#include "llm/runtime.h"
#include "llm/sampler.h"
#include "llm/tools.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;
namespace fs = std::filesystem;

namespace {

// "<int> <op> <int>", integer math; throws on anything else (same contract as
// nishachar_demo's calc).
std::string eval_calc(const std::string& expr) {
    std::istringstream in(expr);
    long long a = 0, b = 0;
    std::string op;
    if (!(in >> a >> op >> b))
        throw std::runtime_error("bad expression: expected '<int> <op> <int>'");
    if (op.size() != 1 || std::string("+-*/").find(op) == std::string::npos)
        throw std::runtime_error("unknown operator '" + op + "'");
    switch (op[0]) {
        case '+': return std::to_string(a + b);
        case '-': return std::to_string(a - b);
        case '*': return std::to_string(a * b);
        default:
            if (b == 0) throw std::runtime_error("division by zero");
            return std::to_string(a / b);
    }
}

// read_file confined to `root`: relative paths only, no "..", no symlink escape.
std::string confined_read(const fs::path& root, const std::string& raw) {
    if (raw.empty()) throw std::runtime_error("'path' argument is required");
    fs::path p(raw);
    if (p.is_absolute()) throw std::runtime_error("absolute paths are not allowed");
    for (const auto& part : p)
        if (part == "..") throw std::runtime_error("'..' is not allowed");
    // Compare components, not string prefixes: "/tmp/box2/x" starts with "/tmp/box".
    std::error_code ec;
    const fs::path full = fs::weakly_canonical(root / p, ec);
    const fs::path base = fs::weakly_canonical(root, ec);
    if (ec || std::mismatch(base.begin(), base.end(), full.begin(), full.end()).first != base.end())
        throw std::runtime_error("path escapes the sandbox");
    std::ifstream f(full, std::ios::binary);
    if (!f) throw std::runtime_error("file not found: " + raw);
    std::ostringstream os;
    os << f.rdbuf();
    return os.str();
}

struct Expect {
    std::string tool;
    std::string arg_key;
    std::vector<std::string> arg_has;   // substrings of that argument
    std::string result;                 // exact tool result; empty = any
    bool ok = true;                     // false: the tool must fail
};

struct Case {
    std::string id;
    std::string goal;
    std::vector<Expect> calls;                // empty = no tool call expected
    std::vector<std::string> final_any;       // final contains one of these; empty = any
};

struct StepTrace {
    std::string prompt;
    std::string output;
    GenStats    stats;
    double      wall_s = 0;
    size_t      peak_rss = 0;
};

struct Check { std::string what; bool ok; std::string why; };

const char* format_name(ToolPromptFormat f) { return f == ToolPromptFormat::Qwen ? "qwen" : "compact"; }

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: nishachar_probe <model_path> [--case ID|all] [--format compact|qwen|auto] "
                        "[--max-steps N] [--max-new N] [--temp T] [--threads N] [--ram-budget MB] "
                        "[--fast] [--reuse] [--no-stop-at-call] [--out FILE]\n");
        return 2;
    }
    const std::string model_path = argv[1];
    std::string which = "all", out_path, format_arg = "auto";
    int max_steps = 4, max_new = 256, threads = 0;
    float temp = 0.0f;
    long long budget_mb = -1;   // -1 = AUTO
    bool fast = false, reuse = false, stop_at_call = true;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--case") which = next();
        else if (a == "--format") format_arg = next();
        else if (a == "--max-steps") max_steps = std::stoi(next());
        else if (a == "--max-new") max_new = std::stoi(next());
        else if (a == "--temp") temp = std::stof(next());
        else if (a == "--threads") threads = std::stoi(next());
        else if (a == "--ram-budget") budget_mb = std::stoll(next());
        else if (a == "--fast") fast = true;
        else if (a == "--reuse") reuse = true;
        else if (a == "--no-stop-at-call") stop_at_call = false;
        else if (a == "--out") out_path = next();
        else { fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }
    if (format_arg != "auto" && format_arg != "compact" && format_arg != "qwen") {
        fprintf(stderr, "--format must be compact, qwen or auto\n");
        return 2;
    }

    FILE* out = stdout;
    if (!out_path.empty()) {
        out = std::fopen(out_path.c_str(), "w");
        if (!out) { fprintf(stderr, "cannot open %s\n", out_path.c_str()); return 2; }
    }

    // Sandbox for read_file, with files whose contents the model cannot guess.
    const fs::path sandbox = fs::temp_directory_path() /
        ("nishachar_probe_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(sandbox);
    { std::ofstream(sandbox / "notes.txt") << "The access code is TANGERINE-4711.\n"; }
    { std::ofstream(sandbox / "config.txt") << "host=localhost\nport=8731\n"; }
    { std::ofstream(sandbox / "numbers.txt") << "417 23\n"; }

    const std::vector<Case> cases = {
        {"calc", "What is 1234 * 5678? Use the calc tool.",
         {{"calc", "expr", {"1234", "5678", "*"}, "7006652"}}, {"7006652", "7,006,652"}},
        {"calc_unnamed", "What is 98765 minus 4321?",
         {{"calc", "expr", {"98765", "4321", "-"}, "94444"}}, {"94444", "94,444"}},
        {"echo", "Use the echo tool to echo the text 'violet lantern 42', then tell me what it returned.",
         {{"echo", "text", {"violet lantern 42"}, "violet lantern 42"}}, {"violet lantern 42"}},
        {"echo_unnamed", "Echo the phrase 'quartz falcon 7' back to me.",
         {{"echo", "text", {"quartz falcon 7"}, "quartz falcon 7"}}, {"quartz falcon 7"}},
        {"read", "Read the file notes.txt with the read_file tool and tell me the access code in it.",
         {{"read_file", "path", {"notes.txt"}, "The access code is TANGERINE-4711.\n"}}, {"TANGERINE-4711"}},
        {"read_unnamed", "What port number is set in config.txt?",
         {{"read_file", "path", {"config.txt"}, "host=localhost\nport=8731\n"}}, {"8731"}},
        {"chain", "numbers.txt contains two numbers. Read it, then multiply the two numbers with the calc "
                  "tool and tell me the product.",
         {{"read_file", "path", {"numbers.txt"}, "417 23\n"}, {"calc", "expr", {"417", "23", "*"}, "9591"}},
         {"9591", "9,591"}},
        {"no_tool", "Say hello in one short sentence.", {}, {}},
        {"read_missing", "Read the file missing.txt and tell me what it says.",
         {{"read_file", "path", {"missing.txt"}, "", /*ok=*/false}}, {}},
    };

    int failed_cases = 0, ran = 0;
    try {
        auto src = open_model(model_path);
        LayerLoader::Options opt;
        opt.residency = Residency::Quantized;
        opt.fast_quant = fast;
        // Same rule as sipllm_open's SIPLLM_RAM_BUDGET_AUTO: half of the
        // available RAM, forced so an infeasible plan falls back to the minimum.
        const size_t budget = budget_mb < 0 ? auto_ram_budget(available_ram_bytes())
                                            : (size_t)budget_mb * 1024 * 1024;
        Runtime rt(std::move(src), opt, /*max_ctx=*/0, threads, budget, /*force_budget=*/budget > 0);
        rt.set_context_reuse(reuse);
        const ChatTemplateStyle style = style_from_model(rt.config());
        const ToolPromptFormat format = format_arg == "auto" ? tool_format_from_model(rt.config())
                                      : format_arg == "qwen" ? ToolPromptFormat::Qwen
                                                             : ToolPromptFormat::Compact;

        fprintf(out, "# nishachar_probe\nmodel: %s\narch: %s  layers: %lld  dim: %lld  vocab: %lld\n"
                     "chat_style: %d  tool_format: %s (%s)  threads: %d  temp: %.2f  max_new: %d  "
                     "max_steps: %d  ram_budget: %zu MB  fast: %d  reuse: %d  stop_at_call: %d\n",
                model_path.c_str(), rt.config().arch.c_str(), (long long)rt.config().n_layers,
                (long long)rt.config().dim, (long long)rt.config().vocab_size, (int)style,
                format_name(format), format_arg.c_str(), rt.thread_pool() ? rt.thread_pool()->size() : 1,
                temp, max_new, max_steps, budget / (1024 * 1024), fast, reuse, stop_at_call);
        fprintf(out, "[gguf chat_template]\n%s\n[/gguf chat_template]\n", rt.config().chat_template.c_str());

        for (const Case& c : cases) {
            if (which != "all" && which != c.id) continue;
            ++ran;
            Nishachar agent;
            ToolDef calc{"calc", "Evaluate an integer arithmetic expression '<int> <op> <int>', op one of + - * /.",
                         {{"expr", ToolParamType::String, true, "the expression, e.g. '6 * 7'", ""}}};
            agent.add_tool(calc, [](const ToolCall& call) { return eval_calc(call.get("expr")); });
            ToolDef echo{"echo", "Return the given text verbatim.",
                         {{"text", ToolParamType::String, true, "the text to echo back", ""}}};
            agent.add_tool(echo, [](const ToolCall& call) { return call.get("text"); });
            ToolDef rd{"read_file", "Read a text file from the working directory.",
                       {{"path", ToolParamType::String, true, "relative path of the file", ""}}};
            agent.add_tool(rd, [&](const ToolCall& call) { return confined_read(sandbox, call.get("path")); });

            AgentConfig cfg;
            cfg.max_steps = max_steps;
            cfg.max_new_tokens = max_new;
            cfg.style = style;
            cfg.tool_format = format;
            SamplerConfig scfg;
            scfg.temperature = temp;

            std::vector<StepTrace> traces;
            AgentGenerator gen = [&](const std::string& prompt) {
                StepTrace t;
                t.prompt = prompt;
                if (!reuse) rt.reset();
                std::string acc;
                Runtime::TokenCallback on_token = [&](const std::string& piece, int64_t) {
                    acc += piece;
                    return !(stop_at_call && acc.find("</tool_call>") != std::string::npos);
                };
                auto t0 = std::chrono::steady_clock::now();
                t.output = rt.generate(prompt, cfg.max_new_tokens, scfg, on_token, &t.stats);
                t.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                t.peak_rss = peak_rss_bytes();
                traces.push_back(t);
                return t.output;
            };
            const auto c0 = std::chrono::steady_clock::now();
            AgentResult r = agent.run(c.goal, gen, cfg);
            const double case_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();

            fprintf(out, "\n================ case %s ================\ngoal: %s\n", c.id.c_str(), c.goal.c_str());
            for (size_t i = 0; i < traces.size(); ++i) {
                const StepTrace& t = traces[i];
                fprintf(out, "\n---- step %zu: prompt_tokens=%d processed=%d reused=%d gen_tokens=%d "
                             "prefill=%.2fs decode=%.2fs wall=%.2fs pinned=%d/%lld peak_rss=%zu MB\n",
                        i, t.stats.prompt_tokens, t.stats.processed_tokens, t.stats.reused_prefix_tokens,
                        t.stats.gen_tokens, t.stats.prefill_s, t.stats.decode_s, t.wall_s,
                        t.stats.pinned_layers, (long long)rt.config().n_layers, t.peak_rss / (1024 * 1024));
                fprintf(out, "[prompt]\n%s\n[/prompt]\n[output]\n%s\n[/output]\n", t.prompt.c_str(), t.output.c_str());
                if (i < r.steps.size() && r.steps[i].had_tool_call)
                    fprintf(out, "[dispatch] tool=%s ok=%d args=%s\n[result] %s\n", r.steps[i].tool_name.c_str(),
                            r.steps[i].tool_ok ? 1 : 0, r.steps[i].tool_args_json.c_str(), r.steps[i].tool_result.c_str());
            }

            std::vector<const AgentStep*> calls;
            for (const AgentStep& s : r.steps) if (s.had_tool_call) calls.push_back(&s);
            std::vector<Check> checks;
            if (c.calls.empty()) {
                checks.push_back({"no tool call", calls.empty(),
                                  calls.empty() ? "" : "called '" + calls[0]->tool_name + "'"});
            }
            for (size_t k = 0; k < c.calls.size(); ++k) {
                const Expect& e = c.calls[k];
                const AgentStep* s = k < calls.size() ? calls[k] : nullptr;
                const std::string tag = "call " + std::to_string(k + 1) + " ";
                checks.push_back({tag + "emitted and parsed", s != nullptr,
                                  s ? "" : "only " + std::to_string(calls.size()) + " call(s) parsed"});
                if (!s) continue;
                checks.push_back({tag + "dispatched " + e.tool, s->tool_name == e.tool,
                                  "dispatched '" + s->tool_name + "'"});
                std::string arg;
                ToolParser p(agent.tools());
                p.feed(s->model_output);
                if (p.state() == ToolParser::State::Done) arg = p.parsed_call().get(e.arg_key);
                bool arg_ok = s->tool_name == e.tool;
                for (const std::string& h : e.arg_has) arg_ok = arg_ok && arg.find(h) != std::string::npos;
                checks.push_back({tag + e.arg_key + " is right", arg_ok, "got '" + arg + "'"});
                bool res_ok = s->tool_ok == e.ok && (e.result.empty() || s->tool_result == e.result);
                checks.push_back({tag + (e.ok ? "returned the expected result" : "failed as expected"), res_ok,
                                  "ok=" + std::to_string(s->tool_ok) + " got '" + s->tool_result + "'"});
                const size_t next_step = (size_t)s->index + 1;
                const bool fed = next_step < traces.size() &&
                                 traces[next_step].prompt.find(s->tool_result) != std::string::npos;
                checks.push_back({tag + "result fed back in the next prompt", fed,
                                  fed ? "" : "next prompt missing or lacks the result"});
            }
            bool final_ok = r.stop == AgentStop::Done && !r.final_text.empty();
            if (final_ok && !c.final_any.empty()) {
                bool any = false;
                for (const std::string& f : c.final_any) any = any || r.final_text.find(f) != std::string::npos;
                final_ok = any;
            }
            checks.push_back({"final answer" + std::string(c.final_any.empty() ? "" : " carries the result"),
                              final_ok, std::string("stop=") + agent_stop_name(r.stop) + " final='" + r.final_text + "'"});
            checks.push_back({"stopped within the step bound", r.steps_taken() <= max_steps,
                              "steps=" + std::to_string(r.steps_taken())});

            bool all = true;
            fprintf(out, "\n[verdict %s] steps=%d tool_calls=%d stop=%s case_wall=%.1fs\n", c.id.c_str(),
                    r.steps_taken(), r.tool_calls(), agent_stop_name(r.stop), case_s);
            for (const Check& k : checks) {
                fprintf(out, "  %s  %s%s%s\n", k.ok ? "PASS" : "FAIL", k.what.c_str(),
                        k.why.empty() ? "" : "  -- ", k.why.c_str());
                all = all && k.ok;
            }
            if (!all) ++failed_cases;
            fprintf(stderr, "[%s] %s  (%d steps, %.1fs)\n", c.id.c_str(), all ? "PASS" : "FAIL",
                    r.steps_taken(), case_s);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        fs::remove_all(sandbox);
        return 1;
    }
    fs::remove_all(sandbox);
    if (ran == 0) { fprintf(stderr, "no case named '%s'\n", which.c_str()); return 2; }
    fprintf(out, "\nSUMMARY: %d/%d cases passed  peak_rss=%zu MB\n", ran - failed_cases, ran,
            peak_rss_bytes() / (1024 * 1024));
    fprintf(stderr, "SUMMARY: %d/%d cases passed\n", ran - failed_cases, ran);
    if (out != stdout) std::fclose(out);
    return failed_cases == 0 ? 0 : 1;
}
