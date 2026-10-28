// test_nishachar.cpp — Nishachar Path A loop (scripted-generator unit tests).
//
// Exercises the autonomous goal->plan->act->verify loop deterministically with
// a scripted AgentGenerator (no language model): registers real tool handlers
// on a Nishachar and drives run() through every stop reason and dispatch path.
#include "llm/nishachar.h"
#include "llm/tools.h"
#include "tests/test_util.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llm;

static bool has_text(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

// A scripted generator: yields `outputs` in order across successive calls, then
// empty once exhausted. The shared index makes the counter survive the copy the
// std::function wrapper performs.
static AgentGenerator scripted(std::vector<std::string> outputs) {
    auto idx = std::make_shared<size_t>(0);
    auto outs = std::make_shared<std::vector<std::string>>(std::move(outputs));
    return [idx, outs](const std::string&) -> std::string {
        if (*idx >= outs->size()) return "";
        return (*outs)[(*idx)++];
    };
}

// A generator that always returns the same output regardless of step.
static AgentGenerator always(std::string output) {
    auto out = std::make_shared<std::string>(std::move(output));
    return [out](const std::string&) -> std::string { return *out; };
}

// Build a Nishachar with echo / add / boom tools registered.
static Nishachar make_agent() {
    Nishachar agent;

    ToolDef echo;
    echo.name = "echo";
    echo.description = "Echo the text argument back";
    echo.params = {{"text", ToolParamType::String, true, "text to echo", ""}};
    agent.add_tool(echo, [](const ToolCall& c) -> std::string {
        return c.get("text");
    });

    ToolDef add;
    add.name = "add";
    add.description = "Add two integers";
    add.params = {
        {"a", ToolParamType::Int, true, "first addend", ""},
        {"b", ToolParamType::Int, true, "second addend", ""},
    };
    agent.add_tool(add, [](const ToolCall& c) -> std::string {
        int a = std::stoi(c.get("a", "0"));
        int b = std::stoi(c.get("b", "0"));
        return std::to_string(a + b);
    });

    ToolDef boom;
    boom.name = "boom";
    boom.description = "Always fails";
    agent.add_tool(boom, [](const ToolCall&) -> std::string {
        throw std::runtime_error("boom");
    });

    return agent;
}

static const char* kEchoHi =
    "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"hi\"}}</tool_call>";
static const char* kAdd23 =
    "<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":\"2\",\"b\":\"3\"}}</tool_call>";
static const char* kBoom =
    "<tool_call>{\"name\":\"boom\",\"arguments\":{}}</tool_call>";
static const char* kUnknown =
    "<tool_call>{\"name\":\"nope\",\"arguments\":{}}</tool_call>";

TEST(immediate_final) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("say hello", scripted({"Hello there!"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.final_text == "Hello there!");
    CHECK(r.steps_taken() == 1);
    CHECK(r.tool_calls() == 0);
    CHECK(!r.steps[0].had_tool_call);
}

TEST(single_tool_then_final) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("echo hi", scripted({kEchoHi, "all done"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 1);
    CHECK(r.steps_taken() == 2);
    CHECK(r.steps[0].had_tool_call);
    CHECK(r.steps[0].tool_name == "echo");
    CHECK(r.steps[0].tool_result == "hi");
    CHECK(r.steps[0].tool_ok);
    CHECK(!r.steps[1].had_tool_call);
    CHECK(r.final_text == "all done");
}

TEST(multi_step_chain) {
    Nishachar agent = make_agent();
    AgentResult r =
        agent.run("chain", scripted({kEchoHi, kAdd23, "finished"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 2);
    CHECK(r.steps_taken() == 3);
    CHECK(r.steps[0].tool_name == "echo");
    CHECK(r.steps[0].tool_result == "hi");
    CHECK(r.steps[0].tool_ok);
    CHECK(r.steps[1].tool_name == "add");
    CHECK(r.steps[1].tool_result == "5");
    CHECK(r.steps[1].tool_ok);
    CHECK(!r.steps[2].had_tool_call);
    CHECK(r.final_text == "finished");
}

TEST(max_steps_bound) {
    Nishachar agent = make_agent();
    AgentConfig cfg;
    cfg.max_steps = 3;
    AgentResult r = agent.run("loop forever", always(kEchoHi), cfg);
    CHECK(r.stop == AgentStop::MaxSteps);
    CHECK(r.steps_taken() == 3);
    CHECK(r.tool_calls() == 3);
}

TEST(unknown_tool_is_fed_back) {
    // A marked call to an unregistered tool is an attempted call: the loop
    // feeds an error back (listing the real tools) and keeps going.
    Nishachar agent = make_agent();
    AgentResult r = agent.run("call unknown", scripted({kUnknown, "ok then"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 0);
    CHECK(r.steps_taken() == 2);
    CHECK(r.steps[0].malformed);
    CHECK(r.steps[0].tool_result.find("unknown tool 'nope'") != std::string::npos);
    CHECK(r.steps[0].tool_result.find("echo") != std::string::npos);
    CHECK(r.final_text == "ok then");
}

TEST(malformed_call_recovery) {
    Nishachar agent = make_agent();
    // Unterminated (cut off), broken JSON (unescaped quote), then a good call.
    std::vector<std::string> prompts;
    auto inner = scripted({
        "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"hi",
        "<tool_call>{\"name\":\"echo\",\"arguments\":{\"text\":\"say \"hi\" now\"}}</tool_call>",
        kEchoHi,
        "done",
    });
    AgentGenerator gen = [&](const std::string& p) { prompts.push_back(p); return inner(p); };
    AgentResult r = agent.run("echo hi", gen);
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.steps_taken() == 4);
    CHECK(r.steps[0].malformed && !r.steps[0].had_tool_call);
    CHECK(r.steps[0].tool_result.rfind("error: could not parse your tool call (", 0) == 0);
    CHECK(r.steps[0].tool_result.find("escape newlines") != std::string::npos);
    CHECK(r.steps[1].malformed);
    CHECK(r.steps[1].tool_result.find("invalid JSON") != std::string::npos);
    CHECK(r.steps[2].had_tool_call && r.steps[2].tool_result == "hi");
    CHECK(r.tool_calls() == 1);
    CHECK(r.final_text == "done");
    // The error text was fed back into the next prompt.
    CHECK(prompts.size() == 4);
    CHECK(prompts[1].find("could not parse your tool call") != std::string::npos);
    CHECK(r.report().find("malformed") != std::string::npos);
}

TEST(missing_required_arg_is_malformed) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("x", scripted({
        "<tool_call>{\"name\":\"echo\",\"arguments\":{}}</tool_call>", "fine"}));
    CHECK(r.steps[0].malformed);
    CHECK(r.steps[0].tool_result.find("missing required argument 'text'") != std::string::npos);
    CHECK(r.stop == AgentStop::Done);
}

TEST(fenced_json_call) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("echo", scripted({
        "Sure.\n<tool_call>\n```json\n{\"name\": \"echo\", \"arguments\": {\"text\": \"fenced\"}}\n```\n</tool_call>",
        "done"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 1);
    CHECK(r.steps[0].tool_result == "fenced");
}

TEST(string_encoded_arguments) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("add", scripted({
        "<tool_call>{\"name\":\"add\",\"arguments\":\"{\\\"a\\\": 4, \\\"b\\\": 5}\"}</tool_call>",
        "<tool_call>{\"name\":\"echo\",\"parameters\":{\"text\":\"line1\\nline2\"}}</tool_call>",
        "done"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 2);
    CHECK(r.steps[0].tool_result == "9");
    CHECK(r.steps[1].tool_result == "line1\nline2");
}

TEST(gemma4_native_call) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("g4", scripted({
        "<|tool_call>call:echo{text:<|\"|>he said \"yo\"\nok<|\"|>}<tool_call|>",
        "<|tool_call>call:add{a:2,b:40}<tool_call|>",
        "<|tool_call>call:echo{text:<|\"|>never closed",
        "done"}));
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.tool_calls() == 2);
    CHECK(r.steps[0].tool_result == "he said \"yo\"\nok");
    CHECK(r.steps[1].tool_result == "42");
    CHECK(r.steps[2].malformed);
}

TEST(parse_agent_tool_call_kinds) {
    Nishachar agent = make_agent();
    const ToolRegistry& reg = agent.tools();
    CHECK(parse_agent_tool_call("just prose, no call", reg).kind == AgentCallParse::Kind::None);
    CHECK(parse_agent_tool_call("code: int f() { return 1; }", reg).kind == AgentCallParse::Kind::None);
    // Bare JSON naming a registered tool still counts (legacy behaviour).
    auto b = parse_agent_tool_call("{\"name\":\"echo\",\"arguments\":{\"text\":\"x\"}}", reg);
    CHECK(b.kind == AgentCallParse::Kind::Call && b.call.get("text") == "x");
    CHECK(parse_agent_tool_call("<tool_call>oops</tool_call>", reg).kind ==
          AgentCallParse::Kind::Malformed);
}

TEST(loop_guard) {
    Nishachar agent = make_agent();
    AgentConfig cfg;
    cfg.max_steps = 10;
    int runs = 0;
    agent.add_tool(ToolDef{"tick", "count", {}}, [&](const ToolCall&) {
        ++runs;
        return std::string("tock");
    });
    AgentResult r = agent.run("loop",
        always("<tool_call>{\"name\":\"tick\",\"arguments\":{}}</tool_call>"), cfg);
    CHECK(r.stop == AgentStop::Loop);
    CHECK(r.steps_taken() == 4);
    CHECK(runs == 2);  // 3rd identical call is refused, 4th stops the run
    CHECK(!r.steps[2].tool_ok);
    CHECK(r.steps[2].tool_result.find("looping") != std::string::npos);
    CHECK(std::string(agent_stop_name(r.stop)) == "loop");

    // A different call in between resets the streak.
    AgentResult r2 = agent.run("alt", scripted({kEchoHi, kEchoHi, kAdd23, kEchoHi, kEchoHi, "done"}), cfg);
    CHECK(r2.stop == AgentStop::Done);
    for (int i = 0; i < 5; ++i) CHECK(r2.steps[i].tool_ok);
}

TEST(require_tool_nudge) {
    Nishachar agent = make_agent();
    // Default: a tool-less first answer is final.
    AgentResult r0 = agent.run("write code", scripted({"Here is the code: ...", "x"}));
    CHECK(r0.stop == AgentStop::Done && r0.steps_taken() == 1);

    AgentConfig cfg;
    cfg.require_tool_before_done = true;
    std::vector<std::string> prompts;
    auto inner = scripted({"Here is the code: ...", kEchoHi, "done"});
    AgentGenerator gen = [&](const std::string& p) { prompts.push_back(p); return inner(p); };
    AgentResult r = agent.run("write code", gen, cfg);
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.steps_taken() == 3);
    CHECK(r.steps[0].nudged);
    CHECK(prompts[1].find("You have not used any tool yet") != std::string::npos);
    CHECK(r.final_text == "done");

    // Nudges only once: a second tool-less answer is accepted.
    AgentResult r2 = agent.run("w", scripted({"talk", "more talk"}), cfg);
    CHECK(r2.stop == AgentStop::Done && r2.steps_taken() == 2);
    CHECK(r2.final_text == "more talk");
}

TEST(on_step_and_timing) {
    Nishachar agent = make_agent();
    AgentConfig cfg;
    std::vector<int> seen;
    cfg.on_step = [&](const AgentStep& s) { seen.push_back(s.index); };
    AgentResult r = agent.run("t", scripted({kEchoHi, "done"}), cfg);
    CHECK(seen.size() == 2 && seen[0] == 0 && seen[1] == 1);
    CHECK(r.steps[0].gen_s >= 0 && r.steps[0].tool_s >= 0);
    CHECK(r.steps[1].tool_s < 0);
    CHECK(r.report().find("[gen ") != std::string::npos);
}

TEST(tool_error_continue) {
    Nishachar agent = make_agent();
    AgentConfig cfg;
    cfg.stop_on_tool_error = false;
    AgentResult r = agent.run("recover", scripted({kBoom, "recovered"}), cfg);
    CHECK(r.stop == AgentStop::Done);
    CHECK(r.steps_taken() == 2);
    CHECK(r.steps[0].had_tool_call);
    CHECK(!r.steps[0].tool_ok);
    CHECK(r.steps[0].tool_result.rfind("error:", 0) == 0);
    CHECK(r.final_text == "recovered");
}

TEST(tool_error_stop) {
    Nishachar agent = make_agent();
    AgentConfig cfg;
    cfg.stop_on_tool_error = true;
    AgentResult r = agent.run("fail hard", scripted({kBoom}), cfg);
    CHECK(r.stop == AgentStop::ToolError);
    CHECK(r.tool_calls() == 1);
    CHECK(r.steps_taken() == 1);
    CHECK(!r.steps[0].tool_ok);
}

TEST(empty_generation) {
    Nishachar agent = make_agent();
    AgentResult r = agent.run("nothing", scripted({""}));
    CHECK(r.stop == AgentStop::Empty);
    CHECK(r.tool_calls() == 0);
}

TEST(report_mentions) {
    Nishachar agent = make_agent();
    AgentResult r =
        agent.run("please echo hi", scripted({kEchoHi, "all done"}));
    std::string rep = r.report();
    CHECK(rep.find("please echo hi") != std::string::npos);
    CHECK(rep.find("echo") != std::string::npos);
}

TEST(failed_call_tracker_counts_across_interleaved_calls) {
    // The L4 portfolio run: the same no-op edit_file failed 7 times with a
    // successful bash (flutter analyze) in between; the old guard reset there.
    FailedCallTracker t;
    const std::string same = "error: old_string and new_string are identical, so nothing changed.\nmore";
    CHECK(t.record("edit_file", same) == 1);
    CHECK(t.record("bash", "No issues found! [exit code: 0]") == 0);   // success: not counted
    CHECK(t.record("edit_file", same) == 2);                          // still counted after it
    CHECK(t.record("edit_file", "error: old_string not found in a.dart") == 1);  // different error
    CHECK(t.record("read_file", same) == 1);                          // different tool
    CHECK(t.record("edit_file", same) == 3);
    std::string note = repeat_failure_note("edit_file", 3);
    CHECK(note.find("failure #3") != std::string::npos);
    CHECK(note.find("start_line") != std::string::npos);
    CHECK(repeat_failure_note("bash", 2).find("start_line") == std::string::npos);
}

TEST(degenerate_tail_unit_catches_long_units) {
    // v6-r1 L3: Gemma 4 emitted the 28-char empty-thought marker for 8192
    // tokens; the old detector only looked at units of <= 24 chars.
    std::string junk = "<|tool_call>call:edit_file{";
    for (int i = 0; i < 9; ++i) junk += "<|channel>thought\n<channel|>";
    CHECK(degenerate_tail_unit(junk) == "<|channel>thought\n<channel|>" ||
          degenerate_tail_unit(junk).size() == 28);
    // The original short case still caught.
    std::string dots = "RegExp(r'^[a-z";
    for (int i = 0; i < 40; ++i) dots += "\\x2d\\x2e\\x20";
    CHECK(!degenerate_tail_unit(dots).empty());
    // Healthy code: similar-looking but not identical lines are not a rut.
    std::string code;
    const char* names[] = {"about", "skills", "projects", "contact", "experience", "footer", "hero"};
    for (const char* n : names) code += std::string("      _section('") + n + "', const SizedBox(height: 300)),\n";
    CHECK_MSG(degenerate_tail_unit(code).empty(), degenerate_tail_unit(code));
    // Seven identical long lines is under the 8-repeat bar for long units.
    std::string seven;
    for (int i = 0; i < 7; ++i) seven += "            const SizedBox(height: 16, width: 8),\n";
    CHECK(degenerate_tail_unit(seven).empty());
    // Padding runs are legitimate.
    CHECK(degenerate_tail_unit("x" + std::string(500, ' ')).empty());
}

TEST(native_step_classification) {
    const std::string th = "<|channel>thought\n<channel|>";
    // Complete call / call cut off at the cap / rut wins over everything.
    CHECK(classify_native_step(th + "<|tool_call>call:bash{command:<|\"|>ls<|\"|>}<tool_call|>", true, "", false).kind ==
          NativeStepKind::ToolCall);
    CHECK(classify_native_step(th + "<|tool_call>call:write_file{content:<|\"|>...", false, "", true).kind ==
          NativeStepKind::CutInCall);
    CHECK(classify_native_step(th + th + th, false, th, false).kind == NativeStepKind::Rut);
    // v6-r1 L3 step 8: 8192 tokens, no call, no end of turn -> NOT a final answer.
    CHECK(classify_native_step(th + "I will now carefully think about", false, "", true).kind == NativeStepKind::CutNoCall);
    // End of turn with only markers / nothing -> junk; with text -> final, markers stripped.
    CHECK(classify_native_step(th + th + "\n", false, "", false).kind == NativeStepKind::JunkFinal);
    CHECK(classify_native_step("", false, "", false).kind == NativeStepKind::JunkFinal);
    NativeStep f = classify_native_step(th + "Done: added the theme toggle.\n", false, "", false);
    CHECK(f.kind == NativeStepKind::Final && f.final_text == "Done: added the theme toggle.");
}

TEST(native_feed_strings_have_no_control_tokens) {
    CHECK(strip_control_tokens("<|channel>thought\n<channel|>") == "<channel>thought\n<channel>");
    CHECK(strip_control_tokens("<|tool_call>call:x{a:<|\"|>v<|\"|>}<tool_call|>") ==
          "<tool_call>call:x{a:<\">v<\">}<tool_call>");
    CHECK(strip_control_tokens("x |> f, a <| b, List<int> y") == "x |> f, a <| b, List<int> y");   // code untouched
    // One <turn|> closes the model turn (generate() never feeds end-of-turn).
    const std::string u = native_user_turn("stuck on '<|channel>'");
    CHECK(u == "<turn|>\n<|turn>user\nstuck on '<channel>'<turn|>\n<|turn>model\n");
}

TEST(loop_guards_nudges_and_caps) {
    LoopGuardConfig gc;
    gc.check_patterns = {"flutter analyze", "flutter test"};
    LoopGuards g(gc);
    bool give_up = false;
    NativeStep fin;  fin.kind = NativeStepKind::Final;  fin.final_text = "done";
    NativeStep junk; junk.kind = NativeStepKind::JunkFinal;
    NativeStep cut;  cut.kind = NativeStepKind::CutNoCall;
    CHECK(g.on_no_call(fin, &give_up).empty() && !give_up);              // nothing edited: accept
    g.on_tool("edit_file", "", true, "ok: replaced 1 occurrence(s)");
    g.on_tool("bash", "ls -la", true, "[exit code: 0]");                // not a check
    CHECK(g.edited_since_check());
    std::string n = g.on_no_call(fin, &give_up);
    CHECK_MSG(n.find("run the checks") != std::string::npos, n);        // one reminder...
    CHECK(g.on_no_call(fin, &give_up).empty());                         // ...then accept
    g.on_tool("bash", "cd portfolio && flutter analyze && flutter test", true, "[exit code: 0]");
    CHECK(!g.edited_since_check());
    g.on_tool("edit_file", "", false, "error: old_string not found");    // failed edit: still checked
    CHECK(!g.edited_since_check());
    CHECK(g.on_no_call(cut, &give_up).find("cut off") != std::string::npos && !give_up);
    CHECK(!g.on_no_call(junk, &give_up).empty() && !give_up);
    CHECK(!g.on_no_call(junk, &give_up).empty() && !give_up);
    CHECK(g.on_no_call(junk, &give_up).empty() && give_up);             // 3rd empty answer: give up, not Done
    CHECK(g.steps_left_note(3).find("3 steps left") != std::string::npos);
    CHECK(g.steps_left_note(4).empty() && g.steps_left_note(2).empty());
}

TEST(native_step_seeds_differ_per_step_and_reproduce) {
    CHECK(native_step_seed(42, 0) != native_step_seed(42, 1));
    CHECK(native_step_seed(42, 7) == native_step_seed(42, 7));
    CHECK(native_step_seed(42, 3) != native_step_seed(43, 3));
}

TEST(step_token_guard_rules) {
    ControlTokenIds ids;
    ids.channel_open = 1; ids.channel_close = 2; ids.turn_open = 3; ids.tool_response = 4;
    float lg[8];
    auto fresh = [&] { for (float& v : lg) v = 0.f; };
    // Thinking off: "<|channel>" allowed only as the first token; never <|turn> / <|tool_response>.
    StepTokenGuard g(ids, /*prefilled=*/false, /*budget=*/0);
    fresh(); g.filter(lg, 8);
    CHECK(lg[1] == 0.f && std::isinf(lg[2]) && std::isinf(lg[3]) && std::isinf(lg[4]));
    CHECK(g.on_token(1));                                   // <|channel>
    fresh(); g.filter(lg, 8);
    CHECK(std::isinf(lg[1]) && lg[2] == 0.f);               // no 2nd open; close allowed
    CHECK(g.on_token(5)); CHECK(g.on_token(2));             // "thought", <channel|>
    fresh(); g.filter(lg, 8);
    CHECK(std::isinf(lg[1]) && std::isinf(lg[2]));          // the L3 loop is now impossible
    // Thinking on, block prefilled: no open at all; budget enforced.
    StepTokenGuard t(ids, /*prefilled=*/true, /*budget=*/3);
    fresh(); t.filter(lg, 8);
    CHECK(std::isinf(lg[1]) && lg[2] == 0.f);
    CHECK(t.on_token(5) && t.on_token(5) && t.on_token(5));
    CHECK(!t.on_token(5));                                  // 4th thought token: over budget
    t.force_close();
    fresh(); t.filter(lg, 8);
    CHECK(std::isinf(lg[2]) && !t.thought_open());
}

TEST(thought_blocks_are_stripped_from_answers) {
    const std::string th = "<|channel>thought\nThe analyzer says line 38 needs a finder first.<channel|>";
    CHECK(strip_thoughts(th + "Fixed it.") == "Fixed it.");
    CHECK(strip_thoughts("<|channel>thought\nunfinished reasoning") == "");
    NativeStep f = classify_native_step(th + "Done: all checks pass.", false, "", false);
    CHECK(f.kind == NativeStepKind::Final && f.final_text == "Done: all checks pass.");
    CHECK(classify_native_step(th, false, "", false).kind == NativeStepKind::JunkFinal);   // only thinking
    CHECK(classify_native_step(th + "<|tool_call>call:bash{command:<|\"|>ls<|\"|>}<tool_call|>", true, "", false).kind ==
          NativeStepKind::ToolCall);
}

TEST(check_digest_reports_fixed_still_new) {
    // v6-r1 L5: the same three errors on consecutive checks; line 15 never touched.
    std::vector<Diagnostic> a{
        {"error", "test/widget_test.dart", 15, "use_of_void_result", "This expression has a type of 'void' so its value can't be used. Try checking."},
        {"error", "test/widget_test.dart", 38, "argument_type_not_assignable", "The argument type 'String' can't be assigned to the parameter type 'FinderBase<Element>'."},
        {"error", "test/widget_test.dart", 38, "not_enough_positional_arguments", "2 positional arguments expected by 'enterText', but 1 found."}};
    CheckDigest dg;
    std::string s1 = dg.update(a);
    CHECK_MSG(has_text(s1, "3 error(s)") && !has_text(s1, "Compared"), s1);
    CHECK(has_text(s1, "test/widget_test.dart:15: error: This expression has a type of 'void' so its value can't be used."));
    CHECK(!has_text(s1, "Try checking"));                  // advice trimmed
    std::string s2 = dg.update(a);
    CHECK_MSG(has_text(s2, "0 fixed, 3 still failing, 0 new") && has_text(s2, ":15: error") && has_text(s2, "[still failing]"), s2);
    std::vector<Diagnostic> b{a[0], {"error", "test/widget_test.dart", 38, "undefined_named_parameter", "The named parameter 'into' isn't defined."}};
    std::string s3 = dg.update(b);
    CHECK_MSG(has_text(s3, "2 fixed, 1 still failing, 1 new"), s3);
    CHECK(has_text(dg.update({}), "no errors or warnings found") );
}

int main() {
    printf("== test_nishachar ==\n");
    return llmtest::run_all();
}
