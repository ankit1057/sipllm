// test_agent_tools.cpp — the Nishachar system tools (read_file, write_file,
// edit_file, list_dir, bash, grep_search) against a real temp directory.
#include "llm/tools.h"
#include "tests/test_util.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>

using namespace llm;
namespace fs = std::filesystem;

static bool has(const std::string& s, const std::string& sub) {
    return s.find(sub) != std::string::npos;
}

static ToolCall mk(const std::string& name,
                   std::vector<ToolCallArg> args) {
    ToolCall c;
    c.name = name;
    c.args = std::move(args);
    return c;
}

// Fresh workdir per test.
static std::string fresh_dir(const std::string& name) {
    std::string d = llmtest::scratch_path("agent-tools-" + name);
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d;
}

static void put(const std::string& path, const std::string& text) {
    fs::create_directories(fs::path(path).parent_path());
    std::ofstream f(path, std::ios::binary);
    f << text;
}

static std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::ostringstream o;
    o << f.rdbuf();
    return o.str();
}

TEST(register_includes_edit_file) {
    ToolRegistry reg;
    std::unordered_map<std::string, ToolHandler> h;
    register_system_tools(reg, h, ".");
    for (const char* n : {"read_file", "write_file", "edit_file", "list_dir", "bash", "grep_search"}) {
        CHECK_MSG(reg.find(n) != nullptr, std::string("missing tool ") + n);
        CHECK_MSG(h.count(n) == 1, std::string("missing handler ") + n);
    }
    int count = 0;
    register_system_tools([&](ToolDef, ToolHandler) { ++count; }, ".");
    CHECK(count == 6);
}

TEST(read_file_line_numbers_and_paging) {
    std::string d = fresh_dir("read");
    std::string body;
    for (int i = 1; i <= 10; ++i) body += "line" + std::to_string(i) + "\n";
    put(d + "/a.txt", body);
    ToolHandler rd = make_read_file_handler(d);

    std::string all = rd(mk("read_file", {{"path", "a.txt"}}));
    CHECK(has(all, "1\tline1\n"));
    CHECK(has(all, "10\tline10\n"));
    CHECK(!has(all, "[showing"));

    std::string part = rd(mk("read_file", {{"path", "a.txt"}, {"offset", "3"}, {"limit", "2"}}));
    CHECK(has(part, "3\tline3\n"));
    CHECK(has(part, "4\tline4\n"));
    CHECK(!has(part, "line5"));
    CHECK(!has(part, "line2\n"));
    CHECK(has(part, "of 10"));
    CHECK(has(part, "offset=5"));

    CHECK(has(rd(mk("read_file", {{"path", "a.txt"}, {"offset", "99"}})), "past end"));
    CHECK(has(rd(mk("read_file", {{"path", "nope.txt"}})), "not found"));
}

TEST(read_file_caps_large_files) {
    std::string d = fresh_dir("readbig");
    std::string body;
    for (int i = 1; i <= 1000; ++i) body += "x" + std::to_string(i) + "\n";
    put(d + "/big.txt", body);
    std::string out = make_read_file_handler(d)(mk("read_file", {{"path", "big.txt"}}));
    CHECK(has(out, "200\tx200\n"));
    CHECK(!has(out, "x201\n"));
    CHECK(has(out, "of 1000; use offset=201"));
    CHECK(out.size() < 6000);
    // Long lines hit the byte cap first; paging note still points at the next line.
    std::string wide;
    for (int i = 1; i <= 100; ++i) wide += std::string(200, 'w') + "\n";
    put(d + "/wide.txt", wide);
    std::string w = make_read_file_handler(d)(mk("read_file", {{"path", "wide.txt"}}));
    CHECK_MSG(w.size() < 6000 && has(w, "of 100; use offset="), w.substr(w.size() > 200 ? w.size() - 200 : 0));
}

TEST(write_file_creates_dirs_and_reports) {
    std::string d = fresh_dir("write");
    ToolHandler wr = make_write_file_handler(d);
    std::string r = wr(mk("write_file", {{"path", "sub/deep/f.txt"}, {"content", "a\nb\nc"}}));
    CHECK(has(r, "ok: wrote 5 bytes (3 lines)"));
    CHECK(slurp(d + "/sub/deep/f.txt") == "a\nb\nc");
}

TEST(edit_file_unique_replace) {
    std::string d = fresh_dir("edit1");
    put(d + "/m.cpp", "int a = 1;\nint b = 2;\nint c = 3;\n");
    ToolHandler ed = make_edit_file_handler(d);
    std::string r = ed(mk("edit_file", {{"path", "m.cpp"}, {"old_string", "int b = 2;"},
                                        {"new_string", "int b = 42;"}}));
    CHECK_MSG(has(r, "ok: replaced 1 occurrence"), r);
    CHECK(has(r, "2\tint b = 42;"));
    CHECK(slurp(d + "/m.cpp") == "int a = 1;\nint b = 42;\nint c = 3;\n");
}

TEST(edit_file_not_found_and_ambiguous) {
    std::string d = fresh_dir("edit2");
    put(d + "/m.txt", "foo();\nbar();\nfoo();\n");
    ToolHandler ed = make_edit_file_handler(d);

    std::string nf = ed(mk("edit_file", {{"path", "m.txt"}, {"old_string", "baz();"}, {"new_string", "q"}}));
    CHECK_MSG(has(nf, "error: old_string not found"), nf);

    // First line present but the block doesn't match: point at it.
    std::string near = ed(mk("edit_file", {{"path", "m.txt"}, {"old_string", "  bar();\nzzz"}, {"new_string", "q"}}));
    CHECK_MSG(has(near, "not found") && has(near, "2\tbar();"), near);

    std::string amb = ed(mk("edit_file", {{"path", "m.txt"}, {"old_string", "foo();"}, {"new_string", "x();"}}));
    CHECK_MSG(has(amb, "matches 2 times"), amb);
    CHECK(has(amb, "1\tfoo();"));
    CHECK(has(amb, "3\tfoo();"));
    CHECK(slurp(d + "/m.txt") == "foo();\nbar();\nfoo();\n");   // untouched

    std::string all = ed(mk("edit_file", {{"path", "m.txt"}, {"old_string", "foo();"},
                                          {"new_string", "x();"}, {"replace_all", "true"}}));
    CHECK_MSG(has(all, "replaced 2 occurrence"), all);
    CHECK(slurp(d + "/m.txt") == "x();\nbar();\nx();\n");

    CHECK(has(ed(mk("edit_file", {{"path", "m.txt"}, {"old_string", ""}, {"new_string", "a"}})), "error"));
    CHECK(has(ed(mk("edit_file", {{"path", "nope"}, {"old_string", "a"}, {"new_string", "b"}})), "not found"));
}

TEST(edit_file_via_parser) {
    // The model's JSON (escaped newlines/quotes) flows through the parser intact.
    std::string d = fresh_dir("edit3");
    put(d + "/p.py", "def f():\n    return \"a\"\n");
    ToolRegistry reg;
    std::unordered_map<std::string, ToolHandler> h;
    register_system_tools(reg, h, d);
    ToolParser parser(reg);
    parser.set_markers({"<tool_call>", "</tool_call>"});
    CHECK(parser.feed("<tool_call>{\"name\": \"edit_file\", \"arguments\": {\"path\": \"p.py\", "
                      "\"old_string\": \"    return \\\"a\\\"\", \"new_string\": \"    return \\\"b\\\"\\n    # done\"}}"
                      "</tool_call>"));
    std::string r = h["edit_file"](parser.parsed_call());
    CHECK_MSG(has(r, "ok: replaced 1"), r);
    CHECK(slurp(d + "/p.py") == "def f():\n    return \"b\"\n    # done\n");
}

TEST(bash_reports_exit_code) {
    std::string d = fresh_dir("bash");
    ToolHandler sh = make_bash_handler(d);
    std::string ok = sh(mk("bash", {{"command", "echo hi"}}));
    CHECK_MSG(has(ok, "hi\n[exit code: 0]"), ok);
    std::string bad = sh(mk("bash", {{"command", "echo oops >&2; exit 3"}}));
    CHECK_MSG(has(bad, "oops") && has(bad, "[exit code: 3]"), bad);
    std::string to = sh(mk("bash", {{"command", "exit 124"}}));
    CHECK_MSG(has(to, "[exit code: 124] (timed out)"), to);
    std::string quiet = sh(mk("bash", {{"command", "true"}}));
    CHECK_MSG(has(quiet, "(no output)\n[exit code: 0]"), quiet);
    std::string pwd = sh(mk("bash", {{"command", "pwd"}}));
    CHECK_MSG(has(pwd, "agent-tools-bash"), pwd);
}

TEST(path_escape_refused) {
    std::string d = fresh_dir("jail");
    std::string outside = llmtest::scratch_path("agent-tools-outside.txt");
    put(outside, "secret\n");
    put(d + "/in.txt", "ok\n");
    // A sibling dir sharing the name prefix must not pass a string-prefix check.
    std::string sib = d + "2";
    fs::create_directories(sib);
    put(sib + "/s.txt", "sib\n");

    ToolHandler rd = make_read_file_handler(d), wr = make_write_file_handler(d),
                ed = make_edit_file_handler(d), ls = make_list_dir_handler(d),
                gr = make_grep_search_handler(d);
    auto refused = [](const std::string& r) { return has(r, "outside the working directory"); };

    CHECK(refused(rd(mk("read_file", {{"path", "../agent-tools-outside.txt"}}))));
    CHECK(refused(rd(mk("read_file", {{"path", outside}}))));
    CHECK(refused(rd(mk("read_file", {{"path", "sub/../../agent-tools-outside.txt"}}))));
    CHECK(refused(rd(mk("read_file", {{"path", sib + "/s.txt"}}))));
    CHECK(refused(rd(mk("read_file", {{"path", "/etc/passwd"}}))));
    CHECK(refused(wr(mk("write_file", {{"path", "../evil.txt"}, {"content", "x"}}))));
    CHECK(!fs::exists(llmtest::scratch_path("evil.txt")));
    CHECK(refused(ed(mk("edit_file", {{"path", outside}, {"old_string", "secret"}, {"new_string", "x"}}))));
    CHECK(slurp(outside) == "secret\n");
    CHECK(refused(ls(mk("list_dir", {{"path", ".."}}))));
    CHECK(refused(gr(mk("grep_search", {{"pattern", "secret"}, {"path", ".."}}))));

    // Symlink pointing out of the workdir is refused too (and grep skips it).
    std::error_code ec;
    fs::create_symlink(outside, d + "/link.txt", ec);
    if (!ec) {
        CHECK(refused(rd(mk("read_file", {{"path", "link.txt"}}))));
        CHECK(!has(gr(mk("grep_search", {{"pattern", "secret"}})), "secret\n"));
    }

    // Inside paths (relative, ./, absolute-inside, inner ..) still work.
    CHECK(has(rd(mk("read_file", {{"path", "in.txt"}})), "1\tok"));
    CHECK(has(rd(mk("read_file", {{"path", "./x/../in.txt"}})), "1\tok"));
    CHECK(has(rd(mk("read_file", {{"path", d + "/in.txt"}})), "1\tok"));
    CHECK(has(ls(mk("list_dir", {{"path", "."}})), "in.txt"));
}

TEST(list_dir_and_grep) {
    std::string d = fresh_dir("lsgrep");
    put(d + "/src/a.cpp", "int main() { return 0; }\n");
    put(d + "/notes.md", "TODO: fix\n");
    std::string ls = make_list_dir_handler(d)(mk("list_dir", {}));
    CHECK(has(ls, "[DIR]  src/"));
    CHECK(has(ls, "[FILE] notes.md"));
    std::string g = make_grep_search_handler(d)(mk("grep_search", {{"pattern", "return 0"}}));
    CHECK_MSG(has(g, "src/a.cpp:1:"), g);
    CHECK(!has(g, d));   // relative display paths
}

TEST(print_schemas) {
    ToolRegistry reg;
    std::unordered_map<std::string, ToolHandler> h;
    register_system_tools(reg, h, ".");
    if (std::getenv("PRINT_TOOL_SCHEMAS")) printf("%s", reg.system_prompt_block().c_str());
    CHECK(!reg.system_prompt_block().empty());
}

TEST(edit_file_line_range_delete_and_replace) {
    std::string d = fresh_dir("editlines");
    put(d + "/w.dart", "a\nb\nc\nd\n");
    ToolHandler ed = make_edit_file_handler(d);
    // Delete one line by number (no new_string), the case Gemma 4 could not
    // express with exact-text replacement.
    std::string r = ed(mk("edit_file", {{"path", "w.dart"}, {"start_line", "2"}, {"end_line", "2"},
                                        {"first_line", "b"}}));
    CHECK_MSG(has(r, "ok: deleted lines 2-2"), r);
    CHECK_MSG(has(r, "removed: \"b\"") && has(r, "shifted by -1"), r);
    CHECK(slurp(d + "/w.dart") == "a\nc\nd\n");
    // Replace a range; a missing trailing newline is added; first_line ignores indentation.
    r = ed(mk("edit_file", {{"path", "w.dart"}, {"start_line", "2"}, {"end_line", "3"}, {"first_line", "  c"},
                            {"new_string", "X"}}));
    CHECK_MSG(has(r, "ok: replaced lines 2-3"), r);
    CHECK(slurp(d + "/w.dart") == "a\nX\n");
    // Last line, no end_line.
    r = ed(mk("edit_file", {{"path", "w.dart"}, {"start_line", "2"}, {"first_line", "X"}}));
    CHECK(slurp(d + "/w.dart") == "a\n");
    CHECK(has(ed(mk("edit_file", {{"path", "w.dart"}, {"start_line", "5"}, {"first_line", "a"}})), "outside the file"));
    CHECK(has(ed(mk("edit_file", {{"path", "w.dart"}, {"start_line", "x"}})), "must be integers"));
    // Identical old/new now points at the line-number delete.
    put(d + "/v.dart", "keep\n");
    CHECK(has(ed(mk("edit_file", {{"path", "v.dart"}, {"old_string", "keep"}, {"new_string", "keep"}})),
              "start_line"));
}

TEST(edit_file_line_range_rejects_stale_line_numbers) {
    // The L4b portfolio run: after earlier edits shifted the file, a range
    // from the first read_file wiped the wrong ~20 lines. first_line catches it.
    std::string d = fresh_dir("editstale");
    const std::string orig = "class A {\n  int x;\n}\nclass B {\n  int y;\n}\n";
    put(d + "/s.dart", orig);
    ToolHandler ed = make_edit_file_handler(d);
    // Missing first_line: refused, shows the lines it would touch.
    std::string r = ed(mk("edit_file", {{"path", "s.dart"}, {"start_line", "4"}, {"end_line", "6"}}));
    CHECK_MSG(has(r, "need first_line") && has(r, "4\tclass B {"), r);
    // Stale number: caller thinks "class B {" is line 3; it is line 4. Nothing changes.
    r = ed(mk("edit_file", {{"path", "s.dart"}, {"start_line", "3"}, {"end_line", "5"},
                            {"first_line", "class B {"}, {"new_string", "class C {}"}}));
    CHECK_MSG(has(r, "line 3 is \"}\"") && has(r, "now at line 4") && has(r, "Nothing was changed"), r);
    CHECK(slurp(d + "/s.dart") == orig);
    // Text that no longer exists anywhere: told to re-read.
    r = ed(mk("edit_file", {{"path", "s.dart"}, {"start_line", "1"}, {"first_line", "class Z {"}}));
    CHECK_MSG(has(r, "read_file"), r);
    CHECK(slurp(d + "/s.dart") == orig);
}

TEST(edit_checks_reject_breaking_changes_but_allow_repairs) {
    // Stand-in for `dart format -o none {file}`: a file "parses" unless it
    // contains SYNTAX_ERROR. The L4r portfolio run left a missing comma that
    // this check would have bounced in the same step.
    std::string d = fresh_dir("editcheck");
    std::vector<EditCheck> checks{{".dart", "! grep -q SYNTAX_ERROR {file}"}};
    ToolHandler ed = with_edit_checks(make_edit_file_handler(d), d, checks);
    ToolHandler wr = with_edit_checks(make_write_file_handler(d), d, checks);
    put(d + "/a.dart", "line one\nline two\n");
    // 1. Breaking a clean file: undone, file unchanged.
    std::string r = ed(mk("edit_file", {{"path", "a.dart"}, {"old_string", "line two"}, {"new_string", "SYNTAX_ERROR"}}));
    CHECK_MSG(has(r, "error: change rejected") && has(r, "restored unchanged"), r);
    CHECK(slurp(d + "/a.dart") == "line one\nline two\n");
    // 2. Overwriting a clean file with broken content: also undone.
    r = wr(mk("write_file", {{"path", "a.dart"}, {"content", "SYNTAX_ERROR\n"}}));
    CHECK_MSG(has(r, "change rejected"), r);
    CHECK(slurp(d + "/a.dart") == "line one\nline two\n");
    // 3. Already-broken file: an unrelated edit goes through, with a warning.
    put(d + "/b.dart", "SYNTAX_ERROR\nkeep\n");
    r = ed(mk("edit_file", {{"path", "b.dart"}, {"old_string", "keep"}, {"new_string", "kept"}}));
    CHECK_MSG(has(r, "ok: replaced") && has(r, "does not parse yet"), r);
    CHECK(slurp(d + "/b.dart") == "SYNTAX_ERROR\nkept\n");
    // 4. The repair itself goes through cleanly.
    r = ed(mk("edit_file", {{"path", "b.dart"}, {"old_string", "SYNTAX_ERROR"}, {"new_string", "fixed"}}));
    CHECK_MSG(has(r, "ok: replaced") && !has(r, "does not parse"), r);
    // 5. A new broken file is kept (nothing to restore), with a warning.
    r = wr(mk("write_file", {{"path", "c.dart"}, {"content", "SYNTAX_ERROR"}}));
    CHECK_MSG(has(r, "ok: wrote") && has(r, "does not parse yet"), r);
    // 6. Other extensions are not checked; tool errors pass through untouched.
    r = wr(mk("write_file", {{"path", "n.txt"}, {"content", "SYNTAX_ERROR"}}));
    CHECK_MSG(has(r, "ok: wrote") && !has(r, "parse"), r);
    r = ed(mk("edit_file", {{"path", "a.dart"}, {"old_string", "absent"}, {"new_string", "x"}}));
    CHECK_MSG(has(r, "error: old_string not found"), r);
}

TEST(read_file_page_size_and_last_line_guard) {
    std::string d = fresh_dir("readpage");
    std::string big;
    for (int i = 1; i <= 400; ++i) big += "line number " + std::to_string(i) + " of a medium sized dart file\n";
    put(d + "/m.dart", big);
    // Default 5000-byte page cannot show all 400 lines; a 32 KB page can.
    CHECK(has(make_read_file_handler(d)(mk("read_file", {{"path", "m.dart"}})), "use offset="));
    std::string all = make_read_file_handler(d, 32000)(mk("read_file", {{"path", "m.dart"}}));
    CHECK_MSG(has(all, "400\tline number 400") && !has(all, "use offset="), all.substr(all.size() - 200));
    // last_line must match end_line's current text, or nothing changes.
    ToolHandler ed = make_edit_file_handler(d);
    std::string r = ed(mk("edit_file", {{"path", "m.dart"}, {"start_line", "2"}, {"end_line", "3"},
                                        {"first_line", "line number 2 of a medium sized dart file"},
                                        {"last_line", "line number 4 of a medium sized dart file"}}));
    CHECK_MSG(has(r, "not your last_line") && has(r, "Nothing was changed"), r);
    CHECK(slurp(d + "/m.dart") == big);
}

TEST(parse_diagnostics_real_formats_and_noise) {
    // Lines captured from the v6-r1 broken-L7 snapshot (flutter 3.x / dart 3.x).
    const std::string machine =
        "ERROR|COMPILE_TIME_ERROR|UNDEFINED_METHOD|/tmp/p/lib/main.dart|203|48|8|The method 'scrollTo' isn't defined for the type 'PortfolioPage'.\n"
        "INFO|LINT|PREFER_CONST|/tmp/p/lib/main.dart|10|3|5|Use const.\n";
    auto m = parse_diagnostics(machine);
    CHECK(m.size() == 2 && m[0].severity == "error" && m[0].line == 203 && m[0].code == "undefined_method" &&
          m[0].file == "/tmp/p/lib/main.dart" && m[1].severity == "info");
    const std::string human =
        "Analyzing portfolio...\n"
        "  error \xE2\x80\xA2 Undefined name '_projectsKey'. Try correcting the name to one that is defined, or defining the name \xE2\x80\xA2 lib/main.dart:203:57 \xE2\x80\xA2 undefined_identifier\n"
        "warning \xE2\x80\xA2 The value of the field '_x' isn't used \xE2\x80\xA2 lib/widgets/a.dart:7:3 \xE2\x80\xA2 unused_field\n"
        "2 issues found. (ran in 1.2s)\n";
    auto h = parse_diagnostics(human);
    CHECK(h.size() == 2 && h[0].file == "lib/main.dart" && h[0].line == 203 && h[0].code == "undefined_identifier" &&
          h[1].severity == "warning" && h[1].file == "lib/widgets/a.dart");
    const std::string compiler =
        "test/widget_test.dart:15:31: Error: This expression has type 'void' and can't be used.\n"
        "test/widget_test.dart:38:29: Error: Too few positional arguments: 2 required, 1 given.\n"
        "src/x.cpp:4:2: warning: unused variable 'y'\n";
    auto cmp = parse_diagnostics(compiler);
    CHECK(cmp.size() == 3 && cmp[0].line == 15 && cmp[1].line == 38 && cmp[2].severity == "warning");
    // dart analyze CLI (dash format), captured from the v6.2 smoke test.
    auto dd = parse_diagnostics("Analyzing a.dart...\n\n  error - a.dart:1:12 - Undefined name 'missingName'. Try correcting the name to one that is defined, or defining the name. - undefined_identifier\n\n1 issue found.\n");
    CHECK(dd.size() == 1 && dd[0].severity == "error" && dd[0].file == "a.dart" && dd[0].line == 1 &&
          dd[0].code == "undefined_identifier" && dd[0].message.compare(0, 29, "Undefined name 'missingName'.") == 0);
    CHECK(parse_diagnostics("All tests passed!\nexit 0").empty());
    const std::string pub =
        "Resolving dependencies...\nDownloading packages...\n  clock 1.1.2 (1.1.3 available)\n"
        "  vector_math 2.2.0 (2.4.3 available)\nGot dependencies!\n"
        "8 packages have newer versions incompatible with dependency constraints.\n"
        "Try `flutter pub outdated` for more information.\nAnalyzing portfolio...\nNo issues found!";
    CHECK_MSG(strip_check_noise(pub) == "Analyzing portfolio...\nNo issues found!", strip_check_noise(pub));
}

TEST(warn_only_edit_check_appends_findings_never_undoes) {
    std::string d = fresh_dir("editwarn");
    // Fake analyzer: one machine-format error per line containing BAD.
    const std::string cmd = "grep -n BAD {file} | sed 's/^\\([0-9]*\\):.*/ERROR|COMPILE_TIME_ERROR|UNDEFINED_IDENTIFIER|x.dart|\\1|1|1|Undefined name BAD./'";
    std::vector<EditCheck> checks{{".dart", cmd, true}};
    ToolHandler ed = with_edit_checks(make_edit_file_handler(d), d, checks);
    put(d + "/w.dart", "a\nb\n");
    std::string r = ed(mk("edit_file", {{"path", "w.dart"}, {"old_string", "b"}, {"new_string", "BAD"}}));
    CHECK_MSG(has(r, "ok: replaced") && has(r, "has 1 error(s)") && has(r, "line 2: error: Undefined name BAD."), r);
    CHECK(slurp(d + "/w.dart") == "a\nBAD\n");                      // kept, not undone
    r = ed(mk("edit_file", {{"path", "w.dart"}, {"old_string", "BAD"}, {"new_string", "fine"}}));
    CHECK_MSG(has(r, "[analyzer] no errors or warnings in w.dart"), r);
}

int main() {
    printf("== test_agent_tools ==\n");
    return llmtest::run_all();
}
