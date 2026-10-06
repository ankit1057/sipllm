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

int main() {
    printf("== test_agent_tools ==\n");
    return llmtest::run_all();
}
