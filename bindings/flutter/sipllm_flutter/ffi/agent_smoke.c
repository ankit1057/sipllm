/* agent_smoke.c — drives the stepped agent ABI from PURE C on a real model:
 * the caller owns the tools (calc and a fixed-content read_file here) and the
 * engine owns the conversation, the parsing and the step bound.
 *
 *   ./ffi/build_desktop.sh
 *   clang -Iffi ffi/agent_smoke.c -Lbuild/desktop -lsipllm_ffi \
 *         -Wl,-rpath,build/desktop -o build/desktop/agent_smoke
 *   build/desktop/agent_smoke <instruct-model.gguf>
 *
 * Exit 0 only if every goal reaches a final answer carrying the tool's result.
 * Also checks the ABI's guard rails: stepping with a call pending, submitting
 * with none pending, adding a tool mid-run, and stepping after the end. */
#include "sipllm_ffi.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int32_t on_tok(const char* piece, int64_t id, void* user) {
  (void)id; (void)user;
  fputs(piece, stdout);
  fflush(stdout);
  return 1;
}

static const char* arg(sipllm_agent* a, const char* key) {
  for (int32_t i = 0; i < sipllm_agent_arg_count(a); ++i)
    if (strcmp(sipllm_agent_arg_key(a, i), key) == 0) return sipllm_agent_arg_value(a, i);
  return "";
}

/* "<int> <op> <int>" -> result text; returns 0 on a malformed expression. */
static int calc(const char* expr, char* out, size_t cap) {
  long long x = 0, y = 0;
  char op = 0;
  if (sscanf(expr, "%lld %c %lld", &x, &op, &y) != 3) return 0;
  long long r;
  switch (op) {
    case '+': r = x + y; break;
    case '-': r = x - y; break;
    case '*': r = x * y; break;
    case '/': if (y == 0) return 0; r = x / y; break;
    default: return 0;
  }
  snprintf(out, cap, "%lld", r);
  return 1;
}

static int failures = 0;
#define EXPECT(cond, what) do { if (!(cond)) { printf("\nFAIL %s\n", what); ++failures; } } while (0)

static void run_goal(sipllm_agent* a, const char* goal, const char* want_tool, const char* want_final) {
  char err[512] = {0};
  printf("\n== goal: %s\n", goal);
  EXPECT(sipllm_agent_start(a, goal, err, sizeof err) == 0, "start");
  EXPECT(sipllm_agent_add_tool(a, "late", "added mid-run") == -1, "add_tool is refused mid-run");
  int called = 0, ev = 0;
  for (;;) {
    sipllm_stats st;
    ev = sipllm_agent_step(a, NULL, on_tok, NULL, &st, err, sizeof err);
    printf("\n   [step %d: event=%d prompt_tokens=%d gen_tokens=%d prefill=%.2fs decode=%.2fs]\n",
           sipllm_agent_steps(a), ev, st.prompt_tokens, st.gen_tokens, st.prefill_s, st.decode_s);
    if (ev != SIPLLM_AGENT_TOOL_CALL) break;
    const char* name = sipllm_agent_tool_name(a);
    printf("   [call %s %s]\n", name, sipllm_agent_tool_args_json(a));
    EXPECT(sipllm_agent_step(a, NULL, NULL, NULL, NULL, err, sizeof err) == SIPLLM_AGENT_ERROR,
           "step is refused while a call is pending");
    char result[256];
    int ok = 1;
    if (strcmp(name, "calc") == 0) {
      ok = calc(arg(a, "expr"), result, sizeof result);
      if (!ok) snprintf(result, sizeof result, "error: bad expression");
    } else if (strcmp(name, "read_file") == 0) {
      ok = strcmp(arg(a, "path"), "notes.txt") == 0;
      snprintf(result, sizeof result, "%s", ok ? "The access code is TANGERINE-4711." : "error: file not found");
    } else {
      ok = 0;
      snprintf(result, sizeof result, "error: unknown tool");
    }
    if (strcmp(name, want_tool) == 0) called = 1;
    printf("   [result ok=%d %s]\n", ok, result);
    if (sipllm_agent_submit_result(a, result, ok) == 0) { ev = sipllm_agent_stop_reason(a); break; }
  }
  EXPECT(ev != SIPLLM_AGENT_ERROR, err);
  EXPECT(called, "the intended tool was called");
  EXPECT(ev == SIPLLM_AGENT_FINAL, "ended with a final answer");
  EXPECT(strstr(sipllm_agent_text(a), want_final) != NULL, "final answer carries the tool result");
  EXPECT(sipllm_agent_submit_result(a, "x", 1) == -1, "submit is refused with no call pending");
  EXPECT(sipllm_agent_step(a, NULL, NULL, NULL, NULL, err, sizeof err) == ev, "a finished run keeps its end");
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <instruct-model.gguf>\n", argv[0]);
    return 2;
  }
  sipllm_set_log_level(1);
  char err[512] = {0};
  sipllm_params p;
  sipllm_params_default(&p);
  p.fast_quant = 1;
  sipllm_ctx* ctx = sipllm_open(argv[1], &p, err, sizeof err);
  if (!ctx) { fprintf(stderr, "open failed: %s\n", err); return 1; }

  sipllm_agent_params ap;
  sipllm_agent_params_default(&ap);
  sipllm_agent* a = sipllm_agent_new(ctx, &ap, err, sizeof err);
  if (!a) { fprintf(stderr, "agent_new failed: %s\n", err); return 1; }
  EXPECT(sipllm_agent_step(a, NULL, NULL, NULL, NULL, err, sizeof err) == SIPLLM_AGENT_ERROR,
         "step is refused before start");
  EXPECT(sipllm_agent_add_param(a, "nope", "x", SIPLLM_PARAM_STRING, 1, "") == -1,
         "add_param is refused for an undeclared tool");
  sipllm_agent_add_tool(a, "calc", "Evaluate an integer arithmetic expression '<int> <op> <int>', op one of + - * /.");
  sipllm_agent_add_param(a, "calc", "expr", SIPLLM_PARAM_STRING, 1, "the expression, e.g. '6 * 7'");
  sipllm_agent_add_tool(a, "read_file", "Read a text file from the working directory.");
  sipllm_agent_add_param(a, "read_file", "path", SIPLLM_PARAM_STRING, 1, "relative path of the file");

  run_goal(a, "What is 1234 * 5678? Use the calc tool.", "calc", "7006652");
  run_goal(a, "Read the file notes.txt with the read_file tool and tell me the access code in it.",
           "read_file", "TANGERINE-4711");

  sipllm_agent_free(a);
  sipllm_close(ctx);
  printf("\nagent_smoke: %s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
