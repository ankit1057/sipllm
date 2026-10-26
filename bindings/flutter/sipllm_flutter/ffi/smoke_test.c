/* smoke_test.c — exercises the C ABI end to end from PURE C (proving the ABI is
 * C-callable, not just C++-linkable). Build via ffi/build_desktop.sh, or:
 *   clang -Iffi -c ffi/smoke_test.c && link with libsipllm_ffi + engine objs.
 * Usage: sip_smoke <model.gguf|model.llmw> */
#include "sipllm_ffi.h"

#include <stdio.h>
#include <stdlib.h>

static int32_t on_tok(const char* piece, int64_t id, void* user) {
  int* n = (int*)user;
  (*n)++;
  printf("%s", piece);
  fflush(stdout);
  return 1; /* continue */
}

/* Stop after 2 tokens to exercise the callback-driven cancel path. */
static int32_t on_tok_stop(const char* piece, int64_t id, void* user) {
  int* n = (int*)user;
  (*n)++;
  return (*n < 2) ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
    return 2;
  }
  sipllm_set_log_level(1);
  printf("== sipllm_ffi smoke ==\n");
  printf("version:        %s\n", sipllm_version());
  printf("hw concurrency: %d\n", sipllm_hardware_concurrency());
  printf("vulkan:         compiled=%d available=%d\n",
         sipllm_vulkan_compiled(), sipllm_vulkan_available());

  sipllm_params p;
  sipllm_params_default(&p);
  /* #78: the default asks for the automatic budget (half of available RAM). */
  if (p.ram_budget_bytes != SIPLLM_RAM_BUDGET_AUTO) {
    fprintf(stderr, "FAIL: sipllm_params_default must set SIPLLM_RAM_BUDGET_AUTO\n");
    return 1;
  }
  p.ram_budget_bytes = 0; /* unlimited streaming */
  p.threads = 0;          /* hardware_concurrency */

  char err[512];
  err[0] = 0;
  sipllm_ctx* ctx = sipllm_open(argv[1], &p, err, (int)sizeof(err));
  if (!ctx) {
    fprintf(stderr, "open failed: %s\n", err);
    return 1;
  }

  sipllm_model_info mi;
  if (sipllm_get_model_info(ctx, &mi) != 0) {
    fprintf(stderr, "model_info failed\n");
    sipllm_close(ctx);
    return 1;
  }
  printf("model:          arch=%s layers=%d heads=%d/%d dim=%lld vocab=%lld ctx=%lld tok=%d\n",
         mi.arch, mi.n_layers, mi.n_heads, mi.n_kv_heads, (long long)mi.dim,
         (long long)mi.vocab_size, (long long)mi.ctx_len, mi.tokenizer_kind);
  printf("threads:        %d\n", sipllm_get_threads(ctx));

  sipllm_sampler s;
  sipllm_sampler_default(&s);
  s.temperature = 0.0f; /* greedy */

  int count = 0;
  sipllm_stats st;
  printf("generate:       ");
  int32_t n = sipllm_generate(ctx, "hello", 8, &s, on_tok, &count, &st, err, (int)sizeof(err));
  printf("\n");
  if (n < 0) {
    fprintf(stderr, "generate failed: %s\n", err);
    sipllm_close(ctx);
    return 1;
  }
  printf("  -> %d tokens (cb saw %d) | decode %.1f tok/s | ttft %.4fs | peak %.1f MB | pinned %d/%d | streamed %.2f MB\n",
         n, count, st.decode_tok_s, st.ttft_s, st.peak_rss_bytes / 1e6,
         st.pinned_layers, st.n_layers, st.bytes_read / 1e6);

  /* callback-cancel path */
  int c2 = 0;
  sipllm_reset(ctx);
  int32_t n2 = sipllm_generate(ctx, "hello", 64, &s, on_tok_stop, &c2, NULL, err, (int)sizeof(err));
  printf("cancel-via-cb:  requested 64, stopped after %d generated tokens\n", n2);

  /* embeddings */
  int dim = sipllm_embed_dim(ctx);
  printf("embed_dim:      %d\n", dim);
  float* emb = (float*)malloc(sizeof(float) * (size_t)dim);
  int32_t ed = sipllm_embed(ctx, "the capital of france is", SIPLLM_POOL_LAST, emb, dim, err,
                            (int)sizeof(err));
  if (ed < 0) {
    fprintf(stderr, "embed failed: %s\n", err);
  } else {
    double norm = 0;
    for (int i = 0; i < dim; ++i) norm += (double)emb[i] * emb[i];
    printf("embed:          dim=%d L2=%.4f first3=[% .4f % .4f % .4f]\n", ed, norm,
           emb[0], dim > 1 ? emb[1] : 0.f, dim > 2 ? emb[2] : 0.f);
  }
  free(emb);

  sipllm_close(ctx);
  printf("== OK ==\n");
  return 0;
}
