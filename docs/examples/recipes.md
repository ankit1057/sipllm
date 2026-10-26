# Recipes

<div class="hero"><p class="lead"><strong>Copy-paste starting points, one per common use.</strong> Every flag and function on this page exists in the source: the CLI's argument parser in <code>main.cpp</code>, the server in <code>server/server.cpp</code>, the C ABI in <code>sipllm_ffi.h</code> and the Dart package in <code>bindings/flutter/</code>. Wherever output is shown, it is illustrative: the shape is real, the values are elided. To decide whether SipLLM fits your application at all, start with <a href="applications.html">Applications</a>.</p></div>

> [!NOTE] Conventions
> Commands run from the root of a SipLLM checkout. `$MODEL` is a path to a GGUF file. The examples use TinyLlama-1.1B Q4_K_M because the measured numbers on this site are for that file. Sizes for `--ram-budget` accept `K`, `M` or `G` suffixes (powers of 1024), for example `512M` or `1.5G`; `0` means stream everything with no ceiling.

## Build and fetch a model

```bash
git clone https://github.com/ankit1057/sipllm.git && cd sipllm
make                  # build/llm plus tools (inspect_gguf, dump_logits, ...)
make server           # build/llm_server, needed only for recipe 4

./sipllm pull tinyllama:q4_k_m             # cached under ~/.sipllm/models
MODEL="$(./sipllm which tinyllama:q4_k_m)"  # prints the local path
```

If you installed with `install.sh` instead, `sipllm run <name> [engine args...]` passes every argument after the model name to the engine, so each `./build/llm "$MODEL" ...` line below also works as `sipllm run tinyllama:q4_k_m ...`.

<!-- verify: make && make server && ./sipllm pull tinyllama:q4_k_m && ls -l "$(./sipllm which tinyllama:q4_k_m)" -->

## 1. Run a model in a fixed RAM budget

Give the engine a hard peak-RSS ceiling. The memory planner pins as many layers as fit and streams the rest.

```bash
./build/llm "$MODEL" -p "The capital of France is" -n 40 --greedy \
  --ctx 512 --ram-budget 256M
```

- Before loading, the planner prints its execution plan to stderr. If the budget cannot be met, the run stops with `RAM budget impossible. Use --ram-budget-force to run anyway.`
- The planner prices the KV cache at the full context. Pass an explicit `--ctx` that matches your use; without one, it steps the context down until the plan fits.
- When generation ends, the stats block shows `pinned layers: N / M` and how many MB were streamed from disk. Its `peak rss` line is the resident size at the moment generation ends. For the true peak, wrap the command in `/usr/bin/time` (recipe 8).

To run a model much larger than your free RAM, use the exact settings of the measured 8B and 13B runs. Expect well under 1 tok/s at this extreme (0.55 tok/s for the 8B on an Apple M3):

```bash
./sipllm pull llama3.1:8b:q4_k_m
./build/llm "$(./sipllm which llama3.1:8b:q4_k_m)" -p "The capital of France is" \
  -n 20 --greedy --ctx 512 --stream-lm-head --no-async
```

<!-- verify: ./build/llm "$MODEL" -p "The capital of France is" -n 40 --greedy --ctx 512 --ram-budget 256M ; then the same with --ram-budget 64M to see the refusal message -->

## 2. Fully resident, fast mode

When the machine has the RAM, pin every layer and turn on the int8 kernels.

```bash
./build/llm "$MODEL" -p "The capital of France is" -n 64 --greedy \
  --ctx 512 --ram-budget 768M --fast
```

- In the measured run, 768M was enough to pin all 22 layers of this file at `--ctx 512`. If the stats block shows fewer, raise the budget.
- `--fast` covers Q8_0, Q4_K and Q6_K weights. It quantises activations to int8, so output can differ from the exact path at near-ties. The stats line `fast kernel:` reports `on`, or `off (no int8 kernel for this CPU)`.
- Optional and experimental: `SIPLLM_FAST_LAYOUT=il4 ./build/llm ... --fast` repacks resident Q4_K and Q6_K matrices for the interleaved kernels. The scorecard's best M3 decode figure used it.
- On phones with big and little cores, add `--schedule fixed16`. On the measured 2+6-core phone, it decoded at more than twice the speed of `proportional2`.

<!-- verify: ./build/llm "$MODEL" -p "The capital of France is" -n 64 --greedy --ctx 512 --ram-budget 768M --fast  (check "pinned layers: 22 / 22" and "fast kernel: on") -->

## 3. Interactive chat

```bash
./build/llm "$MODEL" --chat --ctx 2048 --ram-budget 768M --fast
```

- `--chat` (or `-i`) opens a REPL. Commands: `/help`, `/clear` (resets history and KV cache), `/save <path>` (writes the session), `/bye` or `/exit`. Start a line with `"""` to type several lines, and end with `"""`.
- Interactive mode formats the conversation with a chat template chosen from the model's metadata. One-shot `-p` runs send raw text with no template.
- Interactive mode turns on context reuse, so each turn reprocesses only the new tail. It generates up to 512 tokens per reply unless you pass `-n`.
- `--save-session F` writes the session when you exit, and `--load-session F` restores it on the next start. A session file only loads into the model that wrote it.

<!-- verify: ./build/llm "$MODEL" --chat --ctx 2048 --ram-budget 768M --fast ; type a question, then /save /tmp/s.sips, then /bye ; restart with --load-session /tmp/s.sips -->

## 4. Local OpenAI-compatible server (this machine only)

> [!CAUTION] Read before starting the server
> Since v0.6.0, `llm_server` listens on **127.0.0.1 only** unless you pass `--host`, and it refuses requests with `"agent": true` (HTTP 403) unless you start it with `--allow-agent`. It has **no authentication**. Agent mode runs a tool loop that can execute shell commands and write files in the server's working directory, so anyone who can reach the port can then run commands as you. Do not combine `--host` with `--allow-agent` on a network you do not control. The CLI (outside LAN worker mode) and the C ABI open no listening socket. Prefer them whenever you can.

Start the server and call it from the same machine:

```bash
./build/llm_server "$MODEL" --port 8080 --ctx 2048

# in a second terminal, on the same machine
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Name three primary colours."}],"max_tokens":64,"temperature":0}'

curl -s http://127.0.0.1:8080/api/model      # model, layer count, tokenizer, backend
```

- The server accepts only `--port`, `--host`, `--allow-agent`, `--residency`, `--mmap`, `--threads` and `--ctx`. It ignores any other flag without an error, so `--ram-budget` and `--fast` have no effect and it always runs the default streaming configuration.
- `--host 0.0.0.0` (or a LAN address) makes the server reachable from other machines, and it prints a warning when you do. Put your own access control in front of it.
- It runs one generation at a time. `"stream": true` is accepted but ignored, so the response always arrives in one piece.
- `temperature` at or below 0 means greedy decoding. `max_tokens` defaults to 256.

<!-- verify: ./build/llm_server "$MODEL" --port 8080 --ctx 2048 ; run both curl commands ; lsof -nP -iTCP:8080 -sTCP:LISTEN shows 127.0.0.1:8080 ; a request with "agent":true returns 403 -->

## 5. Embed the engine through the C ABI

The C ABI is the one surface meant for other languages: opaque handles, plain structs and a C token callback. Save this as `embed_demo.c`:

```c
#include "sipllm_ffi.h"
#include <stdio.h>

static int32_t on_token(const char* piece, int64_t id, void* user) {
  (void)id; (void)user;
  fputs(piece, stdout);
  fflush(stdout);
  return 1; /* return 0 to stop early */
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; }
  sipllm_set_log_level(1);                    /* errors only */

  sipllm_params p;
  sipllm_params_default(&p);                  /* always start from the defaults */
  p.ram_budget_bytes = 256ull * 1024 * 1024;  /* hard peak-RSS ceiling */
  p.max_ctx = 512;

  char err[512] = {0};
  sipllm_ctx* ctx = sipllm_open(argv[1], &p, err, (int32_t)sizeof err);
  if (!ctx) { fprintf(stderr, "open failed: %s\n", err); return 1; }

  sipllm_sampler s;
  sipllm_sampler_default(&s);
  s.temperature = 0.0f;                       /* greedy */

  sipllm_stats st;
  int32_t n = sipllm_generate(ctx, "The capital of France is", 32, &s,
                              on_token, NULL, &st, err, (int32_t)sizeof err);
  if (n < 0) { fprintf(stderr, "\ngenerate failed: %s\n", err); sipllm_close(ctx); return 1; }

  printf("\n%d tokens, %.1f tok/s decode, %d/%d layers pinned\n",
         n, st.decode_tok_s, st.pinned_layers, st.n_layers);
  sipllm_close(ctx);
  return 0;
}
```

Build the shared library, then compile and run against it:

```bash
FFI=bindings/flutter/sipllm_flutter/ffi
LIB="$PWD/bindings/flutter/sipllm_flutter/build/desktop"
"$FFI/build_desktop.sh"                       # -> $LIB/libsipllm_ffi.dylib (macOS) or .so (Linux)
cc -I"$FFI" embed_demo.c -L"$LIB" -lsipllm_ffi -Wl,-rpath,"$LIB" -o embed_demo
./embed_demo "$MODEL"
```

- `build_desktop.sh` compiles every source file with the same flags (`CXX`, default `clang++`, and `ARCHFLAGS`, default empty). On macOS arm64 the compiler's default target already includes the SDOT instruction. On Linux aarch64 the default build leaves out the int8 `--fast` kernels and prints a compile-time message saying so. If the host supports SDOT, build with `ARCHFLAGS="-march=armv8.2-a+dotprod"`. Use `CXX=g++` if `clang++` is not installed.
- Call `sipllm_params_default()` before you change any field. A zero-initialised `sipllm_params` is **not** the same: it selects the static scheduler, zero stream buffers and no async prefetch.
- One `sipllm_ctx` serves one thread at a time. `sipllm_cancel()` is the one call that is safe from another thread; it stops generation at the next token boundary.
- `sipllm_embed()` clears the KV cache. Open a second context for embeddings if a conversation is live.
- `sipllm_params` grew a field (`spec_draft`) in this release. Rebuild any caller compiled against an older header.
- `smoke_test.c` in the same `ffi/` directory is a longer example that also covers cancellation and embeddings.

<!-- verify: bindings/flutter/sipllm_flutter/ffi/build_desktop.sh && cc -Ibindings/flutter/sipllm_flutter/ffi embed_demo.c -L"$PWD/bindings/flutter/sipllm_flutter/build/desktop" -lsipllm_ffi -Wl,-rpath,"$PWD/bindings/flutter/sipllm_flutter/build/desktop" -o embed_demo && ./embed_demo "$MODEL" -->

## 6. Use it from Flutter / Dart

> [!WARNING] Host-verified only
> The Dart, isolate and FFI path passes its tests on macOS (Apple M3). The Android APK and on-device inference through the plugin have **not** been run on a phone or watch yet.

Add the plugin as a path dependency. The plugin's native build compiles the engine sources from the repository root, so point the path inside a SipLLM checkout:

```yaml
dependencies:
  sipllm_flutter:
    path: /path/to/sipllm/bindings/flutter/sipllm_flutter
```

```dart
import 'package:sipllm_flutter/sipllm_flutter.dart';

Future<void> runOnce(String modelPath) async {
  final rt = await SipllmRuntime.open(
    modelPath,
    params: const SipllmParams(fastQuant: true).copyWithBudgetMiB(768),
  );
  try {
    final out = StringBuffer();
    await for (final tok in rt.generate('The capital of France is',
        maxNew: 32, sampler: const SipllmSampler.greedy())) {
      out.write(tok.piece);              // update your UI here
    }
    print(out);
    final st = rt.lastStats;
    if (st != null) {
      print('${st.decodeTokensPerSecond.toStringAsFixed(1)} tok/s, '
          '${st.pinnedLayers}/${st.nLayers} layers pinned, '
          'peak ${st.peakRssMiB.toStringAsFixed(0)} MiB');
    }
  } finally {
    await rt.close();
  }
}
```

- Inference runs on a worker isolate. Call `rt.cancel()` from the UI to stop at the next token. `generate()` errors if a generation is already in flight.
- `generate()` sends raw text. For a chat model, format the conversation with the model's chat template yourself.
- `rt.embed(text)` returns an L2-normalised `Float32List` for the `EmbeddingStore`, and clears the KV cache, so use a separate runtime for it.
- On Android, the plugin's Gradle build compiles `libsipllm_ffi.so` per ABI through `src/CMakeLists.txt`. On desktop, run `ffi/build_desktop.sh` first, or set `SIPLLM_FFI_LIBRARY`, or pass `libraryPath:` to `SipllmRuntime.open`.
- For downloads, `DownloadManager().enqueue(url: ..., destPath: ..., sha256: ...)` resumes across restarts.

<!-- verify: cd bindings/flutter/sipllm_flutter && ffi/build_desktop.sh && SIPLLM_TEST_MODEL="$MODEL" flutter test test/runtime_ffi_test.dart -->

## 7. Pool two machines on a LAN

Split one model's layers between a coordinator and a worker on the same network. Each machine keeps its own RAM budget.

**On both machines:** put the same GGUF file in place, and find its layer count:

```bash
./build/inspect_gguf "$MODEL" | grep block_count     # TinyLlama-1.1B has 22
```

**Create one shared secret** of at least 16 characters, and set the identical value on both machines. The engine reads it only from the environment (or from a file named by `SIPLLM_LAN_TOKEN_FILE`), never from the command line:

```bash
openssl rand -hex 16          # run once; copy the output to both machines
export SIPLLM_LAN_TOKEN='paste-the-same-value-here'
```

**Worker** (machine B, LAN address 192.168.1.20 in this example) serves layers 11 to 21:

```bash
./build/llm "$MODEL" --worker --layers 11:22 --listen 192.168.1.20:7070 \
  --ctx 512 --ram-budget 400M
```

**Coordinator** (machine A) runs layers 0 to 10 plus the embeddings and output head:

```bash
./build/llm "$MODEL" -p "The three most important inventions in history are" \
  -n 64 --greedy --ctx 512 --ram-budget 400M --nodes 192.168.1.20:7070:11-22
```

- Layer ranges are end-exclusive. The coordinator owns `[0, A)`; worker ranges must be contiguous and end at the layer count. Several workers: `--nodes h1:p1:11-17,h2:p2:17-22`.
- Every node must use the same model file, `--ctx`, `--fast` setting, `--kv-q8` setting and `--residency`. The worker checks a fingerprint and refuses a mismatch. `--ram-budget` may differ per node.
- `--listen` takes an explicit IPv4 address. Use the machine's LAN address, not `0.0.0.0`.
- Traffic is authenticated but **not encrypted**. Activations can reveal prompt content to anyone who can read the network. Use only on a network you trust.
- The coordinator's stats block gains a `lan worker:` line: calls, MB sent and received, KB per token, and time spent waiting on the worker. In the measured Wi-Fi run, waiting dominated, and decode fell to 2.6–2.8 tok/s from 15.2–15.3 on one machine.

<!-- verify: on two hosts, run the worker then the coordinator commands above with the same SIPLLM_LAN_TOKEN; compare output to a single-machine run with --ram-budget 400M -->

## 8. Benchmark your own device

Measure one configuration with the OS's peak-RSS figure, which is the authoritative number on this site:

```bash
# macOS uses /usr/bin/time -l; Linux needs GNU time (/usr/bin/time -v)
if [ "$(uname -s)" = Darwin ]; then T=-l; else T=-v; fi
/usr/bin/time $T ./build/llm "$MODEL" -p "Once upon a time" -n 48 --greedy \
  --ctx 512 --threads 4 --no-autotune --schedule fixed16 --ram-budget 0 2> run.log

grep -i "maximum resident set size" run.log     # bytes on macOS, kbytes on Linux
grep -E "pinned layers|decode:|prefill:|TTFT:|streamed:" run.log
./build/llm "$MODEL" --profile-info              # hardware id and details, for your report
```

Illustrative shape of the engine's stats block (values elided):

```text
── sipllm ────────────────
peak rss:        … MB
pinned layers:   … / 22
fast kernel:     off
decode:          … tok/s
prefill:         … tok/s
TTFT:            … s
...
streamed:        … MB (from disk)
prefetch:        … hits / … misses
```

To sweep the whole RAM-versus-speed dial, use the harness. It picks up only these two cache file names:

```bash
./sipllm pull tinyllama:q4_k_m        # -> ~/.sipllm/models/tinyllama-q4_k_m.gguf
./sipllm pull smollm2:135m            # -> ~/.sipllm/models/smollm2-135m.gguf
BUDGETS="0 256M 512M 768M" N=3 ./scripts/bench_ram_budget.sh
```

- The sweep writes `bench/results/ram-budget-<host>-<date>.json` and prints a table. It builds first unless `SKIP_BUILD=1`, and it refuses to run while the machine's load is above the thread count.
- `--no-autotune` and an explicit `--schedule` keep runs comparable; otherwise the auto-tuner may pick a different thread policy.
- Numbers are warm-cache unless you purge the page cache. When you report results, state the model, quantisation, RAM budget, hardware, compiler and commit, as every result on this site does.

<!-- verify: on macOS and on Linux: /usr/bin/time $T ./build/llm "$MODEL" -p "Once upon a time" -n 48 --greedy --ctx 512 --threads 4 --no-autotune --schedule fixed16 --ram-budget 0 ; and BUDGETS="0 256M" N=1 ./scripts/bench_ram_budget.sh -->

## Flag reference for these recipes

| Flag | What it does | Where |
|:-----|:-------------|:------|
| `-p`, `-n`, `--greedy`, `--seed`, `-t` | prompt, tokens to generate, greedy decoding, RNG seed, temperature | `main.cpp` |
| `--ctx N` | context length; the planner prices KV at this size | `main.cpp` |
| `--ram-budget SIZE` | hard peak-RSS ceiling; `0` streams everything | `main.cpp` |
| `--ram-budget-force` | run even when the plan is infeasible, with safe minimum settings | `main.cpp` |
| `--fast` | int8 kernels for Q8_0, Q4_K, Q6_K | `main.cpp` |
| `--stream-lm-head`, `--no-async` | stream the output head; disable the async prefetch thread | `main.cpp` |
| `--threads N`, `--schedule P`, `--no-autotune` | thread count, scheduler policy, skip the auto-tuner | `main.cpp` |
| `--chat` or `-i` | interactive REPL with a chat template | `main.cpp` |
| `--reuse`, `--save-session F`, `--load-session F` | reuse a shared prefix; persist the session across runs | `main.cpp` |
| `--spec N` | prompt-lookup speculative decoding, N draft tokens | `main.cpp` |
| `--semantic-cache-budget SIZE` | keep prompt-prefix KV in RAM and reuse it for later prompts that share the prefix (FP32 KV only) | `main.cpp` |
| `--worker`, `--layers A:B`, `--listen IP:PORT`, `--nodes host:port:A-B` | LAN pooling | `main.cpp` |
| `--profile-info` | print hardware details and exit | `main.cpp` |
| `--port`, `--host`, `--allow-agent`, `--ctx`, `--threads`, `--residency`, `--mmap` | the only flags `llm_server` reads | `server/server.cpp` |
