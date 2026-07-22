# Golden test — validate our engine against llama.cpp

The claim "our engine is correct" is only believable when, for the **same model
and prompt**, our per-layer hidden states and final logits match llama.cpp
within quantization tolerance, and the greedy tokens agree.

A **bit-exact hash** will *not* match across engines even when both are correct
— summation order and fp16/fp32 rounding differ. So we compare **numerically**:
per-layer `max|Δ|`, `mean|Δ|`, cosine similarity, plus final-logit top-k overlap
and argmax agreement. (We still print per-layer FNV-1a hashes for *our own*
run-to-run determinism — see `dump_logits`.)

## Steps

1. **Build our engine**
   ```
   make            # produces build/llm, build/dump_logits
   ```

2. **Build llama.cpp** (CPU backend is enough; `cmake`+`git` are present)
   ```
   git clone https://github.com/ggml-org/llama.cpp
   cmake -S llama.cpp -B llama.cpp/build -DGGML_NATIVE=ON
   cmake --build llama.cpp/build -j --target llama ggml
   ```

3. **Build the llama.cpp-side dumper** (writes our LGDN format)
   ```
   g++ -std=c++17 -O2 golden/llama_dump.cpp \
       -I llama.cpp/include -I llama.cpp/ggml/include \
       -L llama.cpp/build/src -L llama.cpp/build/ggml/src \
       -lllama -lggml -lggml-base -lpthread -o build/llama_dump
   ```

4. **Get a small real model** (fast to iterate; the 8B works identically)
   ```
   # e.g. TinyLlama 1.1B or Llama-3.2-1B, Q8_0 for a tight tolerance:
   curl -L -o model.gguf <hf-resolve-url>
   ```

5. **Dump from both engines, same prompt**
   ```
   ./build/dump_logits model.gguf -p "The capital of France is" --raw ours.dump
   ./build/llama_dump  model.gguf    "The capital of France is"       llama.dump
   ```

6. **Compare**
   ```
   python3 golden/golden_compare.py ours.dump llama.dump --topk 10 --tol 0.05
   ```

## Reading the result

```
layer   max|Δ|      mean|Δ|     cosine     rel_max  verdict
    0   0.0007      0.0002      0.999978   0.0015   ok
   ...
# final logits: cosine=0.9999   top-10 overlap: 10/10   argmax match: True
RESULT: PASS
```

* **cosine ≈ 1.0 per layer** and **argmax match** ⇒ the engines agree; residual
  differences are pure quantization/rounding.
* A layer where cosine suddenly drops localizes a bug to that block (RoPE,
  attention scaling, a transposed weight, a wrong dequant, …).

The comparator itself is validated in the repo without llama.cpp by diffing an
fp32 build of a model against its Q8_0 build (see the top-level README) — that
reproduces exactly the small per-layer deltas expected against a reference.
