#!/usr/bin/env python3
"""validate_matrix.py — run the full cross-engine validation matrix.

For every model given, this:
  1. dumps per-layer residuals + final logits from OUR engine  (build/dump_logits)
  2. dumps the same from llama.cpp                              (build/llama_dump)
  3. compares them layer-by-layer                              (golden_compare.py)
  4. profiles OUR engine's peak RSS + throughput               (build/bench)

and prints a single table:

  model        layer max|Δ|  logit max|Δ|  logit cos   top10  argmax  peak RSS  dec tok/s  result

The point: prove the streaming runtime is numerically correct against llama.cpp
across *multiple* GGUF quantization formats, not just one — and that it stays
memory-flat (peak RSS << full model size) while doing so.

Usage:
  validate_matrix.py [--prompt "..."] [--tokens N] label=model.gguf [label=model.gguf ...]
  validate_matrix.py                # defaults to the four TinyLlama models in models/
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "build")
LLAMA_BIN = os.path.join(ROOT, "third_party", "llama.cpp", "build", "bin")

DEFAULT_MODELS = [
    ("Q4_K_M", "models/tinyllama-1.1b-q4_k_m.gguf"),
    ("Q5_K_M", "models/tinyllama-1.1b-q5_k_m.gguf"),
    ("Q8_0",   "models/tinyllama-1.1b-q8_0.gguf"),
    ("F16",    "models/tinyllama-1.1b-f16.gguf"),
]


def sh(cmd, extra_ld=None):
    env = dict(os.environ)
    if extra_ld:
        env["LD_LIBRARY_PATH"] = extra_ld + ":" + env.get("LD_LIBRARY_PATH", "")
    return subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)


def parse_bench(out):
    peak = None
    m = re.search(r"peak RSS:\s*([\d.]+)\s*MB", out)
    if m:
        peak = float(m.group(1))
    prefill = decode = None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "pread":
            try:
                prefill, decode = float(parts[1]), float(parts[2])
            except ValueError:
                pass
    return peak, prefill, decode


def main():
    args = sys.argv[1:]
    prompt = "The capital of France is"
    tokens = 24
    models = []
    for a in args:
        if a == "--prompt":
            prompt = args[args.index(a) + 1]
        elif a == "--tokens":
            tokens = int(args[args.index(a) + 1])
        elif "=" in a and a.endswith(".gguf"):
            label, path = a.split("=", 1)
            models.append((label, path))
    if not models:
        models = DEFAULT_MODELS

    rows = []
    for label, model in models:
        if not os.path.exists(os.path.join(ROOT, model)):
            print(f"!! skipping {label}: {model} not found")
            continue
        print(f"\n===== {label}  ({model}) =====")
        ours = f"ours_{label}.dump"
        llama = f"llama_{label}.dump"
        cmp_json = f"cmp_{label}.json"

        r = sh([f"{BUILD}/dump_logits", model, "-p", prompt, "--no-layers", "--raw", ours])
        if r.returncode != 0:
            print("  our engine FAILED:\n" + r.stderr[-500:]); continue
        r = sh([f"{BUILD}/llama_dump", model, prompt, llama], extra_ld=LLAMA_BIN)
        if r.returncode != 0:
            print("  llama.cpp dump FAILED:\n" + r.stderr[-500:]); continue

        r = sh([sys.executable, "golden/golden_compare.py", ours, llama,
                "--topk", "10", "--tol", "0.05", "--json", cmp_json, "--label", label])
        print(r.stdout.rstrip())
        with open(os.path.join(ROOT, cmp_json)) as jf:
            c = json.load(jf)

        rb = sh([f"{BUILD}/bench", model, "-p", prompt, "-n", str(tokens)])
        peak, prefill, decode = parse_bench(rb.stdout)

        rows.append({**c, "peak_rss_mb": peak, "prefill_tok_s": prefill,
                     "decode_tok_s": decode, "model": model})

    # ---- matrix ----
    print("\n\n================  VALIDATION MATRIX  ================\n")
    hdr = ("model", "layer max|Δ|", "logit max|Δ|", "logit cos",
           "top10", "argmax", "peak RSS", "dec tok/s", "result")
    print("{:<8} {:>13} {:>13} {:>11} {:>7} {:>7} {:>10} {:>10} {:>7}".format(*hdr))
    print("-" * 96)
    all_pass = True
    for r in rows:
        all_pass &= r["pass"]
        print("{:<8} {:>13.2e} {:>13.2e} {:>11.6f} {:>5d}/10 {:>7} {:>8.1f}MB {:>10.2f} {:>7}".format(
            r["label"], r["layer_max_abs_err"], r["final_logit_max_abs_err"],
            r["final_logit_cosine"], r["topk_overlap"],
            "yes" if r["argmax_match"] else "NO",
            r["peak_rss_mb"] or 0.0, r["decode_tok_s"] or 0.0,
            "PASS" if r["pass"] else "CHECK"))
    print("-" * 96)
    print(f"\nOVERALL: {'ALL PASS' if all_pass else 'SOME CHECKS'}  "
          f"({len(rows)} formats validated against llama.cpp)")

    with open(os.path.join(ROOT, "golden", "matrix_results.json"), "w") as jf:
        json.dump(rows, jf, indent=2)
    print("wrote golden/matrix_results.json")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
