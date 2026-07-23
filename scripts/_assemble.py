#!/usr/bin/env python3
# _assemble.py — parse raw benchmark captures into the machine-readable JSON
# consumed by bench_report.py. Invoked by bench.sh; not run directly.
#
# Reads a TSV manifest (one row per raw capture file) from $MANIFEST, parses
# each capture according to its runtime, groups repeated runs, computes the
# median of every metric, and writes the final JSON to $OUT.
#
# Stdlib only. macOS `/usr/bin/time -l` reports "maximum resident set size" in
# BYTES; that is the authoritative, cross-runtime peak-RSS number.
import json
import os
import re
import statistics
from collections import defaultdict

MANIFEST = os.environ["MANIFEST"]
OUT = os.environ["OUT"]

# ---- regexes ---------------------------------------------------------------
RE_MAXRSS = re.compile(r"(\d+)\s+maximum resident set size")
# SipLLM `llm` stats block (stderr):
SIP = {
    "load_ms":               (re.compile(r"load:\s+([\d.]+)\s*s"),        1000.0),
    "ttft_ms":               (re.compile(r"TTFT:\s+([\d.]+)\s*s"),        1000.0),
    "prefill_tok_s":         (re.compile(r"prefill:\s+([\d.]+)\s*tok/s"), 1.0),
    "decode_tok_s":          (re.compile(r"decode:\s+([\d.]+)\s*tok/s"),  1.0),
    "weights_resident_mb":   (re.compile(r"weights resident:\s*([\d.]+)\s*MB"), 1.0),
    "kv_mb":                 (re.compile(r"kv cache:\s+([\d.]+)\s*MB"),   1.0),
    "streamed_mb":           (re.compile(r"streamed:\s+([\d.]+)\s*MB"),   1.0),
}
RE_PTOK = re.compile(r"prompt tokens:\s+(\d+)")
RE_GTOK = re.compile(r"generated:\s+(\d+)")
RE_PREFETCH = re.compile(r"prefetch:\s+(\d+)\s+hits\s*/\s*(\d+)\s+misses")
# llama-cli compact summary: "[ Prompt: 3286.9 t/s | Generation: 407.8 t/s ]"
RE_LC = re.compile(r"Prompt:\s+([\d.]+)\s*t/s\s*\|\s*Generation:\s+([\d.]+)\s*t/s")
# llama-bench markdown rows: "| ... | pp5 | 1672.72 ± 26.94 |"
RE_LB = re.compile(r"\|\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+)\s*(?:±\s*[\d.]+)?\s*\|")


def maxrss_mb(text):
    m = RE_MAXRSS.search(text)
    return round(int(m.group(1)) / 1e6, 1) if m else None


def parse_sipllm(text):
    d = {}
    for key, (rx, scale) in SIP.items():
        m = rx.search(text)
        d[key] = round(float(m.group(1)) * scale, 3) if m else None
    m = RE_PTOK.search(text); d["prompt_tokens"] = int(m.group(1)) if m else None
    m = RE_GTOK.search(text); d["gen_tokens"] = int(m.group(1)) if m else None
    m = RE_PREFETCH.search(text)
    d["prefetch_hits"] = int(m.group(1)) if m else None
    d["prefetch_misses"] = int(m.group(2)) if m else None
    d["peak_rss_mb"] = maxrss_mb(text)
    # streamed MB per token processed (prefill+decode) — exposes the P× re-stream
    # defect: a value ≈ the on-disk model size means the whole model is streamed
    # from storage for EVERY token.
    st, pt, gt = d.get("streamed_mb"), d.get("prompt_tokens"), d.get("gen_tokens")
    if st and pt is not None and gt is not None and (pt + gt) > 0:
        d["streamed_mb_per_token"] = round(st / (pt + gt), 1)
    else:
        d["streamed_mb_per_token"] = None
    return d


def parse_llama_cli(text):
    d = {"peak_rss_mb": maxrss_mb(text)}
    m = RE_LC.search(text)
    if m:
        d["prefill_tok_s"] = round(float(m.group(1)), 2)
        d["decode_tok_s"] = round(float(m.group(2)), 2)
    return d


def parse_llama_bench(text):
    d = {"peak_rss_mb": maxrss_mb(text)}
    for name, tps in RE_LB.findall(text):
        if name.startswith("pp"):
            d["prefill_tok_s"] = float(tps)
            d["prefill_prompt_tokens"] = int(name[2:])
        elif name.startswith("tg"):
            d["decode_tok_s"] = float(tps)
    # TTFT for a P-token prompt = P / prefill_tok_s (pure prefill compute; model
    # is already resident). Comparable to SipLLM's TTFT on the same 5-token prompt.
    if d.get("prefill_tok_s") and d.get("prefill_prompt_tokens"):
        d["ttft_ms"] = round(d["prefill_prompt_tokens"] / d["prefill_tok_s"] * 1000.0, 2)
    return d


PARSERS = {
    "sipllm": parse_sipllm,
    "llama_cli": parse_llama_cli,
    "llama_bench": parse_llama_bench,
}

# ---- read manifest, parse every capture ------------------------------------
groups = defaultdict(list)  # key -> list of per-run metric dicts
meta_by_key = {}
with open(MANIFEST) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        section, model, quant, backend, layers, runtime, ngen, threads, raw = line.split("\t")
        try:
            with open(raw, encoding="utf-8", errors="replace") as rf:
                text = rf.read()
        except FileNotFoundError:
            continue
        parsed = PARSERS[runtime](text)
        key = (section, runtime, os.path.basename(model), backend, int(layers), int(threads))
        groups[key].append(parsed)
        base = os.path.basename(model)
        cm = re.search(r"_c(\d+)", base)
        meta_by_key[key] = {
            "section": section, "runtime": runtime,
            "model": base, "model_path": model,
            "quant": quant, "backend": backend, "layers": int(layers),
            "ctx": int(cm.group(1)) if cm else None,
            "threads": int(threads), "gen_tokens_requested": int(ngen),
        }


def median_of(dicts, field):
    vals = [d[field] for d in dicts if d.get(field) is not None]
    if not vals:
        return None
    med = statistics.median(vals)
    return round(med, 3) if isinstance(med, float) else med


def collapse(key):
    runs = groups[key]
    meta = dict(meta_by_key[key])
    fields = set()
    for r in runs:
        fields.update(r.keys())
    out = dict(meta)
    out["n_runs"] = len(runs)
    for fld in sorted(fields):
        out[fld] = median_of(runs, fld)
        samples = [r.get(fld) for r in runs if r.get(fld) is not None]
        if len(samples) > 1 and isinstance(samples[0], (int, float)):
            out.setdefault("_samples", {})[fld] = samples
    return out


sipllm_real, toy, llama, sweep = [], [], [], []
for key in groups:
    row = collapse(key)
    if row["runtime"] == "sipllm" and row["section"] == "real":
        sipllm_real.append(row)
    elif row["runtime"] == "sipllm" and row["section"] == "toy":
        toy.append(row)
    elif row["runtime"] == "sipllm" and row["section"] == "sweep":
        sweep.append(row)
    else:
        llama.append(row)

toy.sort(key=lambda r: (r.get("ctx") or 0, r["layers"]))
sipllm_real.sort(key=lambda r: r["model"])
sweep.sort(key=lambda r: r["threads"])
llama.sort(key=lambda r: (r["model"], r["runtime"]))

# on-disk sizes for the RSS-vs-size story
def disk_mb(path):
    try:
        return round(os.path.getsize(path) / 1e6, 1)
    except OSError:
        return None

for row in sipllm_real + toy + llama + sweep:
    row["disk_mb"] = disk_mb(row["model_path"])

doc = {
    "meta": {
        "host": os.environ.get("HOST"),
        "date": os.environ.get("DATE"),
        "os": os.environ.get("OS_STR"),
        "cpu": os.environ.get("CPU_STR"),
        "cores": os.environ.get("CORES"),
        "threads": int(os.environ.get("THREADS", "0")),
        "gen_tokens": int(os.environ.get("NGEN", "0")),
        "prompt": os.environ.get("PROMPT"),
        "median_of_n": int(os.environ.get("NREP", "0")),
        "cache_state": os.environ.get("CACHE_STATE"),
        "cache_note": os.environ.get("CACHE_NOTE"),
        "power_note": os.environ.get("POWER_NOTE"),
        "llama_cpp_note": os.environ.get("LLAMA_NOTE"),
        "git_commit": os.environ.get("GIT_COMMIT"),
        "models_dir": os.environ.get("MODELS_DIR"),
        "peak_rss_source": "/usr/bin/time -l 'maximum resident set size' (bytes) — authoritative, cross-runtime",
        "reproduce": "./scripts/bench.sh   (knobs: N NGEN PROMPT THREADS LLAMA_BIN PURGE; see script header)",
    },
    "sipllm": sipllm_real,
    "toy_scaling": toy,
    "llama_cpp": llama,
    "thread_sweep": sweep,
}

with open(OUT, "w") as f:
    json.dump(doc, f, indent=2)
    f.write("\n")
print(f"wrote {OUT}: {len(sipllm_real)} sipllm, {len(toy)} toy, {len(llama)} llama rows")
