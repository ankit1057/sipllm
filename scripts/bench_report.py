#!/usr/bin/env python3
# bench_report.py — render the SipLLM v0.2 north-star baseline from a results
# JSON produced by scripts/bench.sh. Stdlib only.
#
#   ./scripts/bench_report.py [bench/results/<host>-<date>.json]
#
# With no argument it picks the newest bench/results/*.json. Prints:
#   1. run metadata (host / cpu / threads / cache / power / reproduce cmd)
#   2. north-star table (Peak RSS / TTFT / decode tok/s / power) for SipLLM,
#      with weights-resident, KV, and streamed-MB/token context columns
#   3. peak-RSS-vs-depth curve for the toy models (the bounded-RSS proof)
#   4. SipLLM vs llama.cpp head-to-head (peak RSS / TTFT / decode tok/s)
#   5. optional thread-sweep table
import glob
import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "bench", "results")


def load(path=None):
    if path is None:
        cands = sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json")), key=os.path.getmtime)
        if not cands:
            sys.exit(f"no results JSON found in {RESULTS_DIR}; run ./scripts/bench.sh first")
        path = cands[-1]
    with open(path) as f:
        return path, json.load(f)


def fmt(v, spec=""):
    if v is None:
        return "n/a"
    if spec:
        try:
            return format(v, spec)
        except (ValueError, TypeError):
            return str(v)
    return str(v)


def table(headers, rows, aligns=None):
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    widths = [max(len(str(c)) for c in col) for col in cols]
    aligns = aligns or [">"] * len(headers)
    aligns[0] = "<"

    def line(cells):
        return "  " + "  ".join(
            format(str(c), f"{a}{w}") for c, a, w in zip(cells, aligns, widths))

    out = [line(headers), "  " + "  ".join("-" * w for w in widths)]
    out += [line(r) for r in rows]
    return "\n".join(out)


def h(title):
    return f"\n\033[1m{title}\033[0m" if sys.stdout.isatty() else f"\n== {title} =="


def main():
    path, d = load(sys.argv[1] if len(sys.argv) > 1 else None)
    m = d["meta"]

    print(f"SipLLM benchmark report  <  {os.path.relpath(path)}")
    print(h("run metadata"))
    for k in ("host", "os", "cpu", "cores", "threads", "gen_tokens", "prompt",
              "median_of_n", "cache_state", "git_commit"):
        print(f"  {k:14s}: {m.get(k)}")
    print(f"  {'cache_note':14s}: {m.get('cache_note')}")
    print(f"  {'power':14s}: {m.get('power_note')}")
    print(f"  {'peak_rss_src':14s}: {m.get('peak_rss_source')}")
    print(f"  {'reproduce':14s}: {m.get('reproduce')}")

    # --- 1. north-star table (SipLLM) --------------------------------------
    print(h("north star — SipLLM (priority: Peak RSS v, TTFT v, decode tok/s ^, Power v)"))
    hdr = ["model", "quant", "disk MB", "PEAK RSS MB", "TTFT ms", "decode t/s",
           "power", "wt-res MB", "KV MB", "streamed MB/tok"]
    rows = []
    for r in d["sipllm"]:
        rows.append([
            r["model"], r.get("quant"), fmt(r.get("disk_mb"), ".1f"),
            fmt(r.get("peak_rss_mb"), ".1f"), fmt(r.get("ttft_ms"), ".1f"),
            fmt(r.get("decode_tok_s"), ".2f"), "N/A",
            fmt(r.get("weights_resident_mb"), ".1f"), fmt(r.get("kv_mb"), ".1f"),
            fmt(r.get("streamed_mb_per_token"), ".1f"),
        ])
    print(table(hdr, rows))
    print("  power N/A: 'sudo powermetrics' needs privileges on this host (never fabricated).")
    print("  streamed MB/tok ~= on-disk size  =>  the whole model is re-streamed for EVERY token")
    print("  (prefill re-streams the model once per prompt token: the dominant TTFT/IO cost).")

    # --- 2. peak-RSS-vs-depth curve (toy) ----------------------------------
    print(h("bounded-RSS proof — toy models, IDENTICAL geometry, increasing depth"))
    print("  Only --layers varies (dim/heads/kv/ffn/vocab fixed) => disk size grows ~linearly.")
    print("  weights-resident MUST stay flat (streaming); peak-RSS growth is the O(layers*ctx)")
    print("  dense fp32 KV cache alone, not weights.\n")
    hdr = ["ctx", "layers", "disk MB", "PEAK RSS MB", "wt-res MB", "KV MB", "decode t/s"]
    rows = []
    for r in d.get("toy_scaling", []):
        rows.append([
            fmt(r.get("ctx")), r["layers"], fmt(r.get("disk_mb"), ".1f"),
            fmt(r.get("peak_rss_mb"), ".1f"), fmt(r.get("weights_resident_mb"), ".1f"),
            fmt(r.get("kv_mb"), ".1f"), fmt(r.get("decode_tok_s"), ".1f"),
        ])
    print(table(hdr, rows))
    # ascii curve of weights-resident and peak vs layers (per ctx)
    _ascii_depth_curve(d.get("toy_scaling", []))

    # --- 3. SipLLM vs llama.cpp -------------------------------------------
    print(h("cross-runtime baseline — SipLLM vs llama.cpp (same gguf/prompt/threads, CPU-only)"))
    print(f"  {m.get('llama_cpp_note')}\n")
    _cross_runtime(d)

    # --- 4. thread sweep ---------------------------------------------------
    sweep = d.get("thread_sweep", [])
    if sweep:
        print(h("thread sweep — SipLLM smollm2 (batch=1 GEMV over-subscription)"))
        hdr = ["threads", "PEAK RSS MB", "TTFT ms", "prefill t/s", "decode t/s"]
        rows = [[r["threads"], fmt(r.get("peak_rss_mb"), ".1f"),
                 fmt(r.get("ttft_ms"), ".1f"), fmt(r.get("prefill_tok_s"), ".2f"),
                 fmt(r.get("decode_tok_s"), ".2f")] for r in sweep]
        print(table(hdr, rows))
        best = max(sweep, key=lambda r: r.get("decode_tok_s") or 0)
        print(f"  peak decode at {best['threads']} threads "
              f"({fmt(best.get('decode_tok_s'), '.1f')} tok/s); gains flatten past 4 "
              f"(M3 has 4 performance cores).")
    print()


def _ascii_depth_curve(toy):
    by_ctx = {}
    for r in toy:
        by_ctx.setdefault(r.get("ctx"), []).append(r)
    for ctx in sorted(by_ctx, key=lambda c: c or 0):
        rows = sorted(by_ctx[ctx], key=lambda r: r["layers"])
        peaks = [r.get("peak_rss_mb") or 0 for r in rows]
        wts = [r.get("weights_resident_mb") or 0 for r in rows]
        vmax = max(peaks + [1e-9])
        print(f"\n  ctx={ctx}: peak RSS (#) vs weights-resident (=), bar scaled to {vmax:.1f} MB")
        for r, pk, wt in zip(rows, peaks, wts):
            nb = int(pk / vmax * 40 + 0.5)
            nw = int(wt / vmax * 40 + 0.5)
            bar = "".join("#" if i < nb else ("=" if i < max(nw, 1) else " ")
                          for i in range(40))
            print(f"    {r['layers']:>3} layers |{bar}| peak {pk:6.1f}  wt {wt:4.1f} MB")


def _cross_runtime(d):
    # index SipLLM + llama rows by model
    sip = {r["model"]: r for r in d["sipllm"]}
    llama = d.get("llama_cpp", [])
    if not llama:
        print("  llama.cpp not present in this run — see meta.llama_cpp_note to build it.")
        return
    lc = {}
    for r in llama:
        lc.setdefault(r["model"], {})[r["runtime"]] = r
    hdr = ["model", "metric", "SipLLM", "llama.cpp", "winner", "gap"]
    rows = []
    for model in sorted(set(sip) & set(lc)):
        s = sip[model]
        cli = lc[model].get("llama_cli", {})
        bench = lc[model].get("llama_bench", {})
        # peak RSS: compare SipLLM generation vs llama-cli generation
        rows += _cmp_rows(model, s, cli, bench)
    print(table(hdr, rows, aligns=["<", "<", ">", ">", "<", ">"]))
    print("  peak RSS: SipLLM `llm` gen run vs llama-cli `-st` gen run (both /usr/bin/time -l).")
    print("  decode/prefill t/s: SipLLM stats vs llama-bench tg/pp (standardized, error-barred).")
    print("  TTFT: prefill time for the 5-token prompt (5 / prefill_t_s); excludes model load.")


def _cmp_rows(model, s, cli, bench):
    out = []

    def row(metric, sv, lv, lower_better, unit=""):
        if sv is None or lv is None:
            return [model if not out else "", metric, fmt(sv), fmt(lv), "n/a", "n/a"]
        if lower_better:
            win = "SipLLM" if sv < lv else "llama.cpp"
            ratio = (max(sv, lv) / min(sv, lv)) if min(sv, lv) else 0
        else:
            win = "SipLLM" if sv > lv else "llama.cpp"
            ratio = (max(sv, lv) / min(sv, lv)) if min(sv, lv) else 0
        return [model if not out else "", metric,
                f"{sv:.1f}{unit}", f"{lv:.1f}{unit}", win, f"{ratio:.2f}x"]

    out.append(row("peak RSS MB", s.get("peak_rss_mb"),
                   (cli or {}).get("peak_rss_mb"), True))
    out.append(row("TTFT ms", s.get("ttft_ms"), (bench or {}).get("ttft_ms"), True))
    out.append(row("decode tok/s", s.get("decode_tok_s"),
                   (bench or {}).get("decode_tok_s"), False))
    return out


if __name__ == "__main__":
    main()
