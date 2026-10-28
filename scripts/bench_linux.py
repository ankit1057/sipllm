#!/usr/bin/env python3
"""bench_linux.py — Linux counterpart of bench.sh's measurement core.

bench.sh reads peak RSS from macOS `/usr/bin/time -l`. On Linux this script
runs one configuration median-of-N under GNU `/usr/bin/time -v` (peak RSS from
the kernel, comparable across runtimes), pinned with taskset, and appends one
row to a JSON file shaped like bench/results/*.json ({"meta", "sipllm",
"llama_cpp"}). It also records what a RAM number alone would hide:

  * vram_peak_mb  — peak GPU memory in use during the run minus the idle
                    baseline (nvidia-smi), so a --gpu-layers row cannot look
                    lighter than it is;
  * cooldown      — before every run it waits until the CPU package is below
                    --max-start-temp, so later rows are not throttled rows.

Usage (one configuration per call; rows accumulate in --out):
  scripts/bench_linux.py --out R.json --label cpu-default --reps 3 \\
      --cpus 0,2,4,6,8,10 -- build/llm model.gguf -p "..." -n 32 --greedy --threads 6
  scripts/bench_linux.py --out R.json --label llama --kind llama-bench --reps 1 \\
      --cpus 0,2,4,6,8,10 -- llama-bench -m model.gguf -t 6 -p 200 -n 32 -r 3 -ngl 0 -o json

Only the Python standard library is used.
"""
import argparse, json, os, platform, re, shutil, statistics, subprocess, sys, threading, time


def read_coretemp_c():
    """Package temperature (deg C) from the coretemp/k10temp hwmon, else None."""
    base = "/sys/class/hwmon"
    try:
        for h in sorted(os.listdir(base)):
            d = os.path.join(base, h)
            try:
                name = open(os.path.join(d, "name")).read().strip()
            except OSError:
                continue
            if name in ("coretemp", "k10temp", "zenpower"):
                return int(open(os.path.join(d, "temp1_input")).read()) / 1000.0
    except OSError:
        pass
    return None


def cool_down(max_start, timeout_s=600):
    t0 = time.time()
    while True:
        t = read_coretemp_c()
        if t is None or t <= max_start or time.time() - t0 > timeout_s:
            return t
        time.sleep(5)


def gpu_used_mb():
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return sum(float(x) for x in out.split())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class GpuSampler(threading.Thread):
    def __init__(self, period=0.25):
        super().__init__(daemon=True)
        self.period, self.peak, self.stop_flag = period, None, False

    def run(self):
        while not self.stop_flag:
            v = gpu_used_mb()
            if v is not None:
                self.peak = v if self.peak is None else max(self.peak, v)
            time.sleep(self.period)


def num(pattern, text, cast=float):
    m = re.search(pattern, text, re.M)
    return cast(m.group(1)) if m else None


def wall_seconds(time_v):
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", time_v)
    if not m:
        return None
    sec = 0.0
    for part in m.group(1).split(":"):
        sec = sec * 60 + float(part)
    return round(sec, 2)


def parse_sipllm(text):
    return {
        "decode_tok_s": num(r"^decode:\s+([\d.]+) tok/s", text),
        "prefill_tok_s": num(r"^prefill:\s+([\d.]+) tok/s", text),
        "ttft_ms": (lambda s: s * 1000 if s is not None else None)(num(r"^TTFT:\s+([\d.]+) s", text)),
        "prompt_tokens": num(r"^prompt tokens:\s+(\d+)", text, int),
        "gen_tokens": num(r"^generated:\s+(\d+)", text, int),
        "kv_mb": num(r"^kv cache:\s+([\d.]+) MB", text),
        "gpu_layers": num(r"^gpu layers:\s+(\d+) /", text, int),
        "gpu_weights_mb": num(r"^gpu layers:.*\((\d+) MB VRAM", text, int),
        "internal_peak_rss_mb": num(r"^peak rss:\s+(\d+) MB", text, int),
    }


def parse_llama_bench(text):
    """llama-bench -o json: one object per test; pp = prefill, tg = decode."""
    i = text.find("[")
    rows = json.loads(text[i:text.rfind("]") + 1]) if i >= 0 else []
    r = {}
    for t in rows:
        if t.get("n_prompt", 0) > 0 and t.get("n_gen", 0) == 0:
            r["prefill_tok_s"], r["prompt_tokens"] = t["avg_ts"], t["n_prompt"]
            r["_prefill_samples"] = t.get("samples_ts")
        elif t.get("n_gen", 0) > 0 and t.get("n_prompt", 0) == 0:
            r["decode_tok_s"], r["gen_tokens"] = t["avg_ts"], t["n_gen"]
            r["_decode_samples"] = t.get("samples_ts")
        r.setdefault("build", t.get("build_commit"))
    return r


def meta(args):
    cpu = "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    gpu = None
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or None
    ac = None
    for p in ("/sys/class/power_supply/AC/online", "/sys/class/power_supply/ADP1/online"):
        if os.path.exists(p):
            ac = open(p).read().strip() == "1"
    mem_gb = None
    try:
        mem_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
    except (ValueError, OSError):
        pass
    return {"host": args.host, "date": time.strftime("%Y-%m-%d"), "os": f"{platform.system()} {platform.release()}",
            "cpu": cpu, "logical_cpus": os.cpu_count(), "ram_gb": mem_gb, "gpu": gpu, "on_ac_power": ac,
            "peak_rss_source": "GNU /usr/bin/time -v (maximum resident set size)",
            "vram_source": "nvidia-smi memory.used, peak during run minus idle baseline",
            "cache_state": "warm (model file read at least once before timing)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--kind", choices=["sipllm", "llama-bench"], default="sipllm")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cpus", default=None, help="taskset CPU list, e.g. 0,2,4,6,8,10")
    ap.add_argument("--host", default="x86-linux", help="label stored in meta (not the hostname)")
    ap.add_argument("--max-start-temp", type=float, default=70.0)
    ap.add_argument("--note", default=None)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("missing command after --")

    full = ["/usr/bin/time", "-v"] + (["taskset", "-c", a.cpus] if a.cpus else []) + cmd
    samples = []
    for rep in range(a.reps):
        start_temp = cool_down(a.max_start_temp)
        base_gpu = gpu_used_mb()
        sampler = GpuSampler()
        sampler.start()
        p = subprocess.run(full, capture_output=True, text=True)
        sampler.stop_flag = True
        sampler.join()
        text = p.stdout + p.stderr
        if p.returncode != 0:
            sys.stderr.write(text[-3000:])
            sys.exit(f"run {rep} failed with exit {p.returncode}")
        s = parse_sipllm(text) if a.kind == "sipllm" else parse_llama_bench(p.stdout)
        rss_kb = num(r"Maximum resident set size \(kbytes\):\s+(\d+)", p.stderr, int)
        s["peak_rss_mb"] = round(rss_kb / 1024, 1) if rss_kb else None
        s["vram_peak_mb"] = (round(sampler.peak - base_gpu, 1)
                             if sampler.peak is not None and base_gpu is not None else None)
        s["wall_s"] = wall_seconds(p.stderr)
        s["start_temp_c"] = start_temp
        samples.append(s)
        print(f"[{a.label}] rep {rep}: decode={s.get('decode_tok_s')} prefill={s.get('prefill_tok_s')} "
              f"rss={s['peak_rss_mb']}MB vram={s['vram_peak_mb']}MB start={start_temp}C", flush=True)

    def med(k):
        v = [x[k] for x in samples if x.get(k) is not None]
        return round(statistics.median(v), 2) if v else None

    row = {"label": a.label, "runtime": "sipllm" if a.kind == "sipllm" else "llama_bench",
           "cmd": " ".join(cmd).replace(os.path.expanduser("~"), "~"), "cpus": a.cpus, "n_runs": a.reps, "note": a.note}
    for k in ("decode_tok_s", "prefill_tok_s", "ttft_ms", "peak_rss_mb", "vram_peak_mb", "kv_mb", "wall_s"):
        row[k] = med(k)
    for k in ("prompt_tokens", "gen_tokens", "gpu_layers", "gpu_weights_mb", "build"):
        row[k] = samples[-1].get(k)
    row["_samples"] = {k: [x.get(k) for x in samples]
                       for k in ("decode_tok_s", "prefill_tok_s", "peak_rss_mb", "vram_peak_mb", "start_temp_c")}
    if a.kind == "llama-bench":
        row["_samples"]["llama_bench_prefill_ts"] = samples[-1].get("_prefill_samples")
        row["_samples"]["llama_bench_decode_ts"] = samples[-1].get("_decode_samples")

    doc = {"meta": meta(a), "sipllm": [], "llama_cpp": []}
    if os.path.exists(a.out):
        doc = json.load(open(a.out))
    section = "sipllm" if a.kind == "sipllm" else "llama_cpp"
    doc[section] = [r for r in doc.get(section, []) if r["label"] != a.label] + [row]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
