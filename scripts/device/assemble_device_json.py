#!/usr/bin/env python3
"""assemble_device_json.py — aggregate on-device evidence into one committed JSON.

usage: assemble_device_json.py <evidence_dir> <out.json> [--commit SHA]
Reads device-bench*.tsv (SipLLM rows) and runs/out/*.txt (raw captures, incl.
llama.cpp via device_cmd.sh) and the baseline/ directory for device identity.
Identifiers that could single out the handset (serial, installed packages) are
never copied; model/SoC/Android version are.
"""
import glob, json, os, re, sys

def kv(path):
    d = {}
    for line in open(path, errors="replace"):
        m = re.match(r"\[([^\]]+)\]: \[(.*)\]", line.strip())
        if m: d[m.group(1)] = m.group(2)
    return d

def parse_run(path):
    r = {"file": os.path.basename(path)}
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        m = re.match(r"### exit=(\d+) wall_ms=(-?\d+) VmHWM_kB=(\d+)", line)
        if m: r.update(exit=int(m[1]), wall_ms=int(m[2]), vmhwm_mb=round(int(m[3]) / 1024, 1))
        m = re.match(r"### pre: battery_temp_dC=(\d+) level=(\d+)", line)
        if m: r.update(battery_temp_c=int(m[1]) / 10, battery_level=int(m[2]))
        m = re.match(r"### pre: cpufreq cur.*?: (\d+) (\d+)", line)
        if m: r.update(little_khz_pre=int(m[1]), big_khz_pre=int(m[2]))
        m = re.match(r"### post: cpufreq cur.*?: (\d+) (\d+)", line)
        if m: r.update(little_khz_post=int(m[1]), big_khz_post=int(m[2]))
        for key, pat in (("decode_tok_s", r"^decode:\s+([\d.]+)"), ("prefill_tok_s", r"^prefill:\s+([\d.]+)"),
                         ("ttft_s", r"^TTFT:\s+([\d.]+)"), ("peak_rss_engine_mb", r"^peak rss:\s+([\d.]+)")):
            m = re.match(pat, line)
            if m: r[key] = float(m[1])
        m = re.match(r"^pinned layers:\s+(\d+) / (\d+)", line)
        if m: r.update(pinned=int(m[1]), layers=int(m[2]))
        m = re.match(r"^spec decode:\s+(\d+) drafted, (\d+) accepted.*?(\d+) verify passes", line)
        if m: r.update(spec_drafted=int(m[1]), spec_accepted=int(m[2]), spec_passes=int(m[3]))
        # llama-bench markdown rows: | model | size | params | backend | threads | test | t/s |
        m = re.match(r"\|\s*llama.*?\|\s*(\d+)\s*\|\s*(pp\d+|tg\d+)\s*\|\s*([\d.]+) ± ([\d.]+)\s*\|", line)
        if m: r.setdefault("llama_bench", []).append({"threads": int(m[1]), "test": m[2], "tok_s": float(m[3]), "sd": float(m[4])})
        m = re.search(r"(prompt eval|eval) time =\s+([\d.]+) ms /\s+(\d+) (tokens|runs).*?([\d.]+) tokens per second", line)
        if m: r["llama_cli_" + ("prefill" if m[1] == "prompt eval" else "decode") + "_tok_s"] = float(m[5])
    return r

def main():
    ev, out = sys.argv[1], sys.argv[2]
    commit = sys.argv[sys.argv.index("--commit") + 1] if "--commit" in sys.argv else "unknown"
    props = kv(os.path.join(ev, "baseline", "01-getprop.txt")) if os.path.exists(os.path.join(ev, "baseline", "01-getprop.txt")) else {}
    cpuinfo = open(os.path.join(ev, "baseline", "02-cpuinfo.txt"), errors="replace").read() if os.path.exists(os.path.join(ev, "baseline", "02-cpuinfo.txt")) else ""
    feat = re.search(r"Features\s*:\s*(.*)", cpuinfo)
    meminfo = open(os.path.join(ev, "baseline", "05-meminfo.txt"), errors="replace").read() if os.path.exists(os.path.join(ev, "baseline", "05-meminfo.txt")) else ""
    mt = re.search(r"MemTotal:\s+(\d+)", meminfo)
    runs = [parse_run(p) for p in sorted(glob.glob(os.path.join(ev, "runs", "out", "*.txt")))]
    doc = {
        "meta": {
            "benchmark": "on_device_shell (scripts/device)",
            "device_model": props.get("ro.product.model"), "brand": props.get("ro.product.brand"),
            "soc": props.get("ro.soc.model") or props.get("ro.board.platform"),
            "android": props.get("ro.build.version.release"), "sdk": props.get("ro.build.version.sdk"),
            "cpu_features": feat.group(1).strip() if feat else None,
            "cores": cpuinfo.count("processor\t:") or cpuinfo.count("processor"),
            "mem_total_mb": round(int(mt.group(1)) / 1024) if mt else None,
            "git_commit": commit,
            "peak_rss_source": "/proc/<pid>/status VmHWM sampled 5x/s (kernel high-water RSS)",
            "method": "adb shell, nice 10, no app installed, third-party apps force-stopped; per-run battery temp + big/little scaling_cur_freq recorded",
        },
        "runs": runs,
    }
    with open(out, "w") as f: json.dump(doc, f, indent=1)
    print(f"wrote {out}: {len(runs)} runs")

if __name__ == "__main__": main()
