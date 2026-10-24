#!/usr/bin/env python3
"""bench_sip_batch.py — aggregate decode throughput of build/sip_batch vs batch
size, and whether every job's output is byte-identical to batch 1.

Jobs are code-continuation prompts: the first ~400 bytes of the first N files
in src/ (sorted), so the prompt set is reproducible from the tree itself.

  scripts/bench_sip_batch.py --model m.gguf --batches 1,4,8,12 --jobs 12 \\
      --max-new 48 --cpus 0,2,4,6,8,10 --out R.json -- --threads 6 --ram-budget 20G

Arguments after `--` go to sip_batch unchanged (e.g. --gpu-layers max).
Peak RSS comes from GNU /usr/bin/time -v. Standard library only.
"""
import argparse, json, os, re, subprocess, sys, time


def jobs_from_tree(root, n, nbytes):
    files = sorted(f for f in os.listdir(os.path.join(root, "src")) if f.endswith(".cpp"))[:n]
    out = []
    for f in files:
        t = open(os.path.join(root, "src", f), "rb").read()[:nbytes]
        cut = t.rfind(b"\n")
        out.append((f, t[:cut + 1] if cut > 0 else t))
    return out


def cool_down(max_c=70.0, timeout_s=600):
    """Wait until the CPU package (coretemp/k10temp hwmon) is at or below max_c."""
    base, t0 = "/sys/class/hwmon", time.time()
    sensor = None
    for h in sorted(os.listdir(base)):
        try:
            if open(os.path.join(base, h, "name")).read().strip() in ("coretemp", "k10temp", "zenpower"):
                sensor = os.path.join(base, h, "temp1_input")
                break
        except OSError:
            pass
    while sensor and time.time() - t0 < timeout_s:
        t = int(open(sensor).read()) / 1000.0
        if t <= max_c:
            return t
        time.sleep(5)
    return None


def run(binary, model, batch, jobs, max_new, extra, cpus):
    payload = b"".join(b"JOB %s %d %d\n" % (name.encode(), max_new, len(p)) + p for name, p in jobs) + b"END\n"
    cmd = ["/usr/bin/time", "-v"] + (["taskset", "-c", cpus] if cpus else []) + \
          [binary, model, "--batch", str(batch)] + extra
    p = subprocess.run(cmd, input=payload, capture_output=True)
    err = p.stderr.decode(errors="replace")
    if p.returncode != 0:
        sys.exit(f"sip_batch failed (batch {batch}):\n{err[-3000:]}")
    results, out, i = {}, p.stdout, 0
    while i < len(out):
        nl = out.index(b"\n", i)
        head = out[i:nl].decode().split()
        nb = int(head[-1])
        results[head[1]] = {"stop": head[2], "gen_tokens": int(head[5]), "text": out[nl + 1:nl + 1 + nb]}
        i = nl + 1 + nb + 1
    m = re.search(r"decode (\d+) tokens in ([\d.]+) s = ([\d.]+) tok/s", err)
    rss = re.search(r"Maximum resident set size \(kbytes\):\s+(\d+)", err)
    return results, {"decode_tokens": int(m.group(1)), "decode_s": float(m.group(2)),
                     "decode_tok_s": float(m.group(3)),
                     "peak_rss_mb": round(int(rss.group(1)) / 1024, 1) if rss else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default="build/sip_batch")
    ap.add_argument("--batches", default="1,4,8,12")
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=48)
    ap.add_argument("--prompt-bytes", type=int, default=400)
    ap.add_argument("--cpus", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="sip_batch")
    ap.add_argument("extra", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = a.extra[1:] if a.extra and a.extra[0] == "--" else a.extra
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    jobs = jobs_from_tree(root, a.jobs, a.prompt_bytes)

    rows, ref = [], None
    for b in [int(x) for x in a.batches.split(",")]:
        start_c = cool_down()
        res, st = run(a.binary, a.model, b, jobs, a.max_new, extra, a.cpus)
        st["start_temp_c"] = start_c
        if ref is None:
            ref = res
        same = sum(1 for k in ref if k in res and res[k]["text"] == ref[k]["text"])
        row = {"label": a.label, "batch": b, "jobs": len(jobs), "max_new": a.max_new,
               "args": " ".join(extra), "cpus": a.cpus, **st,
               "identical_to_batch1": f"{same}/{len(ref)}"}
        rows.append(row)
        print(json.dumps(row), flush=True)

    doc = {"meta": {"date": time.strftime("%Y-%m-%d"), "model": os.path.basename(a.model),
                    "prompts": f"first {a.prompt_bytes} bytes of the first {a.jobs} src/*.cpp files",
                    "decoding": "greedy"}, "rows": []}
    if os.path.exists(a.out):
        doc = json.load(open(a.out))
    doc["rows"] = [r for r in doc["rows"] if r["label"] != a.label] + rows
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
