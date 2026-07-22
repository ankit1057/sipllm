#!/usr/bin/env python3
"""golden_compare.py — compare two LGDN dumps layer-by-layer.

Cross-engine bit-exact hashes never match (rounding/summation order differ), so
this reports the metrics that actually tell you whether two engines agree:

  per layer:  max|Δ|, mean|Δ|, cosine similarity
  final:      top-k token overlap, argmax match, logits cosine

Usage:
    golden_compare.py ours.dump other.dump [--topk 10] [--tol 0.05]

Dump format (little-endian), written by tools/dump_logits --raw and by
golden/llama_dump (the llama.cpp side):
    magic "LGDN" | u32 n_layers | u32 dim | u32 vocab
    n_layers × (dim × f32)   residual stream after each block (last prompt pos)
    vocab × f32              final logits (first prediction)
"""
import struct, sys, math, argparse


def load(path):
    with open(path, "rb") as f:
        magic, nl, dim, vocab = struct.unpack("<4sIII", f.read(16))
        assert magic == b"LGDN", f"{path}: bad magic {magic!r}"
        hidden = [list(struct.unpack(f"<{dim}f", f.read(4 * dim))) for _ in range(nl)]
        logits = list(struct.unpack(f"<{vocab}f", f.read(4 * vocab)))
    return nl, dim, vocab, hidden, logits


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def stats(a, b):
    d = [abs(x - y) for x, y in zip(a, b)]
    return max(d), sum(d) / len(d), cosine(a, b)


def topk(v, k):
    return [i for i, _ in sorted(enumerate(v), key=lambda t: -t[1])[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max acceptable per-layer max|Δ| relative to layer L2")
    ap.add_argument("--json", default="",
                    help="also write a machine-readable summary to this path")
    ap.add_argument("--label", default="",
                    help="tag stored in the JSON summary (e.g. the quant name)")
    args = ap.parse_args()

    nla, dima, va, ha, la = load(args.a)
    nlb, dimb, vb, hb, lb = load(args.b)
    if (nla, dima, va) != (nlb, dimb, vb):
        print(f"shape mismatch: {(nla,dima,va)} vs {(nlb,dimb,vb)}"); sys.exit(2)

    print(f"# comparing {args.a}  vs  {args.b}")
    print(f"# layers={nla} dim={dima} vocab={va}\n")
    print(f"{'layer':>5} {'max|Δ|':>12} {'mean|Δ|':>12} {'cosine':>10} {'rel_max':>10}  verdict")
    worst_cos = 1.0
    layer_max = 0.0        # largest per-layer max|Δ| over all layers
    ok = True
    for l in range(nla):
        mx, mean, cos = stats(ha[l], hb[l])
        l2 = math.sqrt(sum(x * x for x in ha[l])) or 1.0
        rel = mx / l2
        verdict = "ok" if rel <= args.tol else "DIVERGE"
        if rel > args.tol:
            ok = False
        worst_cos = min(worst_cos, cos)
        layer_max = max(layer_max, mx)
        print(f"{l:>5} {mx:>12.6f} {mean:>12.6f} {cos:>10.6f} {rel:>10.4f}  {verdict}")

    mx, mean, cos = stats(la, lb)
    ta, tb = topk(la, args.topk), topk(lb, args.topk)
    overlap = len(set(ta) & set(tb))
    print(f"\n# final logits: max|Δ|={mx:.6f} mean|Δ|={mean:.6f} cosine={cos:.6f}")
    print(f"# top-{args.topk} overlap: {overlap}/{args.topk}   argmax match: {ta[0] == tb[0]} "
          f"(a={ta[0]} b={tb[0]})")
    print(f"# worst per-layer cosine: {worst_cos:.6f}")
    passed = ok and ta[0] == tb[0]
    print(f"\nRESULT: {'PASS' if passed else 'CHECK'} "
          f"(layers within tol={args.tol} and argmax agrees)")

    if args.json:
        import json
        with open(args.json, "w") as jf:
            json.dump({
                "label": args.label,
                "layers": nla, "dim": dima, "vocab": va,
                "layer_max_abs_err": layer_max,       # worst per-layer max|Δ|
                "final_logit_max_abs_err": mx,
                "final_logit_cosine": cos,
                "worst_layer_cosine": worst_cos,
                "topk_overlap": overlap, "topk": args.topk,
                "argmax_match": bool(ta[0] == tb[0]),
                "argmax_a": ta[0], "argmax_b": tb[0],
                "pass": bool(passed),
            }, jf, indent=2)

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
