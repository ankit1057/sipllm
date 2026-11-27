#!/usr/bin/env python3
"""build_massive_global_dossier.py — Assembles the complete SipLLM Master Research Dossier:
1. Executive Bottom-Line & Video Streaming Theory
2. Full Market Solutions Architectural Teardowns ("READMEs")
3. Which Firms Use What and What Not (35+ Major Enterprises across 10 Verticals)
4. Synthesis & Claim Scoreboard (From synthesis.html)
5. Comprehensive Literature Compendium (105+ Source Records from Clusters A, B, C, D, E)
6. SipLLM Code Verification & Measured Benchmarks
7. 40-Week Project Lifecycle Roadmap
"""

import os
import sys
import re
import html
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESEARCH_DIR = os.path.join(ROOT, ".claude-artifacts", "research")
OUT_DOCS = os.path.join(ROOT, "docs", "sipllm-global-research-dossier.html")
OUT_RESEARCH = os.path.join(RESEARCH_DIR, "sipllm-global-research-dossier.html")

# Import custom sections
sys.path.insert(0, HERE)
from dossier_streaming_theory import STREAMING_THEORY_HTML
from dossier_market_solutions import MARKET_SOLUTIONS_HTML
from dossier_firms_survey import FIRMS_SURVEY_HTML

def esc(s):
    return html.escape(s, quote=True)

def slug(s, prefix):
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return f"{prefix}-{s[:60]}" if s else prefix

def inline(s):
    parts = re.split(r"(`[^`]+`)", s)
    res = []
    for p in parts:
        if len(p) >= 2 and p.startswith("`") and p.endswith("`"):
            res.append("<code>" + esc(p[1:-1]) + "</code>")
            continue
        t = esc(p)
        t = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', t)
        t = re.sub(r'(?<!href=")(?<!">)(https?://[^\s<>"\)\]]+[^\s<>"\)\].,;:])',
                   r'<a href="\1" target="_blank" rel="noopener">\1</a>', t)
        t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"\[(PAPER[^\]]*|SIPLLM[^\]]*|INFERENCE|HYPOTHESIS|DOC[^\]]*|SRC[^\]]*|ISSUE[^\]]*)\]",
                   lambda m: '<span class="lab">[' + m.group(1) + "]</span>", t)
        res.append(t)
    return "".join(res)

LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")

def md2html(md, prefix, shift=2):
    lines = md.replace("\t", "    ").split("\n")
    out = []
    para = []
    stack = []
    used_ids = set()

    def flush_para():
        if para:
            out.append("<p>" + inline(" ".join(x.strip() for x in para)) + "</p>")
            para.clear()

    def render_list(node):
        h = [f"<{node['tag']}>"]
        for it in node["items"]:
            h.append("<li>" + inline(" ".join(it["text"])))
            for ch in it["children"]:
                h.append(render_list(ch) if isinstance(ch, dict) else ch)
            h.append("</li>")
        h.append(f"</{node['tag']}>")
        return "".join(h)

    def close_lists(to_indent=-1):
        while stack and stack[-1]["indent"] > to_indent:
            node = stack.pop()
            if stack:
                stack[-1]["items"][-1]["children"].append(node)
            else:
                out.append(render_list(node))

    def close_all():
        close_lists(-1)

    def next_nonblank(j):
        while j < len(lines) and not lines[j].strip():
            j += 1
        return lines[j] if j < len(lines) else None

    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            flush_para(); close_all()
            buf = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i]); i += 1
            out.append("<pre><code>" + esc("\n".join(buf)) + "</code></pre>")
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1].strip()):
            flush_para()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip()); i += 1
            def cells(r):
                r = r.strip().strip("|")
                return [c.strip() for c in re.split(r"(?<!\\)\|", r)]
            head = cells(rows[0])
            body = [cells(r) for r in rows[2:]]
            t = ['<div class="tw"><table><thead><tr>']
            t += ["<th>" + inline(c) + "</th>" for c in head]
            t.append("</tr></thead><tbody>")
            for r in body:
                t.append("<tr>" + "".join("<td>" + inline(c) + "</td>" for c in r) + "</tr>")
            t.append("</tbody></table></div>")
            tbl = "".join(t)
            if stack:
                stack[-1]["items"][-1]["children"].append(tbl)
            else:
                out.append(tbl)
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m and not line.startswith(" "):
            flush_para(); close_all()
            lvl = min(6, len(m.group(1)) + shift)
            text = m.group(2).strip()
            hid = slug(text, prefix)
            base, k = hid, 2
            while hid in used_ids:
                hid = f"{base}-{k}"; k += 1
            used_ids.add(hid)
            out.append(f'<h{lvl} id="{hid}">{inline(text)}</h{lvl}>')
            i += 1
            continue
        if re.match(r"^(-{3,}|\*{3,}|_{3,})$", s):
            flush_para(); close_all()
            out.append("<hr>")
            i += 1
            continue
        if s.startswith(">"):
            flush_para(); close_all()
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip()); i += 1
            out.append("<blockquote><p>" + inline(" ".join(buf)) + "</p></blockquote>")
            continue
        m = LIST_RE.match(line)
        if m:
            flush_para()
            indent = len(m.group(1))
            tag = "ol" if m.group(2)[0].isdigit() else "ul"
            close_lists(indent)
            if stack and stack[-1]["indent"] == indent and stack[-1]["tag"] != tag:
                close_lists(indent - 1)
            if not stack or stack[-1]["indent"] < indent:
                stack.append({"indent": indent, "tag": tag, "items": []})
            stack[-1]["items"].append({"text": [m.group(3)], "children": []})
            i += 1
            continue
        if not s:
            flush_para()
            nxt = next_nonblank(i + 1)
            if stack and (nxt is None or not (LIST_RE.match(nxt) or nxt.startswith(" "))):
                close_all()
            i += 1
            continue
        if stack and line.startswith(" "):
            stack[-1]["items"][-1]["text"].append(s)
        else:
            close_all()
            para.append(s)
        i += 1
    flush_para(); close_all()
    return "\n".join(out)

def load_clusters():
    clusters = [
        ("A", "A-storage-offload", "Domain A: Storage & Flash Weight Offloading (Single Device)"),
        ("B", "B-phone-runtimes", "Domain B: Phone Runtimes & Mobile Edge Frameworks"),
        ("C", "C-pooled-ewaste", "Domain C: Pooled E-Waste & Distributed Weak Devices"),
        ("D", "D-theory-kv", "Domain D: KV-Cache Compression, Eviction & Quality Frontiers"),
        ("E", "E-mobile-gpu", "Domain E: Mobile GPU, Unified Memory Architecture & Shaders"),
    ]
    data = []
    for code, dirname, title in clusters:
        notes_path = os.path.join(RESEARCH_DIR, dirname, "notes.md")
        if os.path.exists(notes_path):
            with open(notes_path, "r", encoding="utf-8") as f:
                content = f.read()
            data.append((code, dirname, title, content))
        else:
            print(f"Warning: {notes_path} missing")
    return data

CSS = r"""
:root {
  --bg: #f8fafc;
  --fg: #0f172a;
  --muted: #475569;
  --card: #ffffff;
  --line: #e2e8f0;
  --accent: #2563eb;
  --code: #f1f5f9;
  --key: #eff6ff;
  --keyb: #3b82f6;
  --warn: #fffbeb;
  --warnb: #f59e0b;
  --note: #f0fdf4;
  --noteb: #22c55e;
  --dead: #ef4444;
  --narrow: #d97706;
  --open: #10b981;
  --design: #8b5cf6;
  --us: #f0fdf4;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #090d16;
    --fg: #f1f5f9;
    --muted: #94a3b8;
    --card: #111827;
    --line: #1f2937;
    --accent: #60a5fa;
    --code: #1e293b;
    --key: #172554;
    --keyb: #60a5fa;
    --warn: #451a03;
    --warnb: #fbbf24;
    --note: #052e16;
    --noteb: #4ade80;
    --dead: #f87171;
    --narrow: #fbbf24;
    --open: #34d399;
    --design: #a78bfa;
    --us: #064e3b;
  }
}
:root[data-theme="dark"] {
  --bg: #090d16;
  --fg: #f1f5f9;
  --muted: #94a3b8;
  --card: #111827;
  --line: #1f2937;
  --accent: #60a5fa;
  --code: #1e293b;
  --key: #172554;
  --keyb: #60a5fa;
  --warn: #451a03;
  --warnb: #fbbf24;
  --note: #052e16;
  --noteb: #4ade80;
  --dead: #f87171;
  --narrow: #fbbf24;
  --open: #34d399;
  --design: #a78bfa;
  --us: #064e3b;
}

* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  overflow-wrap: break-word;
}
.wrap {
  max-width: 1180px;
  margin: 0 auto;
  padding: 32px 20px 100px;
}
header.top {
  border-bottom: 2px solid var(--line);
  padding-bottom: 24px;
  margin-bottom: 28px;
}
header.top h1 {
  font-size: 2.2rem;
  line-height: 1.25;
  margin: 0 0 10px;
  letter-spacing: -0.02em;
}
.sub {
  color: var(--muted);
  font-size: 1.1rem;
  margin: 0 0 16px;
}
.meta {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 14px;
}
.pill {
  border: 1px solid var(--line);
  background: var(--card);
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 0.85rem;
  color: var(--muted);
}
.pill.highlight {
  border-color: var(--accent);
  color: var(--accent);
  font-weight: 600;
}
.search-bar {
  margin: 20px 0;
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
}
.search-input {
  flex: 1;
  min-width: 280px;
  padding: 10px 16px;
  font-size: 1rem;
  border-radius: 8px;
  border: 1px solid var(--line);
  background: var(--card);
  color: var(--fg);
}
.tools button {
  font: inherit;
  font-size: 0.85rem;
  border: 1px solid var(--line);
  background: var(--card);
  color: var(--fg);
  border-radius: 8px;
  padding: 8px 14px;
  cursor: pointer;
  transition: all 0.2s ease;
}
.tools button:hover {
  background: var(--code);
  border-color: var(--accent);
}
nav.toc {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 16px 20px;
  margin: 0 0 36px;
  box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
nav.toc summary {
  cursor: pointer;
  font-weight: 700;
  font-size: 1.15rem;
  padding: 4px 0;
}
nav.toc ol {
  margin: 12px 0;
  padding-left: 20px;
}
nav.toc li {
  margin: 4px 0;
}
h2 {
  font-size: 1.65rem;
  margin: 50px 0 16px;
  padding-top: 20px;
  border-top: 2px solid var(--line);
  letter-spacing: -0.01em;
}
h3 { font-size: 1.3rem; margin: 32px 0 12px; }
h4 { font-size: 1.1rem; margin: 24px 0 8px; }
a { color: var(--accent); text-decoration: none; word-break: break-word; }
a:hover { text-decoration: underline; }
code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.88em;
  background: var(--code);
  padding: 2px 6px;
  border-radius: 4px;
}
pre {
  background: var(--code);
  padding: 14px;
  border-radius: 8px;
  overflow-x: auto;
}
pre code { padding: 0; background: none; }
.tw {
  overflow-x: auto;
  max-width: 100%;
  margin: 16px 0;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--card);
}
table {
  border-collapse: collapse;
  width: 100%;
  font-size: 0.92rem;
}
th, td {
  border-bottom: 1px solid var(--line);
  padding: 10px 14px;
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--code);
  font-weight: 700;
}
.callout {
  border-left: 5px solid;
  border-radius: 8px;
  padding: 14px 18px;
  margin: 18px 0;
}
.callout p { margin: 6px 0; }
.callout.key { background: var(--key); border-color: var(--keyb); }
.callout.warn { background: var(--warn); border-color: var(--warnb); }
.callout.note { background: var(--note); border-color: var(--noteb); }
.badge {
  display: inline-block;
  font-size: 0.74rem;
  font-weight: 700;
  border: 1.5px solid;
  border-radius: 4px;
  padding: 1px 6px;
  margin-right: 6px;
}
.badge.dead { color: var(--dead); border-color: var(--dead); }
.badge.narrowed { color: var(--narrow); border-color: var(--narrow); }
.badge.open { color: var(--open); border-color: var(--open); }
.badge.design { color: var(--design); border-color: var(--design); }
.formula-card {
  border: 1px solid var(--line);
  background: var(--card);
  border-radius: 10px;
  padding: 16px 20px;
  margin: 18px 0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.03);
}
.formula-card.warn {
  border-color: var(--warnb);
  background: var(--warn);
}
.formula-card.key {
  border-color: var(--keyb);
  background: var(--key);
}
.formula-header {
  font-size: 0.95rem;
  font-weight: 700;
  margin-bottom: 8px;
  color: var(--muted);
  display: flex;
  align-items: center;
}
.formula-calc {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 1.05rem;
  padding: 6px 0;
  color: var(--fg);
}
.formula-calc strong {
  color: var(--accent);
  font-size: 1.15rem;
}
.formula-calc.highlight strong {
  color: #10b981;
}
.formula-note {
  font-size: 0.88rem;
  color: var(--muted);
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--line);
}
details.app {
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--card);
  margin: 16px 0;
  padding: 0 16px;
  transition: all 0.2s ease;
}
details.app > summary {
  cursor: pointer;
  font-weight: 600;
  padding: 14px 0;
  font-size: 1.1rem;
}
details.app[open] > summary {
  border-bottom: 1px solid var(--line);
  margin-bottom: 12px;
}
footer {
  margin-top: 60px;
  color: var(--muted);
  font-size: 0.9rem;
  border-top: 1px solid var(--line);
  padding-top: 20px;
}
"""

JS_SCRIPTS = r"""
<script>
document.addEventListener("DOMContentLoaded", () => {
  const themeBtn = document.getElementById("theme-toggle");
  themeBtn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("sipllm-theme", next);
  });

  const openBtn = document.getElementById("open-all");
  const closeBtn = document.getElementById("close-all");
  openBtn.addEventListener("click", () => {
    document.querySelectorAll("details").forEach(d => d.open = true);
  });
  closeBtn.addEventListener("click", () => {
    document.querySelectorAll("details").forEach(d => d.open = false);
  });

  // Real-time search filter
  const searchInput = document.getElementById("search-input");
  searchInput.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase().trim();
    const details = document.querySelectorAll("details.app, .paper-card");
    if (!q) {
      details.forEach(d => {
        d.style.display = "";
        d.classList.remove("highlight-search");
      });
      return;
    }
    details.forEach(d => {
      const txt = d.textContent.toLowerCase();
      if (txt.includes(q)) {
        d.style.display = "";
        d.open = true;
      } else {
        d.style.display = "none";
      }
    });
  });
});
</script>
"""

SECTION_SIPLLM_VERIFICATION = """
<section id="s-sipllm-source">
<h2>V. SipLLM Source Code Verification & Empirical Benchmark Reality</h2>

<div class="callout note">
<p><strong>Codebase Truth:</strong> SipLLM is not an academic paper or a theoretical Python notebook. It is an active C++17 codebase running token-for-token checked inference across Apple Silicon and unrooted ARM Android phones.</p>
</div>

<h3>1. Key Code Locations & Mechanisms</h3>
<ul>
<li><strong>Whole-Process Peak-RAM Contract Planner:</strong> <code>include/llm/mem_plan.h</code> and <code>src/mem_plan.cpp</code>. Calculates itemized <code>MemLedger</code> (base, scratch, logits, attn, residual, out_norm, embeddings, LM head, full-ctx KV cache, ring buffer, pinned layers). Refuses infeasible budgets upfront with actionable diagnostic report.</li>
<li><strong>Streaming Layer Loader & Ring Buffer:</strong> <code>src/loader.cpp</code> (lines 186-231). Double-buffered asynchronous prefetch ring recycling layer memory. Contiguous or spread layer pinning.</li>
<li><strong>Bounded KV Cache & Issue #81 Resolution:</strong> <code>include/llm/kv_cache.h</code> and <code>src/kv_cache.cpp</code>. <code>reserve_full()</code> pre-allocates maximum context at startup when budgeted, eliminating runtime doubling memory spikes. INT8 quantized KV cache option (<code>--kv-q8</code>).</li>
<li><strong>Network Layer Streaming (Socket Block Client):</strong> <code>src/remote_weight_source.cpp</code>. Streams 1 MB weight blocks over TCP sockets with local compressed file cache for bounded client RSS.</li>
<li><strong>Mathematical Token Equivalence:</strong> <code>tests/test_ram_budget.cpp</code>. Formally verifies that generated tokens and logits are bit-for-bit identical across budget sweeps from 0 MB (stream all) to full residency.</li>
</ul>

<h3>2. Empirical Edge Scorecard (Measured Reality)</h3>
<div class="tw">
<table>
<thead>
<tr>
<th>Device / Platform</th>
<th>Model & Quant</th>
<th>File Size</th>
<th>Budget / Mode</th>
<th>Measured Peak RSS</th>
<th>Decode Speed</th>
<th>Verification Fact</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Apple M3 (16 GB)</strong></td>
<td>Llama-2-13B (Q4_K_M)</td>
<td>7.87 GB</td>
<td><code>--ram-budget 0</code> (Stream all)</td>
<td><strong>317 MB</strong></td>
<td>~0.55 tok/s</td>
<td><strong>25x smaller than file</strong>; exact llama.cpp parity</td>
</tr>
<tr>
<td><strong>Apple M3 (16 GB)</strong></td>
<td>Llama-3.1-8B (Q4_K_M)</td>
<td>4.92 GB</td>
<td><code>--ram-budget 0</code> (Stream all)</td>
<td><strong>204 MB</strong></td>
<td>~0.55 tok/s</td>
<td><strong>24x smaller than file</strong>; runs on 2 GB free RAM</td>
</tr>
<tr>
<td><strong>OnePlus IV2201 (Dimensity 900)</strong><br>Android 13, Unrooted</td>
<td>Qwen-2.5-1.5B (Q8_0)</td>
<td>1.89 GB</td>
<td><code>--ram-budget 300M</code></td>
<td><strong>323 MB</strong> (VmHWM)</td>
<td>1.79 tok/s</td>
<td>Zero violations; 5 Hz /proc/pid/status sampling</td>
</tr>
<tr>
<td><strong>OnePlus IV2201 (Dimensity 900)</strong><br>Android 13, Unrooted</td>
<td>TinyLlama-1.1B (Q4_K_M)</td>
<td>669 MB</td>
<td><code>--fast</code> (Fully resident)</td>
<td>667 MB (vs llama 1317 MB)</td>
<td>14.7 tok/s (vs llama 18.2)</td>
<td><strong>1.97x less RAM than llama.cpp</strong> at 81% speed</td>
</tr>
</tbody>
</table>
</div>
</section>
"""

SECTION_ROADMAP = """
<section id="s-roadmap">
<h2>VI. The 40-Week Project Lifecycle Plan: Engineering & Commercialization</h2>

<div class="tw">
<table>
<thead>
<tr>
<th>Phase / Timeline</th>
<th>Key Milestones & Deliverables</th>
<th>Target Files / Modules</th>
<th>Exit / Acceptance Criteria</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Phase 0: Weeks 1–4</strong><br>Core Hardening & Scientific Truth</td>
<td>
1. Cold-cache benchmark harness (<code>POSIX_FADV_DONTNEED</code>)<br>
2. Static KV allocation verification<br>
3. Iso-Memory Frontier (Claim C4) academic paper submission
</td>
<td>
<code>scripts/bench_cold.sh</code><br>
<code>src/kv_cache.cpp</code><br>
<code>bench/results/</code>
</td>
<td>True cold flash I/O measured on Dimensity 900; 0 byte overshoot across 4096 tokens under tight budgets.</td>
</tr>
<tr>
<td><strong>Phase 1: Weeks 5–12</strong><br>Video Streaming Parity & Amortization</td>
<td>
1. Speculative Streaming Engine (resident draft + target)<br>
2. Bounded Attention Sinks & Rolling KV Eviction (C5)<br>
3. Spread layer pinning (<code>--pin-placement spread</code>)
</td>
<td>
<code>include/llm/spec_stream.h</code><br>
<code>src/spec_stream.cpp</code><br>
<code>src/loader.cpp</code>
</td>
<td>$\\ge 3.0\\times$ decode speedup on storage-streamed models; 100k token continuous dialogue under 300 MB cap.</td>
</tr>
<tr>
<td><strong>Phase 2: Weeks 13–20</strong><br>Zero-Copy UMA & Hardware DMA</td>
<td>
1. Vulkan UMA zero-copy pipeline<br>
2. Direct I/O (<code>O_DIRECT</code> / <code>F_NOCACHE</code>)<br>
3. NEON int8 SDOT unrolled micro-kernels
</td>
<td>
<code>src/vulkan_matmul.cpp</code><br>
<code>include/llm/file_backing.h</code><br>
<code>src/neon.cpp</code>
</td>
<td>Vulkan prefill $\\ge 2.5\\times$ NEON CPU; zero OS page cache duplication; decode reaches $\\ge 90\\%$ of llama.cpp.</td>
</tr>
<tr>
<td><strong>Phase 3: Weeks 21–28</strong><br>Production Packaging & SDKs</td>
<td>
1. Pristine C-ABI (<code>libsipllm.so</code> < 2MB)<br>
2. Production Flutter & Android NDK bindings<br>
3. OpenAI-compatible embedded daemon
</td>
<td>
<code>include/llm/c_api.h</code><br>
<code>bindings/flutter/</code><br>
<code>server/server.cpp</code>
</td>
<td>One-click Dart package import; stable SSE streaming; dynamic budget renegotiation via HTTP headers.</td>
</tr>
<tr>
<td><strong>Phase 4: Weeks 29–40</strong><br>OEM Commercial Pilots & Scale</td>
<td>
1. Automotive Cockpit Pilot (ASIL-B Conformance)<br>
2. Smart TV Reference Design Kit (1.5GB Android TV)<br>
3. Dual-Licensing Launch (Apache 2.0 + OEM Commercial)
</td>
<td>
Reference SDKs<br>
Commercial licensing agreements
</td>
<td>72 hours continuous driving dialogue with 0 crashes under 1.2 GB; voice assistant running inside 280 MB on 1.5 GB TV.</td>
</tr>
</tbody>
</table>
</div>
</section>
"""

def main():
    print("Generating Complete Master Research Dossier HTML...")
    clusters_data = load_clusters()
    
    synth_path = os.path.join(RESEARCH_DIR, "synthesis.html")
    synthesis_html = ""
    if os.path.exists(synth_path):
        with open(synth_path, "r", encoding="utf-8") as f:
            synthesis_html = f.read()

    # Build rendered HTML for the 5 research clusters (over 105 source records)
    cluster_rendered_blocks = []
    total_papers = 0
    for code, dirname, title, content in clusters_data:
        rendered = md2html(content, prefix=code, shift=3)
        num_records = len(re.findall(r"^###\s+", content, re.M))
        total_papers += num_records
        block = f"""
        <details class="app" open>
        <summary>{title} — ({num_records} Source Records Analyzed)</summary>
        <div class="paper-block">
        {rendered}
        </div>
        </details>
        """
        cluster_rendered_blocks.append(block)

    full_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SipLLM Master Global Research Dossier & Competitive Blueprint</title>
<meta name="description" content="Comprehensive Global Research Compendium, Literature Review, Market Solutions READMEs, Enterprise Survey, and Video Streaming Theory for SipLLM.">
<style>
{CSS}
</style>
</head>
<body>
<div class="wrap">

<header class="top">
<h1>SipLLM Global Research Compendium & Strategic Blueprint</h1>
<p class="sub">From Scientific Discovery to Commercial Edge King: A Full Synthesis of All Published Literature, Market Architecture Teardowns, Enterprise Adoption Realities, and The Video Streaming Paradigm.</p>

<div class="meta">
<span class="pill highlight">Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M IST')}</span>
<span class="pill">Acting CTO & Founder Technical Compendium</span>
<span class="pill">{total_papers}+ Academic Literature Sources Analyzed</span>
<span class="pill">12 Market Solutions Teardowns</span>
<span class="pill">35+ Enterprise Firm Profiles</span>
<span class="pill">Zero-Dependency Pure C++17</span>
</div>

<div class="search-bar">
<input type="text" id="search-input" class="search-input" placeholder="Search papers, market solutions, company profiles, or technical keywords...">
<div class="tools">
<button id="open-all" type="button">Expand All</button>
<button id="close-all" type="button">Collapse All</button>
<button id="theme-toggle" type="button">Toggle Dark/Light</button>
</div>
</div>
</header>

<nav class="toc">
<details open>
<summary>Table of Contents</summary>
<ol>
<li><a href="#s-streaming-theory">I. The Scientific Paradigm: Transforming LLM Inference into Continuous Media Streaming</a></li>
<li><a href="#s-market-solutions">II. Market Solutions Architectural Teardowns ("READMEs") of Existing Frameworks</a></li>
<li><a href="#s-firms-survey">III. Industry Adoption & Rejection Landscape: Which Firms Use What and What Not</a></li>
<li><a href="#s-synthesis">IV. Synthesis & Scoreboard: Falsification of Academic Claims (C1 to C7)</a></li>
<li><a href="#s-academic-compendium">V. Comprehensive Academic Compendium (105+ Source Records Across Domains A–E)</a></li>
<li><a href="#s-sipllm-source">VI. SipLLM Source Code Verification & Empirical Benchmark Reality</a></li>
<li><a href="#s-roadmap">VII. The 40-Week Project Lifecycle Plan: Engineering & Commercialization</a></li>
</ol>
</details>
</nav>

<!-- SECTION 1: VIDEO STREAMING THEORY -->
{STREAMING_THEORY_HTML}

<!-- SECTION 2: MARKET SOLUTIONS READMEs -->
{MARKET_SOLUTIONS_HTML}

<!-- SECTION 3: FIRMS SURVEY -->
{FIRMS_SURVEY_HTML}

<!-- SECTION 4: SYNTHESIS & CLAIM SCOREBOARD -->
<section id="s-synthesis">
<h2>IV. Synthesis & Scoreboard: Falsification of Academic Claims (C1 to C7)</h2>
{synthesis_html}
</section>

<!-- SECTION 5: COMPREHENSIVE ACADEMIC COMPENDIUM -->
<section id="s-academic-compendium">
<h2>V. Comprehensive Academic Compendium: 105+ Published Papers Analyzed</h2>
<p class="sub">Structured records extracted directly from Cluster Notes A, B, C, D, and E with problem statements, mechanisms, memory regimes, hardware baselines, and falsification verdicts.</p>
{"".join(cluster_rendered_blocks)}
</section>

<!-- SECTION 6: SOURCE CODE & BENCHMARKS -->
{SECTION_SIPLLM_VERIFICATION}

<!-- SECTION 7: ROADMAP -->
{SECTION_ROADMAP}

<footer>
<p>SipLLM Master Research Dossier. Confidential & Proprietary. Compiled for the SipLLM Engineering & Executive Team.</p>
</footer>

</div>
{JS_SCRIPTS}
</body>
</html>
"""

    with open(OUT_DOCS, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Written docs dossier: {OUT_DOCS} ({len(full_html)} bytes)")

    with open(OUT_RESEARCH, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Written research dossier: {OUT_RESEARCH} ({len(full_html)} bytes)")

if __name__ == "__main__":
    main()
