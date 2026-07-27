#!/usr/bin/env python3
"""SipLLM documentation portal — dependency-free static-site generator.

Markdown is the source of truth. Each page under a section folder
(`product/`, `architecture/`, `api/`, `journal/`, `meta/`) is authored as
Markdown and rendered into a polished, offline, searchable HTML portal in
`../site/`. No third-party packages — a small Markdown engine (headings,
paragraphs, lists, GFM tables, fenced code, blockquotes, GitHub-style
admonitions, inline formatting, raw-HTML passthrough) lives here in stdlib.

    python3 docs/build.py

Output (`site/`) has no build step and no network calls: open `index.html`,
or serve it on GitHub Pages (a `.nojekyll` marker is emitted so Pages does
not strip files). All links are relative, so it works at any base path
(e.g. `https://ankit1057.github.io/<repo>/`).

To add a page: drop a `.md` in the right section folder and add one row to
`NAV` below. Ordering + sidebar grouping is defined entirely by `NAV`.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

SITE_TITLE = "SipLLM"
POSITIONING = ("Run GGUF models larger than available RAM through "
               "bounded-memory transformer layer streaming.")
VERSION = "v0.4.0 · portal v0.5"
REPO_URL = "https://github.com/ankit1057/sipllm"

ROOT = Path(__file__).resolve().parent           # docs/
OUT = ROOT.parent / "site"                        # repo-root/site (the Pages submodule)

# section -> [(slug, md relative path, sidebar label)]. Order here IS the order.
NAV = [
    ("Product", [
        ("index",               "product/overview.md",            "Overview"),
        ("what-is-sipllm",      "product/what-is-sipllm.md",      "What is SipLLM?"),
        ("why-streaming",       "product/why-streaming.md",       "Why streaming?"),
        ("why-not-mmap",        "product/why-not-mmap.md",        "Why not mmap everything?"),
        ("why-another-runtime", "product/why-another-runtime.md", "Why another runtime?"),
        ("vision",              "product/vision.md",              "Vision"),
        ("roadmap",             "product/roadmap.md",             "Roadmap"),
        ("benchmarks",          "product/benchmarks.md",          "Benchmarks"),
        ("faq",                 "product/faq.md",                 "FAQ"),
    ]),
    ("Architecture", [
        ("architecture",     "architecture/overview.md",         "Overview"),
        ("streaming-loader", "architecture/streaming-loader.md", "Streaming loader"),
        ("layer-residency",  "architecture/layer-residency.md",  "Layer residency"),
        ("memory-planner",   "architecture/memory-planner.md",   "Memory planner"),
        ("prefetch",         "architecture/prefetch.md",         "Prefetch"),
        ("runtime",          "architecture/runtime.md",          "Runtime"),
        ("transformer",      "architecture/transformer.md",      "Transformer"),
        ("kv-cache",         "architecture/kv-cache.md",         "KV cache"),
        ("quantization",     "architecture/quantization.md",     "Quantization"),
        ("gguf-parser",      "architecture/gguf-parser.md",      "GGUF parser"),
        ("tokenizer",        "architecture/tokenizer.md",        "Tokenizer"),
        ("thread-pool",      "architecture/thread-pool.md",      "Thread pool"),
        ("auto-tuning",      "architecture/auto-tuning.md",      "Auto-tuning"),
        ("flutter-runtime",  "architecture/flutter-runtime.md",  "Flutter runtime"),
        ("wear-support",     "architecture/wear-support.md",     "Wear OS support"),
    ]),
    ("API reference", [
        ("api",      "api/overview.md",    "Overview"),
        ("api-c",    "api/c-abi.md",       "C ABI"),
        ("api-cpp",  "api/cpp-engine.md",  "C++ engine"),
        ("api-dart", "api/dart.md",        "Dart / Flutter"),
    ]),
    ("Engineering journal", [
        ("journal",              "journal/index.md",              "Overview"),
        ("j-streaming-layers",   "journal/streaming-layers.md",   "How SipLLM streams layers"),
        ("j-bigger-than-ram",    "journal/bigger-than-ram.md",    "Running models larger than RAM"),
        ("j-lm-head",            "journal/streaming-lm-head.md",  "Streaming the LM head"),
        ("j-prefill",            "journal/prefill.md",            "Prefill optimization"),
        ("j-prefetch",           "journal/predictive-prefetch.md","Predictive prefetch"),
        ("j-resident-weights",   "journal/resident-weights.md",   "Resident weights"),
        ("j-q4k",                "journal/q4k-explained.md",       "Q4_K explained"),
        ("j-gguf",               "journal/gguf-internals.md",      "GGUF internals"),
        ("j-peak-rss",           "journal/why-peak-rss.md",        "Why peak RSS matters"),
        ("j-android",            "journal/android-inference.md",   "Android inference"),
    ]),
    ("Project", [
        ("design-decisions",     "meta/design-decisions.md",     "Design decisions"),
        ("rfc-index",            "meta/rfc-index.md",            "RFC index"),
        ("performance-history",  "meta/performance-history.md",  "Performance history"),
        ("competitive-analysis", "meta/competitive-analysis.md", "Competitive analysis"),
        ("glossary",             "meta/glossary.md",             "Glossary"),
        ("contributing",         "meta/contributing.md",         "Contributor guide"),
    ]),
]

# flat ordered list for prev/next + lookups
FLAT = [(sec, slug, path, label) for sec, pages in NAV for (slug, path, label) in pages]
LABEL = {slug: label for _, slug, _, label in FLAT}


# ---------------------------------------------------------------- markdown ----
def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text).strip().lower()
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-") or "section"


_CODE = "\x00%d\x00"


def render_inline(s: str) -> str:
    stash: list[str] = []

    def keep_code(m: re.Match) -> str:
        stash.append("<code>" + html.escape(m.group(1)) + "</code>")
        return _CODE % (len(stash) - 1)

    s = re.sub(r"`([^`]+)`", keep_code, s)
    s = html.escape(s)
    s = re.sub(r"&amp;(#?[0-9a-zA-Z]+);", r"&\1;", s)  # keep author entities (&nbsp; &times; …)
    s = re.sub(r"!\[([^\]]*)\]\(([^)\s]+)\)", r'<img src="\2" alt="\1" loading="lazy">', s)
    s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"(?<![\w_])_([^_\n]+)_(?![\w_])", r"<em>\1</em>", s)
    for i, frag in enumerate(stash):
        s = s.replace(_CODE % i, frag)
    return s


_HTML_BLOCK = re.compile(r"^\s*<(?:/?)(?:div|section|aside|figure|svg|table|pre|nav|"
                         r"ul|ol|blockquote|details|summary|h[1-6]|p|img|br|hr|!--)")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")


def _split_row(row: str) -> list[str]:
    row = row.strip()
    if row.startswith("|"):
        row = row[1:]
    if row.endswith("|"):
        row = row[:-1]
    return [c.strip() for c in row.split("|")]


def render_markdown(text: str):
    """Return (html_body, toc[list of (level,id,text)], plaintext_for_search)."""
    lines = text.split("\n")
    out: list[str] = []
    toc: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    plain: list[str] = []
    i, n = 0, len(lines)

    def uid(base: str) -> str:
        hid, k = base, 2
        while hid in seen:
            hid, k = f"{base}-{k}", k + 1
        seen.add(hid)
        return hid

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # fenced code
        m = re.match(r"^```(\w*)\s*$", line)
        if m:
            lang = m.group(1)
            body = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1  # closing fence
            code = html.escape("\n".join(body))
            cls = f' class="language-{lang}"' if lang else ""
            out.append(f"<pre><code{cls}>{code}</code></pre>")
            continue

        # raw HTML block (passthrough until blank line)
        if _HTML_BLOCK.match(line):
            body = []
            while i < n and lines[i].strip() != "":
                body.append(lines[i])
                i += 1
            out.append("\n".join(body))
            continue

        # ATX heading
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            level = len(m.group(1))
            txt = m.group(2)
            hid = uid(slugify(txt))
            inner = render_inline(txt)
            out.append(f'<h{level} id="{hid}">{inner}</h{level}>')
            if level in (2, 3):
                toc.append((level, hid, re.sub(r"<[^>]+>", "", inner)))
            plain.append(re.sub(r"<[^>]+>", "", inner))
            i += 1
            continue

        # horizontal rule
        if re.match(r"^(---+|\*\*\*+|___+)\s*$", stripped):
            out.append("<hr>")
            i += 1
            continue

        # GFM table
        if "|" in line and i + 1 < n and _TABLE_SEP.match(lines[i + 1]):
            header = _split_row(line)
            aligns = []
            for cell in _split_row(lines[i + 1]):
                l, r = cell.startswith(":"), cell.endswith(":")
                aligns.append("center" if l and r else "right" if r else "left" if l else "")
            i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(_split_row(lines[i]))
                i += 1
            th = "".join(
                f'<th{f" class=\"a-{a}\"" if a else ""}>{render_inline(c)}</th>'
                for c, a in zip(header, aligns + [""] * len(header)))
            trs = []
            for r in rows:
                tds = "".join(
                    f'<td{f" class=\"a-{a}\"" if a else ""}>{render_inline(c)}</td>'
                    for c, a in zip(r, aligns + [""] * len(r)))
                trs.append(f"<tr>{tds}</tr>")
                plain.append(" ".join(re.sub(r"<[^>]+>", "", render_inline(c)) for c in r))
            out.append('<div class="table-wrap"><table><thead><tr>'
                       f"{th}</tr></thead><tbody>{''.join(trs)}</tbody></table></div>")
            continue

        # blockquote / admonition
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            adm = re.match(r"^\[!(\w+)\]\s*(.*)$", buf[0]) if buf else None
            if adm:
                kind = adm.group(1).lower()
                title = adm.group(2).strip()
                rest = "\n".join(buf[1:])
                titles = {"note": "Note", "warning": "Caution", "caution": "Caution",
                          "key": "Key idea", "measured": "Measured", "tip": "Tip"}
                cls = {"warning": "warn", "caution": "warn", "key": "key",
                       "measured": "measured", "tip": "note", "note": "note"}.get(kind, "note")
                inner, _, ptext = render_markdown(rest)
                plain.append(ptext)
                label = title or titles.get(kind, "Note")
                out.append(f'<div class="callout {cls}"><div class="callout-title">'
                           f"{html.escape(label)}</div>{inner}</div>")
            else:
                inner, _, ptext = render_markdown("\n".join(buf))
                plain.append(ptext)
                out.append(f"<blockquote>{inner}</blockquote>")
            continue

        # lists (nested by indent)
        if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
            items = []
            while i < n and (re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]) or
                             (lines[i].strip() and lines[i].startswith(("  ", "\t")))):
                items.append(lines[i])
                i += 1
            out.append(_render_list(items, plain))
            continue

        # paragraph
        buf = []
        while i < n and lines[i].strip() and not _para_break(lines[i], lines, i):
            buf.append(lines[i].strip())
            i += 1
        para = render_inline(" ".join(buf))
        out.append(f"<p>{para}</p>")
        plain.append(re.sub(r"<[^>]+>", "", para))

    return "\n".join(out), toc, " ".join(plain)


def _para_break(line: str, lines: list[str], i: int) -> bool:
    if re.match(r"^(#{1,6}\s|```|>|\s*([-*+]|\d+\.)\s|(---+|\*\*\*+)\s*$)", line):
        return True
    if _HTML_BLOCK.match(line):
        return True
    if "|" in line and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
        return True
    return False


def _render_list(item_lines: list[str], plain: list[str]) -> str:
    """Stack-based nested list renderer (2-space indent = one level)."""
    root: list = []
    stack = [(-1, root)]
    for raw in item_lines:
        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", raw)
        if not m:
            if root:
                _append_text(stack[-1][1], " " + raw.strip())
            continue
        indent = len(m.group(1).expandtabs(2))
        ordered = bool(re.match(r"\d+\.", m.group(2)))
        text = m.group(3)
        plain.append(text)
        while stack and stack[-1][0] >= indent and len(stack) > 1:
            stack.pop()
        parent = stack[-1][1]
        node = {"text": text, "ordered": ordered, "children": []}
        parent.append(node)
        stack.append((indent, node["children"]))
    return _emit_list(root)


def _append_text(children: list, extra: str) -> None:
    if children:
        children[-1]["text"] += extra


def _emit_list(nodes: list) -> str:
    if not nodes:
        return ""
    tag = "ol" if nodes[0]["ordered"] else "ul"
    parts = [f"<{tag}>"]
    for node in nodes:
        inner = render_inline(node["text"])
        child = _emit_list(node["children"]) if node["children"] else ""
        parts.append(f"<li>{inner}{child}</li>")
    parts.append(f"</{tag}>")
    return "".join(parts)


# ------------------------------------------------------------------- shell ----
def render_toc(toc) -> str:
    if len(toc) < 2:
        return "<aside class=\"toc\"></aside>"
    items = "".join(
        f'<li class="toc-h{lvl}"><a href="#{hid}">{html.escape(t)}</a></li>'
        for lvl, hid, t in toc)
    return ('<aside class="toc" aria-label="On this page">'
            '<div class="toc-title">On this page</div>'
            f"<ul>{items}</ul></aside>")


def render_sidebar(active: str) -> str:
    out = ['<nav class="sidebar-nav" aria-label="Documentation">']
    for section, pages in NAV:
        out.append(f'<div class="nav-group"><div class="nav-group-title">{html.escape(section)}</div><ul>')
        for slug, _path, label in pages:
            cls = ' class="active"' if slug == active else ""
            cur = ' aria-current="page"' if slug == active else ""
            out.append(f'<li><a href="{slug}.html"{cls}{cur}>{html.escape(label)}</a></li>')
        out.append("</ul></div>")
    out.append("</nav>")
    return "".join(out)


def prevnext(idx: int) -> str:
    parts = []
    if idx > 0:
        _, ps, _, pl = FLAT[idx - 1]
        parts.append(f'<a class="pn prev" href="{ps}.html"><span>Previous</span>{html.escape(pl)}</a>')
    else:
        parts.append('<span class="pn"></span>')
    if idx < len(FLAT) - 1:
        _, ns, _, nl = FLAT[idx + 1]
        parts.append(f'<a class="pn next" href="{ns}.html"><span>Next</span>{html.escape(nl)}</a>')
    return "".join(parts)


SHELL = """<!DOCTYPE html>
<html lang="en" data-slug="{slug}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · {site} docs</title>
<meta name="description" content="{desc}">
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<a class="skip-link" href="#content">Skip to content</a>
<header class="topbar">
  <button class="menu-toggle" aria-label="Toggle navigation" aria-expanded="false">&#9776;</button>
  <a class="brand" href="index.html">
    <span class="brand-mark" aria-hidden="true"></span>
    <span class="brand-name">{site}</span>
    <span class="brand-sub">docs</span>
  </a>
  <div class="search" role="search">
    <input type="search" id="search-input" placeholder="Search the docs…  (press /)"
           aria-label="Search documentation" autocomplete="off">
    <div class="search-results" id="search-results" hidden></div>
  </div>
  <div class="topbar-right">
    <span class="version-badge">{version}</span>
    <a class="repo-link" href="{repo}" target="_blank" rel="noopener">GitHub &#8599;</a>
  </div>
</header>
<div class="layout">
  <div class="sidebar" id="sidebar">{sidebar}</div>
  <div class="sidebar-scrim" id="sidebar-scrim"></div>
  <main id="content" class="content">
    <article class="page {pageclass}">
      <div class="breadcrumb">{section} <span aria-hidden="true">/</span> {label}</div>
      <h1>{h1}</h1>
      {body}
      <nav class="page-footer-nav">{prevnext}</nav>
    </article>
    {toc}
  </main>
</div>
<footer class="site-footer">
  <div><strong>{site}</strong> — {positioning} Performance figures are <em>measured</em>
  (Apple&nbsp;M3, warm cache, median-of-3 unless noted) and reproducible via
  <code>scripts/bench.sh</code>; raw JSON in <code>bench/results/</code>. Unmeasured metrics are marked N/A.</div>
  <div class="footer-meta">Dependency-free C++17 · CPU-first · offline-first · {version}</div>
</footer>
<script src="assets/search-index.js"></script>
<script src="assets/app.js"></script>
</body>
</html>
"""


def meta_desc(body_plain: str) -> str:
    t = re.sub(r"\s+", " ", body_plain).strip()
    return (t[:157] + "…") if len(t) > 158 else (t or POSITIONING)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "assets").mkdir(exist_ok=True)
    (OUT / ".nojekyll").write_text("", encoding="utf-8")

    # copy static assets from docs/assets -> site/assets
    for asset in ("style.css", "app.js"):
        src = ROOT / "assets" / asset
        if src.exists():
            (OUT / "assets" / asset).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    search_index = []
    built = missing = 0
    for idx, (section, slug, relpath, label) in enumerate(FLAT):
        md = ROOT / relpath
        if not md.exists():
            missing += 1
            print(f"  [pending] {relpath}")
            continue
        raw = md.read_text(encoding="utf-8")
        # first H1 becomes the title; strip it from the body (shell renders <h1>)
        h1_match = re.search(r"^#\s+(.*)$", raw, flags=re.MULTILINE)
        title = h1_match.group(1).strip() if h1_match else label
        if h1_match:
            raw = raw[:h1_match.start()] + raw[h1_match.end():]
        body, toc, plain = render_markdown(raw)
        page = SHELL.format(
            slug=slug, title=html.escape(title), site=SITE_TITLE,
            version=html.escape(VERSION), repo=REPO_URL,
            desc=html.escape(meta_desc(plain)), sidebar=render_sidebar(slug),
            section=html.escape(section), label=html.escape(label),
            h1=render_inline(title), body=body, toc=render_toc(toc),
            prevnext=prevnext(idx), positioning=html.escape(POSITIONING),
            pageclass=("home" if slug == "index" else ""),
        )
        (OUT / f"{slug}.html").write_text(page, encoding="utf-8")
        search_index.append({
            "slug": slug, "title": title, "section": section,
            "headings": [html.unescape(t) for _, _, t in toc],
            "text": html.unescape(plain)[:1600],
        })
        built += 1

    idx_json = json.dumps(search_index, ensure_ascii=False)
    (OUT / "assets" / "search-index.js").write_text(
        "window.SIPLLM_SEARCH = " + idx_json + ";\n", encoding="utf-8")
    print(f"Done. {built}/{len(FLAT)} pages built into {OUT} ({missing} pending).")


if __name__ == "__main__":
    build()
