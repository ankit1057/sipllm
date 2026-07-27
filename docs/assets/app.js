// SipLLM docs portal — dependency-free progressive enhancement. Works offline.
(function () {
  "use strict";
  var body = document.body;

  // ---- mobile sidebar ----
  var toggle = document.querySelector(".menu-toggle");
  var scrim = document.getElementById("sidebar-scrim");
  function closeNav() { body.classList.remove("nav-open"); if (toggle) toggle.setAttribute("aria-expanded", "false"); }
  if (toggle) toggle.addEventListener("click", function () {
    var open = body.classList.toggle("nav-open");
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  });
  if (scrim) scrim.addEventListener("click", closeNav);

  // ---- active TOC highlight ----
  var tocLinks = [].slice.call(document.querySelectorAll(".toc a"));
  if (tocLinks.length && "IntersectionObserver" in window) {
    var byId = {}, current = null;
    tocLinks.forEach(function (a) { byId[a.getAttribute("href").slice(1)] = a; });
    var heads = tocLinks.map(function (a) { return document.getElementById(a.getAttribute("href").slice(1)); }).filter(Boolean);
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          if (current) current.classList.remove("active");
          var link = byId[e.target.id];
          if (link) { link.classList.add("active"); current = link; }
        }
      });
    }, { rootMargin: "-80px 0px -70% 0px", threshold: 0 });
    heads.forEach(function (h) { obs.observe(h); });
  }

  // ---- search ----
  var input = document.getElementById("search-input");
  var panel = document.getElementById("search-results");
  var docs = window.SIPLLM_SEARCH || [];
  var results = [], sel = -1;

  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function hl(text, q) {
    if (!q) return esc(text);
    var i = text.toLowerCase().indexOf(q.toLowerCase());
    if (i < 0) return esc(text);
    return esc(text.slice(0, i)) + "<mark>" + esc(text.slice(i, i + q.length)) + "</mark>" + esc(text.slice(i + q.length));
  }
  function snippet(text, q) {
    var i = text.toLowerCase().indexOf(q.toLowerCase());
    if (i < 0) return text.slice(0, 150);
    var s = Math.max(0, i - 40);
    return (s > 0 ? "…" : "") + text.slice(s, s + 150);
  }
  function score(d, q) {
    var t = d.title.toLowerCase(), heads = (d.headings || []).join(" ").toLowerCase(), body = (d.text || "").toLowerCase();
    var s = 0;
    if (t === q) s += 100;
    if (t.indexOf(q) === 0) s += 40;
    if (t.indexOf(q) >= 0) s += 25;
    if (heads.indexOf(q) >= 0) s += 12;
    if (body.indexOf(q) >= 0) s += 5;
    return s;
  }
  function search(q) {
    q = q.trim().toLowerCase();
    if (q.length < 2) return [];
    return docs.map(function (d) { return { d: d, s: score(d, q) }; })
      .filter(function (r) { return r.s > 0; })
      .sort(function (a, b) { return b.s - a.s; })
      .slice(0, 12).map(function (r) { return r.d; });
  }
  function render(q) {
    results = search(q); sel = -1;
    if (!q || q.trim().length < 2) { panel.hidden = true; panel.innerHTML = ""; return; }
    if (!results.length) { panel.hidden = false; panel.innerHTML = '<div class="sr-empty">No matches for “' + esc(q) + '”.</div>'; return; }
    panel.innerHTML = results.map(function (d) {
      return '<a href="' + d.slug + '.html">' +
        '<div class="sr-title">' + hl(d.title, q) + "</div>" +
        '<div class="sr-meta">' + esc(d.section) + "</div>" +
        '<div class="sr-snip">' + esc(snippet(d.text || "", q)) + "</div></a>";
    }).join("");
    panel.hidden = false;
  }
  function move(delta) {
    var items = panel.querySelectorAll("a");
    if (!items.length) return;
    if (sel >= 0) items[sel].classList.remove("active");
    sel = (sel + delta + items.length) % items.length;
    items[sel].classList.add("active");
    items[sel].scrollIntoView({ block: "nearest" });
  }
  if (input && panel) {
    input.addEventListener("input", function () { render(input.value); });
    input.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (e.key === "Enter") {
        var items = panel.querySelectorAll("a");
        if (sel >= 0 && items[sel]) location.href = items[sel].getAttribute("href");
        else if (items[0]) location.href = items[0].getAttribute("href");
      } else if (e.key === "Escape") { panel.hidden = true; input.blur(); }
    });
    document.addEventListener("click", function (e) { if (!e.target.closest(".search")) panel.hidden = true; });
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && document.activeElement !== input) { e.preventDefault(); input.focus(); }
      if (e.key === "Escape") closeNav();
    });
  }
})();
