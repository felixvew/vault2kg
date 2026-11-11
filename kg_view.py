#!/usr/bin/env python3
"""kg_view - turn graph.json into an interactive page (graph.html).

    python kg_view.py ~/Documents/MyVault-kg            # writes MyVault-kg/graph.html
    python kg_view.py ~/Documents/MyVault-kg --open     # and opens it

One self-contained HTML file that works offline: search, filter by kind,
click a node for its connections, open notes in Obsidian. Uses its built-in
style, or your own stylesheet if a proto-ui.css sits next to this script.
No dependencies.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import webbrowser
from pathlib import Path


def load_graph(path: str) -> tuple[dict, Path]:
    """Accept the -kg folder or graph.json itself."""
    p = Path(path).expanduser()
    if p.is_dir():
        p = p / "graph.json"
    if not p.exists():
        sys.exit(f"Not found: {p}\nRun vault2kg.py on your vault first.")
    graph = json.loads(p.read_text(encoding="utf-8"))
    if graph.get("format") != "vault2kg-graph":
        sys.exit(f"{p} is not a vault2kg graph.json")
    return graph, p.parent


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def export_html(graph: dict, path: Path) -> None:
    meta = {"vault": graph.get("vault", "Vault"), "llm": graph.get("llm")}
    index = {n["id"]: i for i, n in enumerate(graph["nodes"])}
    data = {
        "vault": meta["vault"], "llm": meta.get("llm"), "stats": graph["stats"],
        "nodes": [{"id": n["id"], "name": n["name"], "kind": n["kind"], "type": n["type"],
                   "d": truncate(n["description"], 400), "a": n["aliases"][:8], "n": n["notes"][:40],
                   "p": n.get("path"), "r": n.get("resolved", True)} for n in graph["nodes"]],
        "edges": [[index[e["source"]], index[e["target"]], e["type"], int(e["origins"] == ["llm"])]
                  for e in graph["edges"]],
    }
    css_file = Path(__file__).with_name("proto-ui.css")
    css = css_file.read_text(encoding="utf-8") if css_file.exists() else FALLBACK_CSS
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    page = (HTML_TEMPLATE.replace("{{CSS}}", css).replace("{{DATA}}", blob)
            .replace("{{TITLE}}", html.escape(meta["vault"])))
    write_text(path, page)


FALLBACK_CSS = """:root{--u:7px;--bg:#f0f4f6;--panel:#fff;--fill:#e6e9eb;--text:#000;--muted:#a0a0a0;
--black:#000;--white:#fff;--grey-1:#f0f4f6;--grey-2:#e6e9eb;--grey-3:#a0a0a0;--green:#63f450;--violet:#6e32e1;
--selected:#63f450;--note:#6e32e1;--focus:#3b75e2;--font-sans:"Hanken Grotesk",Helvetica,Arial,sans-serif;
--font-mono:"DM Mono",ui-monospace,monospace;--fs-title:44px;--fs-xl:25px;--fs-l:20px;--fs-m:17px;--fs-s:15px;
--fs-xs:12px;--sp-1:3px;--sp-2:6px;--sp-3:8px;--sp-4:14px;--sp-5:18px;--sp-6:30px;--r-pill:999px;--r-panel:10px;--w-wide:1600px}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:var(--fs-s)/1.1 var(--font-sans)}
.header{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;padding:var(--sp-4)}
.header__start{justify-self:start;display:flex;gap:var(--sp-2)}.header__end{justify-self:end;display:flex;gap:var(--sp-2)}
.logo{font-size:var(--fs-l);text-transform:uppercase;text-decoration:none;color:inherit}
.page{padding:0 var(--sp-4) var(--sp-6)}.container{max-width:var(--w-wide);margin:0 auto}
.hero{text-align:center;padding:var(--sp-5) 0}.title{font-size:var(--fs-title);text-transform:uppercase;font-weight:400;margin:0}
.intro{margin:var(--sp-3) auto 0;max-width:700px}.panel{background:var(--panel);border-radius:var(--r-panel);padding:var(--sp-5)}
.stack{display:grid;gap:var(--sp-3);align-content:start}.label{font-family:var(--font-mono);font-size:var(--fs-xs)}
.muted{color:var(--muted)}.panel-title{font-size:var(--fs-s);text-transform:uppercase;text-align:center;font-weight:400;margin:0}
.btn{font:inherit;border:0;border-radius:var(--r-pill);background:var(--fill);padding:6px 12px;cursor:pointer;color:inherit;text-decoration:none}
.btn--primary{background:#000;color:#fff}.options{display:flex;flex-wrap:wrap;gap:var(--sp-2)}
.option{display:inline-flex;gap:6px;align-items:center;background:var(--fill);border-radius:var(--r-pill);padding:6px 12px;cursor:pointer}
.input{font:inherit;width:100%;border:0;background:var(--grey-1);border-radius:var(--r-pill);padding:10px 14px}
.tag{font-family:var(--font-mono);font-size:var(--fs-xs);border:1px solid #000;border-radius:var(--r-pill);padding:2px 8px}
.prose{line-height:1.3}"""


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}} · Vault graph</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=DM+Mono&family=Hanken+Grotesk&display=swap">
<style>{{CSS}}</style>
<style>
  .hero { padding-block: var(--sp-5); }
  .graph-layout { display: grid; grid-template-columns: minmax(0, 1fr) calc(var(--u) * 50); gap: var(--sp-3); align-items: start; }
  .graph-panel { position: relative; padding: 0; overflow: hidden; height: calc(100vh - var(--u) * 34); min-height: calc(var(--u) * 60); }
  .graph-panel canvas { display: block; width: 100%; height: 100%; touch-action: none; cursor: grab; }
  .graph-panel canvas.is-dragging { cursor: grabbing; }
  .graph-hint { position: absolute; left: var(--sp-4); bottom: var(--sp-3); pointer-events: none; }
  .side { display: grid; gap: var(--sp-3); align-content: start; max-height: calc(100vh - var(--u) * 34); overflow: auto; }
  .plain-list { list-style: none; margin: 0; padding: 0; display: grid; gap: var(--sp-1); }
  .plain-list .btn { width: 100%; justify-content: flex-start; text-align: left; }
  .row-meta { margin-left: auto; padding-left: var(--sp-2); }
  .legend { display: flex; flex-wrap: wrap; gap: var(--sp-2) var(--sp-3); align-items: center; }
  .dot { display: inline-block; width: calc(var(--u) * 1.4); height: calc(var(--u) * 1.4); border-radius: 50%; vertical-align: middle; margin-right: var(--sp-1); }
  .dot--note { background: var(--black); } .dot--entity { background: var(--violet); }
  .dot--tag { background: var(--grey-3); } .dot--missing { border: 3px solid var(--black); }
  .details-name { font-size: var(--fs-m); overflow-wrap: anywhere; }
  .conn-type { flex: none; }
  @media (max-width: 900px) {
    .graph-layout { grid-template-columns: 1fr; }
    .graph-panel { height: 65vh; }
    .side { max-height: none; }
  }
</style>
</head>
<body>
<header class="header">
  <div class="header__start"><span class="label hide-sm" id="counts"></span></div>
  <a class="logo" href="#">Vault graph 🕸️</a>
  <div class="header__end">
    <button class="btn" type="button" id="fit">Fit</button>
    <button class="btn btn--primary" type="button" id="relayout">Re-layout</button>
  </div>
</header>
<main class="page">
  <section class="hero">
    <h1 class="title" id="vault-title"></h1>
    <p class="intro" id="intro"></p>
  </section>
  <div class="container graph-layout">
    <div class="panel graph-panel">
      <canvas id="graph" aria-label="Knowledge graph"></canvas>
      <p class="label muted graph-hint">Scroll to zoom · drag to move · click a node for details</p>
    </div>
    <aside class="side">
      <div class="panel stack">
        <label class="visually-hidden" for="q">Search nodes</label>
        <input class="input input--round" id="q" type="search" placeholder="Search notes and entities" autocomplete="off">
        <ul class="plain-list" id="results"></ul>
      </div>
      <div class="panel stack">
        <div class="legend label">
          <span><span class="dot dot--note"></span>note</span>
          <span><span class="dot dot--missing"></span>linked, not created yet</span>
          <span><span class="dot dot--entity"></span>entity</span>
          <span><span class="dot dot--tag"></span>tag</span>
        </div>
        <p class="label">Show</p>
        <div class="options" id="kinds"></div>
        <p class="label">Relationships</p>
        <div class="options" id="rels"></div>
      </div>
      <div class="panel stack" id="details">
        <p class="label">Selection</p>
        <p class="muted">Click a node to see what it is and how it connects.</p>
      </div>
    </aside>
  </div>
</main>
<script id="graph-data" type="application/json">{{DATA}}</script>
<script>
(() => {
  const D = JSON.parse(document.getElementById('graph-data').textContent);
  const css = getComputedStyle(document.documentElement);
  const v = (name, fb) => (css.getPropertyValue(name) || '').trim() || fb;
  const COLOR = { note: v('--black', '#000'), entity: v('--violet', '#6e32e1'), tag: v('--grey-3', '#a0a0a0'),
                  file: v('--grey-3', '#a0a0a0'), selected: v('--green', '#63f450'), text: v('--black', '#000') };
  const FONT = v('--font-sans', 'sans-serif');
  const KIND_LABEL = { note: 'Notes', entity: 'Entities', tag: 'Tags', file: 'Files' };

  // ---------- data ----------
  const N = D.nodes.map((n, i) => ({ ...n, i, x: 0, y: 0, vx: 0, vy: 0, deg: 0, fixed: false }));
  const E = D.edges.map(e => ({ s: N[e[0]], t: N[e[1]], type: e[2], llm: e[3] === 1 }));
  const adj = N.map(() => []);
  E.forEach((e, k) => { e.s.deg++; e.t.deg++; adj[e.s.i].push(k); adj[e.t.i].push(k); });
  const byId = new Map(N.map(n => [n.id, n]));
  const counts = { note: 0, entity: 0, tag: 0, file: 0 };
  N.forEach(n => counts[n.kind]++);

  document.getElementById('vault-title').textContent = D.vault;
  document.getElementById('counts').textContent =
    `${D.stats.notes} notes · ${counts.entity} entities · ${E.length} relationships`;
  document.getElementById('intro').textContent = D.llm
    ? `Links, frontmatter and tags from your notes, plus entities extracted by ${D.llm}.`
    : 'Links, frontmatter and tags from your notes. Run with an LLM to add extracted entities.';

  // ---------- filters ----------
  const show = { note: true, entity: true, tag: counts.note + counts.entity < 3000, file: false };
  const showRel = { structural: true, llm: true };
  const visible = n => show[n.kind];
  const edgeVisible = e => visible(e.s) && visible(e.t) && (e.llm ? showRel.llm : showRel.structural);
  const kindInputs = {};
  function option(container, label, checked, onChange) {
    const l = document.createElement('label');
    l.className = 'option';
    const input = document.createElement('input');
    input.type = 'checkbox'; input.checked = checked;
    input.addEventListener('change', () => { onChange(input.checked); kick(0.5); });
    l.append(input, ' ' + label);
    container.append(l);
    return input;
  }
  const kindsEl = document.getElementById('kinds');
  for (const k of ['note', 'entity', 'tag', 'file']) {
    if (!counts[k]) continue;
    kindInputs[k] = option(kindsEl, `${KIND_LABEL[k]} ${counts[k]}`, show[k], c => { show[k] = c; });
  }
  const relsEl = document.getElementById('rels');
  option(relsEl, 'Links, properties, tags', true, c => { showRel.structural = c; });
  if (E.some(e => e.llm)) option(relsEl, 'Extracted by LLM', true, c => { showRel.llm = c; });

  // ---------- layout (force simulation with a coarse grid for repulsion) ----------
  N.forEach((n, i) => { const a = i * 2.39996, r = 14 * Math.sqrt(i + 1); n.x = Math.cos(a) * r; n.y = Math.sin(a) * r; });
  let alpha = 1, running = true;
  const CELL = 120;
  function tick() {
    const vis = N.filter(visible);
    const grid = new Map();
    for (const n of vis) {
      const key = Math.floor(n.x / CELL) + ',' + Math.floor(n.y / CELL);
      let cell = grid.get(key);
      if (!cell) { cell = { nodes: [], x: 0, y: 0, cx: Math.floor(n.x / CELL), cy: Math.floor(n.y / CELL) }; grid.set(key, cell); }
      cell.nodes.push(n); cell.x += n.x; cell.y += n.y;
    }
    const cells = [...grid.values()];
    cells.forEach(c => { c.x /= c.nodes.length; c.y /= c.nodes.length; });
    const strength = 2600 * alpha;
    for (const c of cells) {
      for (const d of cells) {
        const near = Math.abs(c.cx - d.cx) <= 1 && Math.abs(c.cy - d.cy) <= 1;
        if (near) {
          if (d.cx < c.cx || (d.cx === c.cx && d.cy < c.cy)) continue;
          const same = c === d;
          for (let i = 0; i < c.nodes.length; i++) {
            const a = c.nodes[i];
            const list = same ? c.nodes : d.nodes;
            for (let j = same ? i + 1 : 0; j < list.length; j++) {
              const b = list[j];
              let dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy;
              if (d2 < 0.01) { dx = Math.random() - 0.5; dy = Math.random() - 0.5; d2 = 0.5; }
              const f = strength / d2, dist = Math.sqrt(d2);
              dx /= dist; dy /= dist;
              a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
            }
          }
        } else {
          // far cells push each other as single masses
          let dx = c.x - d.x, dy = c.y - d.y; const d2 = dx * dx + dy * dy || 1, dist = Math.sqrt(d2);
          const f = strength * d.nodes.length / d2;
          dx /= dist; dy /= dist;
          for (const a of c.nodes) { a.vx += dx * f; a.vy += dy * f; }
        }
      }
    }
    for (const e of E) {
      if (!edgeVisible(e)) continue;
      const dx = e.t.x - e.s.x, dy = e.t.y - e.s.y, dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const target = e.s.kind === 'tag' || e.t.kind === 'tag' ? 110 : 70;
      const f = (dist - target) * 0.04 * alpha / dist;
      e.s.vx += dx * f; e.s.vy += dy * f; e.t.vx -= dx * f; e.t.vy -= dy * f;
    }
    for (const n of vis) {
      n.vx -= n.x * 0.003 * alpha; n.vy -= n.y * 0.003 * alpha;
      n.vx *= 0.55; n.vy *= 0.55;
      const sp = Math.hypot(n.vx, n.vy); if (sp > 40) { n.vx *= 40 / sp; n.vy *= 40 / sp; }
      if (!n.fixed) { n.x += n.vx; n.y += n.vy; }
    }
    alpha *= 0.985;
    if (alpha < 0.02) running = false;
  }
  function kick(a) { alpha = Math.max(alpha, a); if (!running) { running = true; requestAnimationFrame(loop); } else draw(); }

  // ---------- drawing ----------
  const canvas = document.getElementById('graph');
  const ctx = canvas.getContext('2d');
  let W = 0, H = 0, dpr = 1;
  const view = { k: 1, x: 0, y: 0 };
  let selected = null, hover = null, neighbours = new Set(), fitted = false;
  const radius = n => Math.min(18, (n.kind === 'note' ? 3 : 2.5) + Math.sqrt(n.deg) * 1.3);
  function resize() {
    const r = canvas.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1; W = r.width; H = r.height;
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    draw();
  }
  new ResizeObserver(resize).observe(canvas);
  const toScreen = n => [n.x * view.k + view.x + W / 2, n.y * view.k + view.y + H / 2];
  const toWorld = (sx, sy) => [(sx - W / 2 - view.x) / view.k, (sy - H / 2 - view.y) / view.k];
  function fit() {
    const vis = N.filter(visible);
    if (!vis.length || !W) return;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const n of vis) { x0 = Math.min(x0, n.x); y0 = Math.min(y0, n.y); x1 = Math.max(x1, n.x); y1 = Math.max(y1, n.y); }
    const k = Math.min(4, 0.9 * Math.min(W / (x1 - x0 + 40), H / (y1 - y0 + 40)));
    view.k = k; view.x = -((x0 + x1) / 2) * k; view.y = -((y0 + y1) / 2) * k;
    draw();
  }
  function draw() {
    if (!W) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(W / 2 + view.x, H / 2 + view.y);
    ctx.scale(view.k, view.k);
    const focus = selected !== null;
    ctx.lineWidth = 1 / view.k;
    for (const e of E) {
      if (!edgeVisible(e)) continue;
      const hot = focus && (e.s === selected || e.t === selected);
      ctx.strokeStyle = hot ? 'rgba(0,0,0,.75)' : focus ? 'rgba(0,0,0,.04)' : (e.llm ? 'rgba(110,50,225,.22)' : 'rgba(0,0,0,.13)');
      ctx.beginPath(); ctx.moveTo(e.s.x, e.s.y); ctx.lineTo(e.t.x, e.t.y); ctx.stroke();
    }
    for (const n of N) {
      if (!visible(n)) continue;
      const dim = focus && n !== selected && !neighbours.has(n);
      ctx.globalAlpha = dim ? 0.15 : 1;
      const r = radius(n);
      ctx.beginPath(); ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      if (n.kind === 'file') { ctx.strokeStyle = COLOR.file; ctx.lineWidth = 1.5 / view.k; ctx.stroke(); }
      else { ctx.fillStyle = COLOR[n.kind]; ctx.fill(); }
      if (n.kind === 'note' && n.r === false) { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(n.x, n.y, r * 0.5, 0, Math.PI * 2); ctx.fill(); }
      if (n === selected || n === hover) {
        ctx.lineWidth = 3 / view.k; ctx.strokeStyle = n === selected ? COLOR.selected : COLOR.text;
        ctx.beginPath(); ctx.arc(n.x, n.y, r + 2.5 / view.k, 0, Math.PI * 2); ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    ctx.restore();
    // labels in screen space
    ctx.font = `12px ${FONT}`;
    ctx.textBaseline = 'middle';
    const labelled = [];
    for (const n of N) {
      if (!visible(n)) continue;
      const r = radius(n) * view.k;
      const important = n === selected || n === hover || neighbours.has(n);
      if (important || (!focus && (r > 9 || view.k > 2.2))) labelled.push(n);
    }
    for (const n of labelled.slice(0, 400)) {
      const [sx, sy] = toScreen(n);
      if (sx < -100 || sy < -20 || sx > W + 100 || sy > H + 20) continue;
      const text = n.kind === 'tag' ? '#' + n.name : n.name;
      const x = sx + radius(n) * view.k + 4;
      ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(255,255,255,.9)'; ctx.strokeText(text, x, sy);
      ctx.fillStyle = COLOR.text; ctx.fillText(text, x, sy);
    }
  }
  function loop() {
    if (!running) return;
    for (let i = 0; i < 2; i++) tick();
    if (!fitted && alpha < 0.35) { fit(); fitted = true; }
    draw();
    requestAnimationFrame(loop);
  }
  requestAnimationFrame(loop);

  // ---------- interaction ----------
  function nodeAt(sx, sy) {
    const [wx, wy] = toWorld(sx, sy);
    let best = null, bestD = Infinity;
    for (const n of N) {
      if (!visible(n)) continue;
      const d = Math.hypot(n.x - wx, n.y - wy), r = radius(n) + 4 / view.k;
      if (d < r && d < bestD) { best = n; bestD = d; }
    }
    return best;
  }
  let drag = null;
  const touches = new Map();  // pointerId -> [x, y], for two-finger pinch zoom
  let pinch = null;
  function zoomAt(sx, sy, k) {
    const [wx, wy] = toWorld(sx, sy);
    view.k = Math.max(0.05, Math.min(12, k));
    view.x = sx - W / 2 - wx * view.k; view.y = sy - H / 2 - wy * view.k;
    draw();
  }
  canvas.addEventListener('pointerdown', ev => {
    const r = canvas.getBoundingClientRect(), sx = ev.clientX - r.left, sy = ev.clientY - r.top;
    touches.set(ev.pointerId, [sx, sy]);
    if (touches.size === 2) {
      const [a, b] = [...touches.values()];
      pinch = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), k: view.k };
      if (drag && drag.n) drag.n.fixed = false;
      drag = null;
      return;
    }
    const n = nodeAt(sx, sy);
    drag = { n, sx, sy, vx: view.x, vy: view.y, moved: false };
    if (n) n.fixed = true;
    canvas.setPointerCapture(ev.pointerId);
    canvas.classList.add('is-dragging');
  });
  canvas.addEventListener('pointermove', ev => {
    const r = canvas.getBoundingClientRect(), sx = ev.clientX - r.left, sy = ev.clientY - r.top;
    if (touches.has(ev.pointerId)) touches.set(ev.pointerId, [sx, sy]);
    if (pinch && touches.size === 2) {
      const [a, b] = [...touches.values()];
      zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, pinch.k * Math.hypot(a[0] - b[0], a[1] - b[1]) / pinch.d);
      return;
    }
    if (!drag) {
      const h = nodeAt(sx, sy);
      if (h !== hover) { hover = h; canvas.style.cursor = h ? 'pointer' : ''; draw(); }
      return;
    }
    if (Math.hypot(sx - drag.sx, sy - drag.sy) > 3) drag.moved = true;
    if (drag.n) { const [wx, wy] = toWorld(sx, sy); drag.n.x = wx; drag.n.y = wy; kick(0.15); }
    else { view.x = drag.vx + sx - drag.sx; view.y = drag.vy + sy - drag.sy; draw(); }
  });
  const release = ev => {
    touches.delete(ev.pointerId);
    if (touches.size < 2) pinch = null;
  };
  canvas.addEventListener('pointercancel', ev => { release(ev); drag = null; canvas.classList.remove('is-dragging'); });
  canvas.addEventListener('pointerup', ev => {
    release(ev);
    if (!drag) return;
    if (drag.n) drag.n.fixed = false;
    if (!drag.moved) select(drag.n);
    drag = null;
    canvas.classList.remove('is-dragging');
  });
  canvas.addEventListener('wheel', ev => {
    ev.preventDefault();
    const r = canvas.getBoundingClientRect(), sx = ev.clientX - r.left, sy = ev.clientY - r.top;
    zoomAt(sx, sy, view.k * Math.exp(-ev.deltaY * 0.0015));
  }, { passive: false });
  canvas.addEventListener('dblclick', ev => {
    const r = canvas.getBoundingClientRect();
    const n = nodeAt(ev.clientX - r.left, ev.clientY - r.top);
    if (n && n.p) window.location.href = obsidianUrl(n.p);
  });
  document.getElementById('fit').addEventListener('click', fit);
  document.getElementById('relayout').addEventListener('click', () => {
    N.forEach((n, i) => { const a = i * 2.39996, r = 14 * Math.sqrt(i + 1); n.x = Math.cos(a) * r; n.y = Math.sin(a) * r; });
    fitted = false; kick(1);
  });

  // ---------- details panel ----------
  const obsidianUrl = p => `obsidian://open?vault=${encodeURIComponent(D.vault)}&file=${encodeURIComponent(p.replace(/\.md$/, ''))}`;
  const el = (tag, cls, text) => { const x = document.createElement(tag); if (cls) x.className = cls; if (text != null) x.textContent = text; return x; };
  function nodeButton(n, meta) {
    const b = el('button', 'btn');
    b.type = 'button';
    b.append(el('span', '', n.kind === 'tag' ? '#' + n.name : n.name));
    if (meta) b.append(el('span', 'label muted row-meta conn-type', meta));
    b.addEventListener('click', () => {
      if (!visible(n)) { show[n.kind] = true; if (kindInputs[n.kind]) kindInputs[n.kind].checked = true; }
      select(n); centre(n);
    });
    return b;
  }
  function centre(n) { view.x = -n.x * view.k; view.y = -n.y * view.k; draw(); }
  function select(n) {
    selected = n || null;
    neighbours = new Set();
    const box = document.getElementById('details');
    box.replaceChildren(el('p', 'label', 'Selection'));
    if (!n) { box.append(el('p', 'muted', 'Click a node to see what it is and how it connects.')); draw(); return; }
    adj[n.i].forEach(k => { const e = E[k]; neighbours.add(e.s === n ? e.t : e.s); });
    box.append(el('p', 'details-name', n.kind === 'tag' ? '#' + n.name : n.name));
    const tags = el('div', 'cluster');
    tags.append(el('span', 'tag', KIND_LABEL[n.kind].replace(/s$/, '')));
    if (n.type && !['Note', 'Tag', 'Attachment', 'Other'].includes(n.type)) tags.append(el('span', 'tag', n.type));
    if (n.kind === 'note' && n.r === false) tags.append(el('span', 'tag', 'not created yet'));
    box.append(tags);
    if (n.d) box.append(el('p', 'prose', n.d));
    if (n.a && n.a.length) box.append(el('p', 'muted', 'Also called: ' + n.a.join(', ')));
    if (n.p && n.r !== false) {
      const a = el('a', 'btn btn--primary', 'Open in Obsidian');
      a.href = obsidianUrl(n.p);
      box.append(el('div', 'cluster')); box.lastChild.append(a);
    }
    const conns = adj[n.i].map(k => E[k]).filter(edgeVisible);
    if (conns.length) {
      box.append(el('p', 'label', `Connections ${conns.length}`));
      const ul = el('ul', 'plain-list');
      for (const e of conns.slice(0, 80)) {
        const out = e.s === n, other = out ? e.t : e.s;
        const li = el('li');
        li.append(nodeButton(other, (out ? '→ ' : '← ') + e.type.toLowerCase().replace(/_/g, ' ')));
        ul.append(li);
      }
      box.append(ul);
    }
    const others = (n.n || []).filter(p => p !== n.p);
    if (others.length && n.kind !== 'note') {
      box.append(el('p', 'label', `Appears in ${others.length} notes`));
      const ul = el('ul', 'plain-list');
      for (const p of others.slice(0, 30)) {
        const target = byId.get('note:' + p.replace(/\.md$/, ''));
        const li = el('li');
        if (target) li.append(nodeButton(target)); else li.append(el('span', '', p));
        ul.append(li);
      }
      box.append(ul);
    }
    draw();
  }

  // ---------- search ----------
  const q = document.getElementById('q'), results = document.getElementById('results');
  function search() {
    const s = q.value.trim().toLowerCase();
    results.replaceChildren();
    if (!s) return [];
    const hits = N.filter(n => n.name.toLowerCase().includes(s) || (n.a || []).some(a => a.toLowerCase().includes(s)))
      .sort((a, b) => (a.name.toLowerCase().startsWith(s) ? 0 : 1) - (b.name.toLowerCase().startsWith(s) ? 0 : 1) || b.deg - a.deg)
      .slice(0, 12);
    if (!hits.length) results.append(el('li', 'muted', 'Nothing found.'));
    for (const n of hits) {
      const li = el('li');
      li.append(nodeButton(n, KIND_LABEL[n.kind].replace(/s$/, '').toLowerCase()));
      results.append(li);
    }
    return hits;
  }
  q.addEventListener('input', search);
  q.addEventListener('keydown', ev => {
    if (ev.key === 'Enter') { const h = search(); if (h[0]) results.querySelector('button').click(); }
    if (ev.key === 'Escape') { q.value = ''; search(); select(null); }
  });
})();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="kg_view", description="graph.json -> interactive graph.html")
    p.add_argument("graph", help="the -kg folder written by vault2kg.py (or its graph.json)")
    p.add_argument("-o", "--out", type=Path, help="output file (default: graph.html next to graph.json)")
    p.add_argument("--open", action="store_true", help="open the page in your browser")
    a = p.parse_args(argv)
    graph, folder = load_graph(a.graph)
    out = a.out or folder / "graph.html"
    export_html(graph, out)
    print(f"Wrote {out}")
    if a.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
