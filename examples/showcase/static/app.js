"use strict";

const $ = (id) => document.getElementById(id);
const SVGNS = "http://www.w3.org/2000/svg";
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};
const svg = (tag, attrs) => {
  const node = document.createElementNS(SVGNS, tag);
  if (attrs) for (const k in attrs) node.setAttribute(k, attrs[k]);
  return node;
};

let mode = "rag";
let running = false;
let isOnline = false;
let genModel = "";

// Client copy of the route explanations, so the how-it-works skeleton can caption
// stages the current query did not fire. Fired stages carry their own explanation
// from the server, so nothing shown about the run itself is invented here.
const EXPLAIN = {
  "cag:exact": ["cache exact hit", "identical prior query served from the answer cache, no model call"],
  "cag:semantic": ["cache semantic hit", "semantically equivalent prior query served from the answer cache"],
  transform: ["query transform", "the query is rewritten or expanded before retrieval"],
  semantic: ["semantic search", "dense vector similarity over the stored chunk embeddings"],
  lexical: ["lexical search", "sparse keyword and BM25 match over the chunk text"],
  filter: ["metadata filter", "a hard metadata constraint applied before ranking"],
  graph: ["graph facts", "the co-occurrence entity graph contributed multi-hop facts"],
  rrf: ["reciprocal rank fusion", "the dense and sparse candidate lists fused by reciprocal rank"],
  expansion: ["query expansion", "extra query variants retrieved and merged in"],
  hype: ["HyPE", "hypothetical prompt embeddings matched and mapped back to real chunks"],
  raptor: ["RAPTOR", "the recursive summary tree matched and mapped back to leaf chunks"],
  rerank: ["rerank", "the fused candidates re-scored by the reranker stage"],
  compress: ["compression", "redundant context compressed before packing"],
  expand: ["hierarchical expand", "parent or neighbour chunks pulled in for context"],
  autocut: ["autocut", "the low-similarity tail dropped from the candidate list"],
  generate: ["generate", "the grounded answer is produced from the packed evidence"],
  cite: ["cite", "each claim is bound to a real stored chunk id"],
  abstain: ["abstain", "the best evidence was below the floor, so an honest no-answer was returned"],
};

// The educational pipeline skeleton, in the order stages run. cite and generate are
// always-on terminal stages of the grounded pipeline; the rest light up from the
// real routes the runtime reports for the query.
const PIPELINE = [
  "transform",
  "semantic",
  "lexical",
  "filter",
  "graph",
  "rrf",
  "expansion",
  "hype",
  "raptor",
  "rerank",
  "compress",
  "expand",
  "autocut",
  "generate",
  "cite",
];

const NODE_COLORS = ["#39ff9e", "#38e6ff", "#b78bff", "#ffcc57", "#ff5c7a", "#7ad0ff", "#9dffd0"];

const session = {
  queries: 0,
  hits: 0,
  totalMs: 0,
  savedMs: 0,
  savedTokens: 0,
  savingsSamples: 0,
};
const lastMiss = new Map();

async function boot() {
  const [featRes, statsRes] = await Promise.all([fetch("/api/features"), fetch("/api/stats")]);
  const data = await featRes.json();
  const stats = await statsRes.json();
  isOnline = !!data.online;
  genModel = data.generation_model || "";

  const badge = $("mode-badge");
  if (data.online) {
    badge.textContent = "Online, " + data.generation_model + " and " + data.embedding_model;
    badge.classList.add("mode-openai");
  } else {
    badge.textContent = "Offline, deterministic";
  }
  renderModeBanner(data);

  const modes = $("modes");
  data.modes.forEach((name, i) => {
    const b = el("button", i === 0 ? "active" : "", name);
    b.addEventListener("click", () => {
      mode = name;
      [...modes.children].forEach((c) => c.classList.remove("active"));
      b.classList.add("active");
    });
    modes.appendChild(b);
  });

  const feats = $("features");
  feats.innerHTML = "";
  data.features.forEach((f) => {
    const row = el("div", "feat-row");
    row.appendChild(el("span", "", f.name));
    const tag = el("span", f.model_lift ? "lift" : "base", f.model_lift ? "uses the model" : "no model needed");
    tag.title = f.model_lift
      ? "Quality comes from the language model. Active now, this server is online."
      : "Pure algorithm, no AI model. Same result every time, runs fully offline.";
    row.appendChild(tag);
    feats.appendChild(row);
  });

  renderCorpusStats(stats);
  renderPipeline(new Set(), null);
  loadCorpus();
  wireObservability();
  wireTts();
  document.querySelectorAll(".prompt-hint code").forEach((c) => {
    c.addEventListener("click", () => {
      $("q").value = c.getAttribute("data-q");
    });
  });
  $("run").addEventListener("click", run);
  $("q").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run();
  });
}

function renderModeBanner(data) {
  const box = $("mode-banner");
  if (!box) return;
  box.innerHTML = "";
  const online = !!data.online;
  box.className = "mode-banner " + (online ? "online" : "offline");
  box.appendChild(el("span", "mode-dot"));
  const label = online
    ? "Online, " + data.generation_model + " and " + data.embedding_model
    : "Offline, deterministic";
  box.appendChild(el("span", "mode-label", label));
  const sub = online
    ? "generation and embeddings served by a real model"
    : "hash embeddings, extractive generation, co-occurrence graph, no model calls";
  box.appendChild(el("span", "mode-sub", sub));
}

function statCard(box, key, value, cls) {
  const card = el("div", "stat" + (cls && cls.wide ? " wide" : ""));
  card.appendChild(el("span", "k", key));
  card.appendChild(el("span", "v" + (cls && cls.tone ? " " + cls.tone : ""), String(value)));
  box.appendChild(card);
}

function renderCorpusStats(stats) {
  const box = $("corpus-stats");
  box.innerHTML = "";
  statCard(box, "documents", stats.documents, { tone: "good" });
  statCard(box, "chunks", stats.chunks, { tone: "good" });
  statCard(box, "entities", stats.entities, { tone: "cyan" });
  statCard(box, "relations", stats.relations, { tone: "cyan" });
  const c = stats.capability_counts || {};
  statCard(box, "capabilities", c.total || 0, {});
  statCard(box, "uses the model", c.model_lift || 0, {});
}

async function loadCorpus() {
  const res = await fetch("/api/corpus");
  const data = await res.json();
  const box = $("corpus");
  box.innerHTML = "";
  const head = el("div", "doc-row");
  head.appendChild(el("span", "", data.document_count + " documents"));
  head.appendChild(el("span", "tag", data.chunk_count + " chunks"));
  box.appendChild(head);
  data.documents.forEach((d) => {
    const row = el("div", "doc-row");
    row.appendChild(el("span", "", d.title));
    row.appendChild(el("span", "tag", d.category || d.source_id));
    box.appendChild(row);
  });
}

function setBusy(state) {
  running = state;
  if (state) stopSpeaking();
  const btn = $("run");
  btn.disabled = state;
  btn.textContent = state ? "running..." : "run >_";
}

function renderCitations(citations) {
  const box = $("citations");
  box.innerHTML = "";
  if (!citations || citations.length === 0) {
    box.appendChild(el("span", "empty", "no citations"));
    return;
  }
  citations.forEach((c) => {
    const card = el("div", "cite");
    const head = el("div", "cid");
    head.appendChild(el("span", "", c.chunk_id));
    head.appendChild(el("span", "src", c.document));
    card.appendChild(head);
    card.appendChild(el("div", "quote", c.quote || "(no quote)"));
    const resolves = c.resolves === false ? "unresolved" : "resolves to stored chunk";
    card.appendChild(el("div", "resolve", "✓ " + resolves));
    box.appendChild(card);
  });
}

function renderRoutes(routes) {
  const box = $("routes");
  box.innerHTML = "";
  const known = routes || [];
  if (known.length === 0) {
    box.appendChild(el("span", "empty", "no routes"));
    return;
  }
  known.forEach((r) => box.appendChild(el("span", "chip fired", r)));
}

function renderPipeline(fired, stages) {
  const box = $("pipeline");
  box.innerHTML = "";
  const explained = {};
  (stages || []).forEach((s) => (explained[s.route] = [s.label, s.explanation]));
  const cacheRoute = [...fired].find((r) => r.indexOf("cag:") === 0);
  const skeleton = cacheRoute ? [cacheRoute, ...PIPELINE] : PIPELINE.slice();
  const caption = $("pipeline-caption");
  skeleton.forEach((route, i) => {
    if (i > 0) box.appendChild(el("span", "stage-arrow", "›"));
    const meta = explained[route] || EXPLAIN[route] || [route, "retrieval stage"];
    const on = fired.has(route) || route === "generate" || route === "cite";
    const isCache = route.indexOf("cag:") === 0;
    const node = el("span", "stage" + (on ? " on" : "") + (isCache && on ? " cache" : ""), meta[0]);
    node.setAttribute("title", meta[1]);
    node.addEventListener("mouseenter", () => {
      caption.innerHTML = "";
      caption.appendChild(el("b", "", meta[0]));
      caption.appendChild(document.createTextNode(" — " + meta[1] + (on ? " (fired)" : " (not used)")));
    });
    box.appendChild(node);
  });
  const firedCount = skeleton.filter((r) => fired.has(r) || r === "generate" || r === "cite").length;
  if (stages) {
    caption.innerHTML = "";
    caption.appendChild(el("b", "", firedCount + " of " + skeleton.length + " stages fired"));
    caption.appendChild(document.createTextNode(" for this query. hover a stage for what it does."));
  }
}

function renderMeta(m) {
  const meta = $("meta");
  meta.innerHTML = "";
  if (m.abstained) {
    meta.appendChild(el("span", "warn", "no relevant evidence"));
  } else {
    meta.appendChild(el("span", m.grounded ? "ok" : "warn", m.grounded ? "grounded ✓" : "ungrounded"));
  }
  meta.appendChild(el("span", "", "mode: " + m.mode));
  meta.appendChild(el("span", "", "citations: " + (m.citations ? m.citations.length : 0)));
  if (typeof m.elapsed_ms === "number") meta.appendChild(el("span", "", "answer: " + m.elapsed_ms.toFixed(1) + " ms"));
  meta.appendChild(el("span", m.cached ? "ok" : "", m.cached ? "cache hit" : "fresh"));
  const extra = m.extra || {};
  if (extra.branch) meta.appendChild(el("span", "", "branch: " + extra.branch));
  if (extra.label) meta.appendChild(el("span", "", "crag: " + extra.label));
}

function renderTrajectory(extra) {
  const box = $("trajectory");
  box.innerHTML = "";
  if (!extra || !extra.trajectory) {
    box.appendChild(el("span", "empty", "agent modes only"));
    return;
  }
  const wrap = el("div", "trajectory");
  extra.trajectory.forEach((s) => {
    wrap.appendChild(el("span", "step " + (s.authorized ? "auth" : "deny"), (s.authorized ? "✓ " : "✗ ") + s.tool));
  });
  if (extra.trajectory.length === 0) wrap.appendChild(el("span", "empty", "no tool steps proposed"));
  box.appendChild(wrap);
}

function usageTokens(usage) {
  if (!usage || typeof usage !== "object") return 0;
  if (typeof usage.total_tokens === "number") return usage.total_tokens;
  return (Number(usage.prompt_tokens) || 0) + (Number(usage.completion_tokens) || 0);
}

function renderQueryStats(m) {
  const box = $("query-stats");
  box.innerHTML = "";
  const r = m.retrieval || {};
  statCard(box, "answer time", (m.elapsed_ms || 0).toFixed(1) + " ms", { tone: m.cached ? "cyan" : "good" });
  statCard(box, "served", m.cached ? "from cache" : "fresh", { tone: m.cached ? "cyan" : "good" });
  statCard(box, "retrieved", r.retrieved != null ? r.retrieved : "—", {});
  statCard(box, "reranked", r.reranked ? "yes" : "no", {});
  statCard(box, "packed tokens", r.packed_tokens != null ? r.packed_tokens : "—", {});
  statCard(box, "citations", m.citations ? m.citations.length : 0, { tone: "good" });
  statCard(box, "stages fired", (m.routes || []).length, {});
  const t = m.tokens || {};
  statCard(box, "model tokens", t.total_tokens || 0, {});
}

function renderSessionStats() {
  const box = $("session-stats");
  box.innerHTML = "";
  const rate = session.queries ? Math.round((session.hits / session.queries) * 100) : 0;
  const avg = session.queries ? session.totalMs / session.queries : 0;
  statCard(box, "queries", session.queries, {});
  statCard(box, "cache hits", session.hits, { tone: "cyan" });
  statCard(box, "hit rate", rate + " %", { tone: "cyan" });
  statCard(box, "avg latency", avg.toFixed(1) + " ms", {});
  statCard(box, "total latency", session.totalMs.toFixed(1) + " ms", {});
  statCard(box, "latency saved", session.savedMs.toFixed(1) + " ms", { tone: "good" });
}

function updateCache(m) {
  const question = m.question;
  const tokens = usageTokens(m.usage);
  session.queries += 1;
  session.totalMs += m.elapsed_ms || 0;
  const badge = $("cache-badge");
  const tally = $("cache-tally");
  const note = $("cache-note");
  badge.innerHTML = "";
  if (m.cached) {
    session.hits += 1;
    badge.className = "cache-badge hit";
    badge.appendChild(el("span", "lead", "served from cache, no model call"));
    const prior = lastMiss.get(question);
    let line = "answer returned in " + (m.elapsed_ms || 0).toFixed(2) + " ms";
    if (prior) {
      const saved = Math.max(0, prior.ms - (m.elapsed_ms || 0));
      session.savedMs += saved;
      session.savedTokens += prior.tokens;
      session.savingsSamples += 1;
      const pct = prior.ms > 0 ? Math.round((saved / prior.ms) * 100) : 0;
      line += " versus " + prior.ms.toFixed(2) + " ms on the fresh run, " + saved.toFixed(2) + " ms saved (" + pct + "% faster)";
    }
    badge.appendChild(document.createTextNode(line));
  } else {
    badge.className = "cache-badge miss";
    badge.appendChild(el("span", "lead", "fresh, model called"));
    badge.appendChild(document.createTextNode("full retrieval and generation in " + (m.elapsed_ms || 0).toFixed(2) + " ms"));
    lastMiss.set(question, { ms: m.elapsed_ms || 0, tokens: tokens });
  }

  tally.innerHTML = "";
  statCard(tally, "cache hits", session.hits + " / " + session.queries, { tone: "cyan" });
  statCard(tally, "latency saved", session.savedMs.toFixed(1) + " ms", { tone: "good" });
  statCard(tally, "tokens saved", session.savedTokens, { tone: "good" });
  statCard(tally, "savings samples", session.savingsSamples, {});

  note.innerHTML = "";
  note.appendChild(document.createTextNode("run the same query twice: the second run returns "));
  note.appendChild(el("b", "", "cached"));
  note.appendChild(
    document.createTextNode(
      " with no model call. every number here is server-measured, the token saving is the real usage the runtime reports (0 offline with the extractive generator, real model tokens once a model is set)."
    )
  );
}

function buildGraph(nodes, edges) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  edges = edges.filter((e) => byId.has(e.source) && byId.has(e.target)).slice(0, 20);
  const keep = new Set();
  edges.forEach((e) => {
    keep.add(e.source);
    keep.add(e.target);
  });
  nodes = nodes.filter((n) => keep.has(n.id));
  return { nodes, edges, byId: new Map(nodes.map((n) => [n.id, n])) };
}

function layout(nodes, edges, byId, W, H) {
  const N = nodes.length;
  const cx = W / 2;
  const cy = H / 2;
  const R = Math.min(W, H) * 0.36;
  nodes.forEach((n, i) => {
    const a = (2 * Math.PI * i) / Math.max(1, N);
    n.x = cx + R * Math.cos(a);
    n.y = cy + R * Math.sin(a);
  });
  if (N < 2) return;
  const k = Math.sqrt((W * H) / N) * 0.72;
  const iterations = 300;
  for (let it = 0; it < iterations; it++) {
    const temp = 1 - it / iterations;
    nodes.forEach((a) => {
      a._dx = 0;
      a._dy = 0;
    });
    for (let i = 0; i < N; i++) {
      const a = nodes[i];
      for (let j = i + 1; j < N; j++) {
        const b = nodes[j];
        let ex = a.x - b.x;
        let ey = a.y - b.y;
        const d = Math.hypot(ex, ey) || 0.01;
        const rep = (k * k) / d;
        const ux = (ex / d) * rep;
        const uy = (ey / d) * rep;
        a._dx += ux;
        a._dy += uy;
        b._dx -= ux;
        b._dy -= uy;
      }
    }
    edges.forEach((e) => {
      const a = byId.get(e.source);
      const b = byId.get(e.target);
      let ex = a.x - b.x;
      let ey = a.y - b.y;
      const d = Math.hypot(ex, ey) || 0.01;
      const att = (d * d) / k;
      const fx = (ex / d) * att;
      const fy = (ey / d) * att;
      a._dx -= fx;
      a._dy -= fy;
      b._dx += fx;
      b._dy += fy;
    });
    const maxStep = 24 * temp + 2;
    nodes.forEach((n) => {
      n._dx += (cx - n.x) * 0.014;
      n._dy += (cy - n.y) * 0.014;
      const d = Math.hypot(n._dx, n._dy) || 0.01;
      const step = Math.min(d, maxStep);
      n.x += (n._dx / d) * step;
      n.y += (n._dy / d) * step;
      n.x = Math.max(28, Math.min(W - 28, n.x));
      n.y = Math.max(20, Math.min(H - 20, n.y));
    });
  }
}

let graphState = null;

function renderGraphSvg(data, question) {
  const stage = $("graph-svg");
  const info = $("graph-info");
  const legend = $("graph-legend");
  while (stage.firstChild) stage.removeChild(stage.firstChild);
  legend.innerHTML = "";
  const nodes = data.nodes || [];
  const edges = data.edges || [];
  if (nodes.length === 0 || edges.length === 0) {
    info.innerHTML = "";
    info.appendChild(el("span", "empty", "no graph relations for this query"));
    graphState = null;
    return;
  }
  const g = buildGraph(nodes, edges);
  const W = 860;
  const H = 560;
  stage.setAttribute("viewBox", "0 0 " + W + " " + H);
  layout(g.nodes, g.edges, g.byId, W, H);

  const kinds = [...new Set(g.nodes.map((n) => n.kind))];
  const colorOf = (kind) => NODE_COLORS[kinds.indexOf(kind) % NODE_COLORS.length];

  const neighbors = new Map(g.nodes.map((n) => [n.id, new Set()]));
  const edgeEls = [];
  const labelEls = [];
  g.edges.forEach((e) => {
    const a = g.byId.get(e.source);
    const b = g.byId.get(e.target);
    neighbors.get(e.source).add(e.target);
    neighbors.get(e.target).add(e.source);
    const line = svg("line", { class: "g-edge", x1: a.x, y1: a.y, x2: b.x, y2: b.y });
    line.dataset.a = e.source;
    line.dataset.b = e.target;
    stage.appendChild(line);
    edgeEls.push(line);
    const label = svg("text", {
      class: "g-edge-label",
      x: (a.x + b.x) / 2,
      y: (a.y + b.y) / 2 - 2,
      "text-anchor": "middle",
    });
    label.textContent = e.relation;
    label.dataset.a = e.source;
    label.dataset.b = e.target;
    stage.appendChild(label);
    labelEls.push(label);
  });

  const nodeEls = new Map();
  g.nodes.forEach((n) => {
    const group = svg("g", { class: "g-node" });
    const deg = neighbors.get(n.id).size;
    const radius = 6 + Math.min(8, deg * 1.4);
    const circle = svg("circle", { cx: n.x, cy: n.y, r: radius, fill: colorOf(n.kind) });
    const text = svg("text", { x: n.x, y: n.y - radius - 3, "text-anchor": "middle" });
    text.textContent = n.label.length > 22 ? n.label.slice(0, 21) + "…" : n.label;
    group.appendChild(circle);
    group.appendChild(text);
    group.dataset.id = n.id;
    stage.appendChild(group);
    nodeEls.set(n.id, group);
    group.addEventListener("mouseenter", () => highlight(n.id));
    group.addEventListener("mouseleave", () => {
      if (!graphState.pinned) clearHighlight();
    });
    group.addEventListener("click", () => {
      graphState.pinned = graphState.pinned === n.id ? null : n.id;
      if (graphState.pinned) highlight(n.id, true);
      else clearHighlight();
    });
  });

  graphState = { nodeEls, edgeEls, labelEls, neighbors, byId: g.byId, pinned: null };

  function highlight(id, pin) {
    const near = graphState.neighbors.get(id);
    graphState.nodeEls.forEach((elm, nid) => {
      const on = nid === id || near.has(nid);
      elm.classList.toggle("hot", nid === id);
      elm.classList.toggle("dim", !on);
    });
    graphState.edgeEls.forEach((line) => {
      const on = line.dataset.a === id || line.dataset.b === id;
      line.classList.toggle("hot", on);
      line.classList.toggle("dim", !on);
    });
    graphState.labelEls.forEach((label) => {
      label.classList.toggle("hot", label.dataset.a === id || label.dataset.b === id);
    });
    const node = graphState.byId.get(id);
    info.innerHTML = "";
    info.appendChild(el("b", "", node.label));
    info.appendChild(document.createTextNode(" · " + near.size + " connected · kind: " + node.kind + (pin ? " · pinned" : "")));
  }
  function clearHighlight() {
    graphState.nodeEls.forEach((elm) => elm.classList.remove("hot", "dim"));
    graphState.edgeEls.forEach((line) => line.classList.remove("hot", "dim"));
    graphState.labelEls.forEach((label) => label.classList.remove("hot"));
    baseInfo();
  }
  function baseInfo() {
    info.innerHTML = "";
    info.appendChild(el("b", "", g.nodes.length + " entities"));
    info.appendChild(document.createTextNode(" · " + g.edges.length + " relations · hover a node, click to pin"));
  }

  kinds.forEach((kind) => {
    const key = el("span", "key");
    const dot = el("span", "dot");
    dot.style.background = colorOf(kind);
    key.appendChild(dot);
    key.appendChild(document.createTextNode(kind));
    legend.appendChild(key);
  });
  baseInfo();
}

async function renderGraph(question) {
  const res = await fetch("/api/graph?question=" + encodeURIComponent(question));
  const data = await res.json();
  renderGraphSvg(data, question);
}

function commit(m) {
  const answerBox = $("answer");
  answerBox.classList.toggle("abstain", !!m.abstained);
  renderCitations(m.abstained ? [] : m.citations);
  renderRoutes(m.routes);
  renderPipeline(new Set(m.routes || []), m.stages);
  renderMeta(m);
  renderTrajectory(m.extra);
  renderQueryStats(m);
  updateCache(m);
  renderSessionStats();
  renderCost(m);
  renderSaved(m);
  updateSpeakButton(m);
}

let speaking = false;

function ttsSupported() {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

function answerText() {
  const box = $("answer");
  if (!box || box.querySelector(".empty")) return "";
  return (box.textContent || "").trim();
}

function stopSpeaking() {
  speaking = false;
  const btn = $("speak");
  if (btn) {
    btn.textContent = "🔊 speak";
    btn.classList.remove("speaking");
  }
  if (ttsSupported()) window.speechSynthesis.cancel();
}

function updateSpeakButton(m) {
  const btn = $("speak");
  if (!btn || !ttsSupported()) return;
  stopSpeaking();
  btn.disabled = !!m.abstained || !answerText();
}

function wireTts() {
  const btn = $("speak");
  if (!btn) return;
  if (!ttsSupported()) {
    btn.hidden = true;
    btn.title = "text to speech is not supported in this browser";
    return;
  }
  btn.title = "read the grounded answer aloud";
  btn.addEventListener("click", () => {
    if (speaking) {
      stopSpeaking();
      return;
    }
    const text = answerText();
    if (!text) return;
    const utter = new SpeechSynthesisUtterance(text);
    const box = $("answer");
    utter.lang = (box && box.getAttribute("data-lang")) || "en-US";
    utter.onend = stopSpeaking;
    utter.onerror = stopSpeaking;
    window.speechSynthesis.cancel();
    speaking = true;
    btn.textContent = "⏹ stop";
    btn.classList.add("speaking");
    window.speechSynthesis.speak(utter);
  });
}

function fmtUsd(v) {
  const n = typeof v === "number" ? v : 0;
  return "$" + n.toFixed(6);
}

function renderCost(m) {
  const box = $("cost-stats");
  box.innerHTML = "";
  const t = m.tokens || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 };
  statCard(box, "prompt tokens", t.prompt_tokens || 0, {});
  statCard(box, "completion tokens", t.completion_tokens || 0, {});
  statCard(box, "total tokens", t.total_tokens || 0, { tone: "good" });
  statCard(box, "usd cost", fmtUsd(m.usd_cost), { tone: "good" });
  const s = m.session_economics || {};
  statCard(box, "session tokens", s.total_tokens != null ? s.total_tokens : 0, { tone: "cyan" });
  statCard(box, "session usd", fmtUsd(s.total_usd), { tone: "cyan" });
  const note = $("cost-note");
  const total = t.total_tokens || 0;
  let msg;
  if (!isOnline) {
    msg =
      "offline mode: the deterministic extractive generator makes no model calls, so tokens and cost are honestly zero. set REFAAT_OPENAI_API_KEY to run a real model and see live cost.";
  } else if (m.cached) {
    msg =
      "served from the answer cache: no model call this query, so zero new tokens. the saved-by-caching panel shows the tokens and cost this hit avoided.";
  } else if (m.abstained) {
    msg =
      "abstained: the best evidence scored below the floor, so no answer was generated and no model tokens were spent.";
  } else if (total === 0) {
    msg = "no model tokens were reported for this query.";
  } else {
    msg = "real usage: cost is the response tokens times the configured price for " + (genModel || "the model") + ".";
  }
  note.textContent = msg;
}

function renderSaved(m) {
  const box = $("saved-stats");
  box.innerHTML = "";
  const saved = m.saved || { tokens: 0, usd: 0.0, latency_ms: 0.0 };
  const s = m.session_economics || {};
  statCard(box, "tokens saved", saved.tokens || 0, { tone: "good" });
  statCard(box, "usd saved", fmtUsd(saved.usd), { tone: "good" });
  statCard(box, "latency saved", (saved.latency_ms || 0).toFixed(1) + " ms", { tone: "cyan" });
  statCard(box, "session tokens saved", s.total_saved_tokens != null ? s.total_saved_tokens : 0, { tone: "good" });
  statCard(box, "session usd saved", fmtUsd(s.total_saved_usd), { tone: "good" });
  statCard(box, "session latency saved", (s.total_saved_latency_ms || 0).toFixed(1) + " ms", { tone: "cyan" });
}

function wireObservability() {
  const modal = $("obs-modal");
  const open = () => {
    modal.hidden = false;
    loadObservability();
  };
  const close = () => {
    modal.hidden = true;
  };
  $("observe").addEventListener("click", open);
  $("obs-close").addEventListener("click", close);
  $("obs-close-bg").addEventListener("click", close);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !modal.hidden) close();
  });
}

async function loadObservability() {
  const trace = $("obs-trace");
  trace.innerHTML = "";
  trace.appendChild(el("span", "empty", "loading..."));
  let data;
  try {
    data = await (await fetch("/api/observability")).json();
  } catch (err) {
    trace.innerHTML = "";
    trace.appendChild(el("span", "warn", "observability error: " + err));
    return;
  }
  renderTrace(data.trace);
  renderBudget(data.budget);
  renderEcon(data.economics);
  renderEvents(data.events);
}

function renderTrace(t) {
  const title = $("obs-trace-title");
  title.innerHTML = "";
  title.appendChild(document.createTextNode("trace "));
  const src = el("span", "obs-src", t.source === "opentelemetry" ? "real OpenTelemetry spans" : "real stage trace");
  title.appendChild(src);
  const box = $("obs-trace");
  box.innerHTML = "";
  const spans = t.spans || [];
  if (spans.length === 0) {
    box.appendChild(el("span", "empty", "no spans captured"));
    return;
  }
  const max = Math.max(...spans.map((s) => s.duration_ms || 0), 0.001);
  const byId = new Map(spans.map((s) => [s.span_id, s]));
  spans.forEach((s) => {
    const row = el("div", "span-row");
    const depth = s.parent_span_id && byId.has(s.parent_span_id) ? 1 : 0;
    const head = el("div", "span-head");
    head.style.paddingLeft = depth * 16 + "px";
    head.appendChild(el("span", "span-name", (depth ? "└ " : "") + s.name));
    head.appendChild(el("span", "span-dur", (s.duration_ms || 0).toFixed(2) + " ms"));
    row.appendChild(head);
    const track = el("div", "span-track");
    const bar = el("div", "span-bar");
    bar.style.width = Math.max(3, ((s.duration_ms || 0) / max) * 100) + "%";
    track.appendChild(bar);
    row.appendChild(track);
    const attrs = s.attributes || {};
    const keys = Object.keys(attrs);
    if (keys.length) {
      const meta = el("div", "span-attrs");
      keys.forEach((k) => meta.appendChild(el("span", "span-attr", k + "=" + attrs[k])));
      row.appendChild(meta);
    }
    box.appendChild(row);
  });
}

function renderBudget(b) {
  const box = $("obs-budget");
  box.innerHTML = "";
  if (!b) return;
  statCard(box, "token ceiling", b.token_ceiling != null ? b.token_ceiling : "—", {});
  statCard(box, "tokens used", b.tokens_used != null ? b.tokens_used : 0, { tone: "good" });
  statCard(box, "tokens remaining", b.remaining_tokens != null ? b.remaining_tokens : "—", { tone: "cyan" });
  statCard(box, "cost used", fmtUsd(b.cost_used_usd), { tone: "good" });
}

function renderEcon(e) {
  const box = $("obs-econ");
  box.innerHTML = "";
  if (!e) return;
  statCard(box, "queries", e.queries || 0, {});
  statCard(box, "cache hits", e.cache_hits || 0, { tone: "cyan" });
  statCard(box, "total tokens", e.total_tokens || 0, { tone: "good" });
  statCard(box, "total usd", fmtUsd(e.total_usd), { tone: "good" });
  statCard(box, "tokens saved", e.total_saved_tokens || 0, { tone: "good" });
  statCard(box, "latency saved", (e.total_saved_latency_ms || 0).toFixed(1) + " ms", { tone: "cyan" });
}

function renderEvents(events) {
  const box = $("obs-events");
  box.innerHTML = "";
  if (!events || events.length === 0) {
    box.appendChild(el("span", "empty", "no events recorded"));
    return;
  }
  events.forEach((ev) => {
    const row = el("div", "event-row");
    row.appendChild(el("span", "event-topic", ev.topic || "event"));
    row.appendChild(el("span", "event-doc", ev.document_id || ""));
    row.appendChild(el("span", "event-ver", "v" + (ev.corpus_version != null ? ev.corpus_version : "?")));
    box.appendChild(row);
  });
}

function clearGraph() {
  renderGraphSvg({ nodes: [], edges: [] }, "");
}

async function run() {
  if (running) return;
  const question = $("q").value.trim();
  if (!question) return;
  setBusy(true);
  const answer = $("answer");
  answer.innerHTML = "";
  const streaming = $("stream").checked;
  try {
    const abstained = streaming ? await runStream(question, answer) : await runOnce(question, answer);
    // An abstention has no evidence, so the graph is emptied rather than showing
    // entities the answer does not stand on.
    if (abstained) clearGraph();
    else await renderGraph(question);
  } catch (err) {
    answer.textContent = "error: " + err;
  } finally {
    setBusy(false);
  }
}

async function runOnce(question, answer) {
  const res = await fetch("/api/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode }),
  });
  const p = await res.json();
  answer.textContent = p.answer || "(no answer)";
  commit({
    question,
    mode: p.mode,
    answer: p.answer,
    citations: p.citations,
    routes: p.routes,
    stages: p.stages,
    grounded: p.grounded,
    abstained: p.abstained,
    cached: p.cached,
    usage: p.usage,
    elapsed_ms: p.elapsed_ms,
    retrieval: p.retrieval,
    extra: p.extra,
    tokens: p.tokens,
    usd_cost: p.usd_cost,
    saved: p.saved,
    session_economics: p.session_economics,
  });
  return !!p.abstained;
}

function runStream(question, answer) {
  return new Promise((resolve, reject) => {
    const span = el("span", "caret");
    answer.appendChild(span);
    const source = new EventSource("/api/query/stream?question=" + encodeURIComponent(question));
    source.addEventListener("delta", (e) => {
      span.textContent += JSON.parse(e.data).text;
    });
    source.addEventListener("final", (e) => {
      const data = JSON.parse(e.data);
      span.classList.remove("caret");
      span.textContent = data.answer;
      const citations = (data.citations || []).map((c) => ({
        chunk_id: c.chunk_id,
        quote: c.quote,
        document: c.document_id,
        resolves: data.citations_resolve,
      }));
      commit({
        question,
        mode: "stream",
        answer: data.answer,
        citations,
        routes: data.routes,
        stages: data.stages,
        grounded: data.citations_resolve,
        abstained: data.abstained,
        cached: data.cached,
        usage: data.usage,
        elapsed_ms: data.elapsed_ms,
        retrieval: data.retrieval,
        extra: null,
        tokens: data.tokens,
        usd_cost: data.usd_cost,
        saved: data.saved,
        session_economics: data.session_economics,
      });
      source.close();
      resolve(!!data.abstained);
    });
    source.onerror = () => {
      source.close();
      reject(new Error("stream closed"));
    };
  });
}

boot();
