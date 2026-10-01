"use strict";

const $ = (id) => document.getElementById(id);
const names = {baseline: "Uncompressed", lowram: "LowRAM · full adaptation", fast: "LowRAM · meta-init + selective"};
// LowRAM keeps one colour everywhere; the uncompressed baseline is the neutral reference.
const colors = {"LowRAM": "#b5412a", "LowRAM-EQ": "#2f6db3", "Quantisation": "#9a7b00", "Eviction": "#d17fa6", "Uncompressed": "#8a8780", "LowRAM-SR": "#b5412a"};
const CONTEXT_TOKENS = 65536;
const TARGET_RATIO = 4;
let report = null;
let paper = null;
let archive = null;
const text = (id, value) => { $(id).textContent = value; };
const number = (value, digits = 1) => Number.isFinite(value) ? value.toFixed(digits) : "—";
const pct = (value) => Number.isFinite(value) ? `${number(value * 100)}%` : "—";
const bytes = (value) => Number.isFinite(value) ? (value >= 2 ** 30 ? `${number(value / 2 ** 30, 2)} GiB` : `${number(value / 2 ** 20)} MiB`) : "—";
const elem = (tag, value, className) => {
  const el = document.createElement(tag);
  if (value !== undefined) el.textContent = value;
  if (className) el.className = className;
  return el;
};
const distinct = (items) => [...new Set(items)].sort((a, b) => typeof a === "number" ? a - b : 0);
function selectOptions(id, values, label = String) {
  const el = $(id), previous = el.value;
  el.replaceChildren(...values.map((v) => {
    const option = elem("option", label(v)); option.value = v; return option;
  }));
  if (values.some((v) => String(v) === previous)) el.value = previous;
  el.disabled = !values.length;
}

function validate(data) {
  if (data.schema_version !== 1 || data.kind !== "lowram_comparison" || !data.metadata?.model ||
      !Array.isArray(data.examples) || !Array.isArray(data.comparisons) || !data.examples.length || !data.comparisons.length) {
    throw new Error("Use a JSON export from python -m benchmarks.compare (schema version 1).");
  }
  for (const example of data.examples) {
    if (typeof example.id !== "string" || typeof example.document !== "string" || typeof example.question !== "string" ||
        !Array.isArray(example.references) || !example.references.length || !example.references.every(v => typeof v === "string") ||
        !Number.isFinite(example.context_tokens)) throw new Error("Invalid document or references in the run.");
  }
  for (const row of data.comparisons) {
    if (!Object.hasOwn(names, row.method) || !["ok", "oom"].includes(row.status) ||
        ![row.context_tokens, row.batch_size, row.target_ratio].every(v => Number.isFinite(v) && v > 0)) throw new Error("Invalid comparison settings.");
    if (row.status === "ok" && (!row.summary?.accuracy || !Array.isArray(row.runs) ||
        !row.summary.physical_compressed_bytes || !row.summary.decode_tokens_per_s || !row.summary.ttft_s)) throw new Error("A completed comparison is missing metrics.");
    if (row.status === "ok" && row.runs.some(run => !Array.isArray(run.answers) ||
        run.answers.some(a => typeof a.answer !== "string" || typeof a.example_id !== "string"))) throw new Error("Invalid recorded answers.");
  }
  if (!data.comparisons.some(r => r.method === "baseline") || !data.comparisons.some(r => r.method !== "baseline")) {
    throw new Error("Include both baseline and a LowRAM configuration in your benchmark.");
  }
  return data;
}

function acceptReport(data) {
  report = validate(data);
  $("load-error").hidden = true;
  text("run-label", "Recorded run");
  text("run-model", report.metadata.model.split("/").pop());
  if (report.comparisons.some(r => r.context_tokens !== CONTEXT_TOKENS || (r.method !== "baseline" && r.target_ratio !== TARGET_RATIO))) {
    throw new Error("The demo requires 64K contexts and a 4× target compression ratio.");
  }
  selectOptions("method-select", distinct(report.comparisons.filter(r => r.method !== "baseline").map(r => r.method)), v => names[v]);
  if (report.comparisons.some(r => r.method === "fast")) $("method-select").value = "fast";
  updateControls("initial");
}

function updateControls(from) {
  if (!report) return;
  const rows = report.comparisons.filter(r => r.method === $("method-select").value);
  const batches = distinct(rows.map(r => r.batch_size));
  selectOptions("batch-select", batches, v => `${v} ${v === 1 ? "sequence" : "sequences"}`);
  if (from === "initial") $("batch-select").value = String(Math.max(...batches));
  renderComparison();
}

function selectedRow(method) {
  return report.comparisons.find(r => r.method === method && r.context_tokens === CONTEXT_TOKENS &&
    r.batch_size === Number($("batch-select").value) && (method === "baseline" || r.target_ratio === TARGET_RATIO));
}

function baselineForDisplay(selected) {
  if (selected?.status !== "oom") return selected;
  return report.comparisons.filter(r => r.method === "baseline" && r.status === "ok" &&
    r.context_tokens === CONTEXT_TOKENS && r.batch_size < selected.batch_size)
    .sort((a, b) => b.batch_size - a.batch_size)[0] || selected;
}

function answerCard(prefix, row, example) {
  const checks = $(`${prefix}-checks`); checks.replaceChildren();
  if (!row || row.status !== "ok") {
    text(`${prefix}-answer`, row?.status === "oom" ? "This configuration ran out of GPU memory." : "No completed measurement for this setting.");
    text(`${prefix}-score`, row?.status === "oom" ? "OOM" : "Not measured"); return;
  }
  const answer = row.runs.filter(r => r.repeat === 0).flatMap(r => r.answers).find(a => a.example_id === example.id);
  if (!answer) { text(`${prefix}-answer`, "Answer not recorded."); text(`${prefix}-score`, "—"); return; }
  text(`${prefix}-answer`, answer.answer || "(Empty answer)");
  text(`${prefix}-score`, pct(answer.score));
  (answer.correct || []).forEach((hit, i) => checks.append(elem("span", `${hit ? "✓" : "×"} ${example.fact_labels?.[i] || `Fact ${i + 1}`}`, hit ? "" : "missed")));
}

function renderComparison() {
  const example = report.examples.find(e => e.context_tokens === CONTEXT_TOKENS);
  if (!example) return;
  const requestedBaseline = selectedRow("baseline"), baseline = baselineForDisplay(requestedBaseline);
  const compressed = selectedRow($("method-select").value);
  const fallback = baseline !== requestedBaseline;
  const batch = Number($("batch-select").value);
  $("baseline-note").hidden = requestedBaseline?.status !== "oom";
  text("baseline-note", fallback
    ? `Baseline: OOM at batch ${batch}. Its answer and prefill cache size below come from batch ${baseline.batch_size}, the largest recorded smaller batch that fits. LowRAM uses batch ${batch}.`
    : `Baseline: OOM at batch ${batch}. No completed smaller-batch baseline is available.`);
  text("baseline-answer-label", `Uncompressed · batch ${baseline?.batch_size ?? batch}${fallback ? " (smaller batch)" : ""}`);
  text("baseline-memory-label", `Baseline · batch ${baseline?.batch_size ?? batch}`);
  text("lowram-memory-label", `LowRAM · batch ${batch}`);
  text("baseline-column", `Baseline · batch ${batch}`);
  text("lowram-column", `LowRAM · batch ${batch}`);
  const b = baseline?.status === "ok" ? baseline.summary : null;
  const baselineMetrics = requestedBaseline?.status === "ok" ? requestedBaseline.summary : null;
  const c = compressed?.status === "ok" ? compressed.summary : null;
  text("question", example.question);
  text("document-text", example.document);
  text("answer-method", `${names[$("method-select").value]} · batch ${batch}`);
  $("facts").replaceChildren(...example.references.map((ref, i) => elem("span", `${example.fact_labels?.[i] ? `${example.fact_labels[i]} · ` : ""}${ref}`, "fact-chip")));
  $("document-map").replaceChildren(...Array.from({length: 48}, () => elem("i")));
  (example.fact_positions || []).forEach((position, i) => {
    if (!Number.isFinite(position) || position < 0 || position > 1) return;
    const marker = elem("button", undefined, "fact-marker");
    marker.style.left = `${position * 100}%`; marker.title = example.facts?.[i] || example.references[i];
    marker.setAttribute("aria-label", `${example.fact_labels?.[i] || `Fact ${i + 1}`} at ${Math.round(position * 100)}% of the prompt`);
    marker.onclick = () => { $("document-text").hidden = false; $("document-toggle").setAttribute("aria-expanded", "true"); text("document-toggle", "Hide document"); };
    $("document-map").append(marker);
  });
  answerCard("baseline", baseline, example); answerCard("lowram", compressed, example);
  const baseBytes = b?.physical_compressed_bytes.mean, lowBytes = c?.physical_compressed_bytes.mean;
  text("baseline-memory", baseline?.status === "oom" ? "OOM" : bytes(baseBytes)); text("lowram-memory", compressed?.status === "oom" ? "OOM" : bytes(lowBytes));
  const scale = Math.max(baseBytes || 0, lowBytes || 0, 1);
  $("baseline-bar").style.width = `${(baseBytes || 0) / scale * 100}%`;
  $("lowram-bar").style.width = `${(lowBytes || 0) / scale * 100}%`;
  text("memory-saving", !fallback && baseBytes && lowBytes ? (baseBytes >= lowBytes ? `${number(baseBytes / lowBytes, 2)}× smaller cache` : `${number(lowBytes / baseBytes, 2)}× larger cache`) : c?.physical_compression_ratio?.mean ? `${number(c.physical_compression_ratio.mean, 2)}× achieved (cache accounting)` : "Not measured");
  const metrics = [
    ["Decode throughput", s => `${number(s?.decode_tokens_per_s.mean)} tok/s`],
    ["Time to first token", s => `${number(s?.ttft_s.mean, 2)} s`],
    ["End-to-end throughput", s => `${number(s?.end_to_end_tokens_per_s?.mean)} tok/s`],
    ["Peak GPU tensor memory", s => bytes(s?.peak_allocated_bytes)],
    ["Peak GPU reserved memory", s => bytes(s?.peak_reserved_bytes)],
    ["Suite fact recall", s => pct(s?.accuracy.score)],
  ];
  $("metric-rows").replaceChildren(...metrics.map(([label, format]) => {
    const tr = elem("tr"); tr.append(elem("td", label), elem("td", baselineMetrics ? format(baselineMetrics) : requestedBaseline?.status === "oom" ? "OOM" : "—"), elem("td", c ? format(c) : compressed?.status === "oom" ? "OOM" : "—")); return tr;
  }));
}

function svgEl(tag, attributes = {}, value) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes).forEach(([key, val]) => el.setAttribute(key, String(val)));
  if (value !== undefined) el.textContent = value;
  return el;
}

function chart(container, series, options) {
  const width = 960, height = 355, pad = {left: 65, right: 120, top: 27, bottom: 58};
  const x = value => pad.left + (value - options.xMin) / (options.xMax - options.xMin) * (width - pad.left - pad.right);
  const y = value => height - pad.bottom - (value - options.yMin) / (options.yMax - options.yMin) * (height - pad.top - pad.bottom);
  const svg = svgEl("svg", {viewBox: `0 0 ${width} ${height}`, role: "group", "aria-label": options.title});
  svg.append(svgEl("title", {}, options.title));
  options.yTicks.forEach(tick => {
    svg.append(svgEl("line", {x1: pad.left, x2: width - pad.right, y1: y(tick), y2: y(tick), class: tick === 100 && options.baseline ? "gridline reference" : "gridline"}));
    svg.append(svgEl("text", {x: pad.left - 12, y: y(tick) + 4, "text-anchor": "end", class: "tick"}, tick));
  });
  options.xTicks.forEach(tick => svg.append(svgEl("text", {x: x(tick), y: height - 33, "text-anchor": "middle", class: "tick"}, options.xFormat ? options.xFormat(tick) : tick)));
  svg.append(svgEl("text", {x: pad.left + (width - pad.left - pad.right) / 2, y: height - 7, "text-anchor": "middle", class: "axis-label"}, options.xLabel));
  svg.append(svgEl("text", {x: pad.left, y: 12, class: "axis-label"}, options.yLabel));
  const legend = elem("div", undefined, "chart-legend"), labels = [];
  series.forEach(({name, points}) => {
    const color = colors[name] || "#b5412a";
    const valid = points.filter(p => p.status !== "oom").sort((a, b) => a.x - b.x);
    svg.append(svgEl("polyline", {points: valid.map(p => `${x(p.x)},${y(p.y)}`).join(" "), fill: "none", stroke: color, "stroke-width": 2.5}));
    points.forEach(point => {
      const oom = point.status === "oom";
      const cx = x(point.x), cy = y(oom ? options.yMin : point.y);
      const circle = svgEl("circle", {cx, cy, r: oom ? 4 : 5, fill: oom ? "var(--bg)" : color, stroke: color, "stroke-width": 1.5, tabindex: 0, role: "button", "aria-label": point.detail, class: "point"});
      circle.append(svgEl("title", {}, point.detail));
      const show = () => text(options.detail, point.detail);
      circle.addEventListener("click", show); circle.addEventListener("mouseenter", show); circle.addEventListener("focus", show);
      circle.addEventListener("keydown", e => { if (["Enter", " "].includes(e.key)) { e.preventDefault(); show(); } });
      svg.append(circle);
      if (oom) svg.append(svgEl("text", {x: cx, y: cy - 11, "text-anchor": "middle", class: "oom-label"}, "OOM"));
    });
    const last = valid[valid.length - 1];
    if (last) labels.push({name, color, x: x(last.x) + 10, y: y(last.y) + 4});
    const label = elem("span"), dot = elem("i"); dot.style.background = color; label.append(dot, document.createTextNode(name)); legend.append(label);
  });
  // Direct labels at line ends, nudged apart so neighbouring series stay readable.
  labels.sort((a, b) => a.y - b.y).forEach((label, i, all) => { if (i && label.y - all[i - 1].y < 14) label.y = all[i - 1].y + 14; });
  labels.forEach(({name, color, x: lx, y: ly}) => {
    svg.append(svgEl("circle", {cx: lx + 3, cy: ly - 4, r: 3, fill: color}));
    svg.append(svgEl("text", {x: lx + 10, y: ly, class: "series-label"}, name));
  });
  $(container).replaceChildren(svg, legend);
}

function accuracyChart() {
  if (!paper) return;
  const methods = Object.keys(paper.long_context);
  const series = methods.map(name => ({name, points: paper.long_context[name].map(([ratio, score]) => {
    return {x: ratio, y: score, detail: `${name} · ${ratio}× compression · ${number(100 / ratio)}% of baseline cache memory · ${score}% recovered LongBench score. Mistral-7B, contexts >10K; paper Table 11.`};
  })}));
  chart("accuracy-chart", series, {title: "Recovered accuracy versus KV cache compression ratio", xMin: 0, xMax: 30, yMin: 25, yMax: 105, xTicks: [1, 5, 10, 15, 20, 25, 30], xFormat: tick => `${tick}×`, yTicks: [40, 60, 80, 100], xLabel: "KV cache compression ratio →", yLabel: "Recovered benchmark score (%) ↑", baseline: true, detail: "accuracy-detail"});
}

function throughputChart() {
  if (!archive) return;
  const context = 262144;
  const series = archive.runs.map(run => ({name: run.label, points: run.results.filter(r => r.prompt_len === context).map(r => {
    return {x: r.batch_size, y: r.decode_tokens_per_s, status: r.status, detail: `${run.label} · batch ${r.batch_size} · ${r.status === "oom" ? "out of memory" : `${number(r.decode_tokens_per_s)} aggregate tok/s`}`};
  })}));
  chart("throughput-chart", series, {title: "Archived decode throughput versus batch size", xMin: 0, xMax: 30, yMin: 0, yMax: 320, xTicks: [1, 4, 8, 12, 18, 24, 28], yTicks: [0, 100, 200, 300], xLabel: "Batch size →", yLabel: "Aggregate decode throughput (tokens / second) ↑", detail: "throughput-detail"});
}

const steps = [
  ["K ≈ A · B", "Shared structure, smaller keys.", "A shared factor captures patterns across a group of layers. Each layer keeps its own smaller reconstruction factor."],
  ["<span class=\"accented\">V<span class=\"accent\">ˆ</span></span> = ƒ<sub>θ</sub>(K)", "Learn the relationship once per context.", "A small per-head network learns to predict values from their keys. Its parameter cost is amortised over a long document."],
  ["<span class=\"accented\">V<span class=\"accent\">˜</span></span> = ƒ<sub>θ</sub>(K) + ΔV", "Spend memory on the hard cases.", "Store sparse residual corrections for the worst-reconstructed values. The target compression ratio determines how many corrections fit."],
  ["Query → select → reconstruct", "Rebuild only what the query needs.", "Landmarks select relevant chunks. Reuse recently reconstructed chunks, while retaining outliers and a local window exactly."],
];
function methodStep(index) {
  document.querySelectorAll(".method-step").forEach((b, i) => { b.classList.toggle("active", i === index); b.setAttribute("aria-pressed", String(i === index)); });
  const matrix = (count, type = "") => { const m = elem("div", undefined, `matrix ${type}`); m.append(...Array.from({length: count}, () => elem("i"))); return m; };
  const op = value => elem("span", value, "operator");
  const predictor = elem("div", "ƒ", "predictor-symbol"); predictor.append(elem("sub", "θ"));
  const stages = [() => [matrix(36), op("≈"), matrix(12, "thin"), op("·"), matrix(12, "wide")],
    () => [matrix(36), op("→"), predictor, op("→"), matrix(36, "wide")],
    () => [predictor, op("+"), matrix(24, "exceptions"), op("→"), matrix(36, "wide")],
    () => [matrix(48, "selected"), op("→"), matrix(12), op("→"), predictor]];
  $("visual-stage").replaceChildren(...stages[index]());
  $("method-formula").innerHTML = steps[index][0]; text("method-title", steps[index][1]); text("method-caption", steps[index][2]);
}

document.querySelector(".full-grid").append(...Array.from({length: 80}, () => elem("i")));
document.querySelector(".small-grid").append(...Array.from({length: 30}, () => elem("i")));
document.querySelectorAll(".method-step").forEach(b => b.addEventListener("click", () => methodStep(Number(b.dataset.step)))); methodStep(0);
["method", "batch"].forEach(key => $(`${key}-select`).addEventListener("change", () => updateControls(key)));
$("document-toggle").onclick = () => { const show = $("document-text").hidden; $("document-text").hidden = !show; $("document-toggle").setAttribute("aria-expanded", String(show)); text("document-toggle", show ? "Hide document" : "Read document"); };
$("copy-command").onclick = async () => {
  try { await navigator.clipboard.writeText($("reproduce-command").textContent); text("copy-command", "Copied ✓"); }
  catch { text("copy-command", "Select text to copy"); }
};
async function getJSON(path) { const response = await fetch(path, {cache: "no-store"}); if (!response.ok) throw new Error(`Could not load ${path}`); return response.json(); }
getJSON("data/comparison-64k.json").then(acceptReport).catch(() => {
  text("run-label", "Results unavailable");
  $("load-error").hidden = false;
  text("load-error", "The recorded results could not be loaded. Refresh this page; if the problem persists, check that the website is served over HTTP with its data folder.");
});
getJSON("data/paper.json").then(data => { paper = data; accuracyChart(); }).catch(() => text("accuracy-detail", "Paper data could not be loaded. Read the linked paper for the results."));
getJSON("data/throughput.json").then(data => { archive = data; throughputChart(); }).catch(() => text("throughput-detail", "Archived data could not be loaded. Serve this folder over HTTP to view the plot."));
