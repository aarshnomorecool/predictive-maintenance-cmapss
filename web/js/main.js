import { EngineView, SEVERITY_COLORS, MODULE_ORDER } from "./engine3d.js";
import { lineChart, divergingBars, formatValue } from "./charts.js";

const $ = (id) => document.getElementById(id);
const POLL_MS = 5000;
const REPLAY_FPS = 14;

const SEVERITY_TEXT = {
  normal: "Normal",
  warning: "Drifting",
  critical: "Near failure level",
  low_signal: "Too little signal",
  unmonitored: "Not used by models",
};
const MODEL_LABELS = { RandomForest: "Random Forest", XGBoost: "XGBoost" };

const state = {
  meta: null,
  fleet: [],
  engine: null,            // /api/engine payload for the selected engine
  explain: null,           // /api/engine/:id/explain payload for the current cycle
  model: "XGBoost",
  threshold: 0.5,
  selectedEngine: null,
  selectedSensor: null,
  cycleIndex: 0,
  playing: false,
  token: null,
};

// ---------------------------------------------------------------- api ----

async function api(path) {
  const response = await fetch(path, { cache: "no-store" });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(body.error || response.statusText), { status: response.status });
  return body;
}

function showBanner(html) {
  $("banner").innerHTML = html;
  $("banner").hidden = !html;
}

function toast(text) {
  const node = $("toast");
  node.textContent = text;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 3200);
}

// ------------------------------------------------------------ helpers ----

const sensorMeta = (column) => state.meta.sensors.find((s) => s.column === column);
const symbolOf = (column) => sensorMeta(column)?.symbol ?? column;

function severityAt(column, index) {
  const meta = sensorMeta(column);
  if (!meta?.monitored) return "unmonitored";
  if (!meta.informative) return "low_signal";
  const drift = state.engine.sensors[column].drift[index];
  const { warning, critical } = state.meta.severity_levels;
  return drift >= critical ? "critical" : drift >= warning ? "warning" : "normal";
}

function severitiesAt(index) {
  const out = {};
  for (const s of state.meta.sensors) out[s.column] = severityAt(s.column, index);
  return out;
}

function driftBar(drift) {
  const { warning, critical } = state.meta.severity_levels;
  const max = 1.25;
  const pct = (v) => (Math.min(Math.max(v, 0), max) / max) * 100;
  return `
    <div class="drift" role="meter" aria-valuemin="0" aria-valuemax="${max}" aria-valuenow="${drift.toFixed(2)}"
         aria-label="Drift towards failure level">
      <div class="drift-track">
        <span style="width:${pct(warning)}%;background:var(--ok)"></span>
        <span style="width:${pct(critical) - pct(warning)}%;background:var(--warn)"></span>
        <span style="flex:1;background:var(--crit)"></span>
      </div>
      <span class="drift-pin" style="left:${pct(drift)}%"></span>
      <div class="drift-scale"><span>Healthy</span><span>Failure level</span></div>
    </div>`;
}

const probabilityAt = (i) => state.engine.probability[i];
const isAlert = (p) => p >= state.threshold;

// --------------------------------------------------------- 3D viewer ----

const view = new EngineView($("stage"), {
  onSelect: (column) => selectSensor(column),
  onHover: (column) => {
    hoveredSensor = column;
  },
});
let hoveredSensor = null;
let pointer = { x: 0, y: 0 };
$("stage").addEventListener("pointermove", (e) => {
  const r = $("stage").getBoundingClientRect();
  pointer = { x: e.clientX - r.left, y: e.clientY - r.top };
});

// Leader-line callout, redrawn every frame so it tracks the orbiting model.
view.onFrame(() => {
  const tag = $("hover-tag");
  if (hoveredSensor && hoveredSensor !== state.selectedSensor && state.meta) {
    tag.textContent = `${symbolOf(hoveredSensor)}: ${sensorMeta(hoveredSensor).description}`;
    tag.style.left = `${pointer.x}px`;
    tag.style.top = `${pointer.y}px`;
    tag.hidden = false;
  } else {
    tag.hidden = true;
  }

  const layer = $("callout-layer");
  const card = $("callout");
  if (!state.selectedSensor || card.hidden) {
    layer.replaceChildren();
    return;
  }
  const p = view.screenPosition(state.selectedSensor);
  const stage = $("stage").getBoundingClientRect();
  if (!p || !p.visible) { layer.replaceChildren(); return; }

  const cw = card.offsetWidth, ch = card.offsetHeight;
  const leftSide = p.x > stage.width / 2;
  const elbowX = leftSide ? p.x - 70 : p.x + 70;
  let cy = Math.min(Math.max(p.y - 120, 12), stage.height - ch - 12);
  let cx = leftSide ? elbowX - 40 - cw : elbowX + 40;
  cx = Math.min(Math.max(cx, 12), stage.width - cw - 12);
  card.style.left = `${cx}px`;
  card.style.top = `${cy}px`;

  const anchorX = leftSide ? cx + cw : cx;
  const anchorY = cy + Math.min(ch / 2, 40);
  layer.innerHTML = `
    <path d="M${p.x},${p.y} L${elbowX},${anchorY} L${anchorX},${anchorY}"/>
    <circle cx="${p.x}" cy="${p.y}" r="4"/>`;
});

function renderCallout() {
  const card = $("callout");
  const column = state.selectedSensor;
  if (!column || !state.engine) { card.hidden = true; return; }
  const meta = sensorMeta(column);
  const i = state.cycleIndex;
  const severity = severityAt(column, i);
  const color = SEVERITY_COLORS[severity];

  if (!meta.monitored) {
    card.innerHTML = `
      <div class="callout-head"><span class="callout-symbol">${meta.symbol}</span>
        <span class="callout-state" style="color:var(--graphite)">${SEVERITY_TEXT.unmonitored}</span></div>
      <p class="callout-desc">${meta.description}</p>
      <div class="callout-reading"><span>Reading</span>
        <b>${formatValue(state.engine.flat_sensors[column])} ${meta.units}</b></div>`;
  } else {
    const s = state.engine.sensors[column];
    card.innerHTML = `
      <div class="callout-head"><span class="callout-symbol">${meta.symbol}</span>
        <span class="callout-state" style="color:${color}">${SEVERITY_TEXT[severity]}</span></div>
      <p class="callout-desc">${meta.description}</p>
      ${driftBar(s.drift[i])}
      <div class="callout-reading"><span>Reading at cycle ${state.engine.cycles[i]}</span>
        <b>${formatValue(s.readings[i])} ${meta.units}</b></div>`;
  }
  card.hidden = false;
}

// ------------------------------------------------------------- header ----

function renderModelSwitch() {
  const box = $("model-switch");
  box.replaceChildren();
  for (const model of state.meta.models) {
    const b = document.createElement("button");
    b.type = "button";
    b.role = "radio";
    b.textContent = MODEL_LABELS[model] ?? model;
    b.setAttribute("aria-checked", String(model === state.model));
    b.addEventListener("click", () => setModel(model));
    box.appendChild(b);
  }
}

// The slider spans what the model actually outputs. A fixed 0.05-0.95 range
// had dead zones: XGBoost never goes below 0.06 or above 0.93, so both ends
// flagged every engine or none.
function fitThresholdToModel() {
  const probs = state.fleet.map((e) => e.probability);
  const lo = Math.max(0.01, Math.floor(Math.min(...probs) * 100) / 100);
  const hi = Math.min(0.99, Math.ceil(Math.max(...probs) * 100) / 100);
  const input = $("threshold");
  input.min = lo;
  input.max = hi;
  state.threshold = Math.min(Math.max(state.threshold, lo), hi);
  input.value = state.threshold;
  $("threshold-value").textContent = state.threshold.toFixed(2);
  $("threshold-hint").textContent =
    `${MODEL_LABELS[state.model]} scores this fleet between ${lo.toFixed(2)} and ${hi.toFixed(2)}, so the alert slider covers that range.`;
}

function renderReport() {
  const m = state.meta;
  const rows = m.metrics.map((r) => `
    <tr><td>${MODEL_LABELS[r.model] ?? r.model}</td><td>${r.split}</td>
      <td>${r.accuracy.toFixed(4)}</td><td>${r.precision.toFixed(4)}</td>
      <td>${r.recall.toFixed(4)}</td><td>${r.f1.toFixed(4)}</td><td><b>${r.roc_auc.toFixed(4)}</b></td></tr>`).join("");
  $("report-body").innerHTML = `
    <p class="muted">Subset ${m.subset}. A cycle counts as failing when true remaining life is
      ${m.failure_threshold} cycles or fewer. Trained ${new Date(m.trained_at).toLocaleString()}.</p>
    <table class="metrics">
      <thead><tr><th>Model</th><th>Split</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="muted" style="margin-top:0.8rem">Test engines stop well before failure, so only 2.5% of test cycles are
      failing. Accuracy is high for any model on data like that; recall and ROC-AUC are the numbers to read.</p>
    <p class="muted">Not used by the models, because their readings do not vary across the fleet:
      ${m.dropped_cols.join(", ")}.</p>`;
}

// -------------------------------------------------------------- fleet ----

function renderFleet() {
  const alerts = state.fleet.filter((e) => isAlert(e.probability));
  const failing = state.fleet.filter((e) => e.in_failure_window);
  const caught = alerts.filter((e) => e.in_failure_window);
  $("fleet-kpis").innerHTML = `
    <div><dt>Engines monitored</dt><dd>${state.fleet.length}</dd></div>
    <div class="is-crit"><dt>Above alert level</dt><dd>${alerts.length}</dd></div>
    <div><dt>Truly within ${state.meta.failure_threshold} cycles of failure</dt><dd>${failing.length}</dd></div>
    <div><dt>Of those, alerted</dt><dd>${caught.length}<span class="muted"> / ${failing.length}</span></dd></div>`;

  const query = $("fleet-search").value.trim();
  const sort = $("fleet-sort").value;
  let rows = state.fleet.filter((e) => !query || String(e.engine).startsWith(query));
  rows = [...rows].sort((a, b) =>
    sort === "engine" ? a.engine - b.engine :
    sort === "rul" ? a.true_rul - b.true_rul :
    b.probability - a.probability);

  const list = $("fleet-list");
  list.replaceChildren();
  for (const e of rows) {
    const li = document.createElement("li");
    li.className = "fleet-row" + (isAlert(e.probability) ? " is-alert" : "");
    li.role = "option";
    li.tabIndex = e.engine === state.selectedEngine ? 0 : -1;
    li.dataset.engine = e.engine;
    li.setAttribute("aria-selected", String(e.engine === state.selectedEngine));
    li.innerHTML = `
      <span class="fleet-id">#${e.engine}<small>${e.true_rul} left</small></span>
      <span class="prob">
        <span class="prob-track"><span class="prob-fill" style="width:${e.probability * 100}%"></span></span>
        <span class="prob-text">${(e.probability * 100).toFixed(1)}% risk</span>
      </span>
      <span class="dots" title="${e.critical_sensors} sensors near failure level, ${e.warning_sensors} drifting">
        <i class="dot" style="background:var(--crit)"></i>${e.critical_sensors}
        <i class="dot" style="background:var(--warn);margin-left:4px"></i>${e.warning_sensors}
      </span>`;
    li.addEventListener("click", () => selectEngine(e.engine));
    list.appendChild(li);
  }
}

$("fleet-list").addEventListener("keydown", (event) => {
  if (!["ArrowDown", "ArrowUp", "Enter"].includes(event.key)) return;
  const rows = [...$("fleet-list").children];
  const current = rows.indexOf(document.activeElement);
  if (event.key === "Enter" && current >= 0) {
    selectEngine(Number(rows[current].dataset.engine));
    return;
  }
  event.preventDefault();
  const next = rows[Math.min(Math.max(current + (event.key === "ArrowDown" ? 1 : -1), 0), rows.length - 1)];
  next?.focus();
});

// ------------------------------------------------------------ stations ----

function renderStations() {
  const box = $("stations");
  box.replaceChildren();
  const severities = severitiesAt(state.cycleIndex);
  for (const [module, label] of MODULE_ORDER) {
    const sensors = state.meta.sensors.filter((s) => s.module === module);
    if (!sensors.length) continue;
    const li = document.createElement("li");
    li.className = "station";
    li.innerHTML = `<div class="station-name">${label}</div><div class="station-chips"></div>`;
    li.addEventListener("pointerenter", () => view.highlightModule(module));
    li.addEventListener("pointerleave", () => view.highlightModule(null));
    const chips = li.querySelector(".station-chips");
    for (const s of sensors) {
      const severity = severities[s.column];
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip" + (severity === "unmonitored" ? " is-unmon" : "");
      b.style.setProperty("--c", SEVERITY_COLORS[severity]);
      b.setAttribute("aria-pressed", String(s.column === state.selectedSensor));
      b.title = `${s.description}: ${SEVERITY_TEXT[severity]}`;
      b.innerHTML = `<i></i>${s.symbol}`;
      b.addEventListener("click", () => selectSensor(s.column === state.selectedSensor ? null : s.column));
      chips.appendChild(b);
    }
    box.appendChild(li);
  }
}

// ------------------------------------------------------- viewer header ----

function renderViewerHead() {
  const e = state.engine;
  const i = state.cycleIndex;
  const p = probabilityAt(i);
  const alert = isAlert(p);
  const severities = Object.values(severitiesAt(i));
  const red = severities.filter((s) => s === "critical").length;
  const amber = severities.filter((s) => s === "warning").length;
  $("engine-title").textContent = `Engine ${e.engine}`;
  $("engine-sub").textContent =
    `${red} sensors near failure level, ${amber} drifting. Scored by ${MODEL_LABELS[state.model]}.`;
  $("gauge").innerHTML = `
    <div class="${alert ? "status-alert" : "status-ok"}"><dt>Failure risk</dt><dd>${(p * 100).toFixed(1)}%</dd></div>
    <div><dt>True life left</dt><dd>${e.true_rul[i]} cycles</dd></div>
    <div class="${alert ? "status-alert" : "status-ok"}"><dt>Status</dt><dd>${alert ? "Inspect" : "Normal"}</dd></div>`;
  $("cycle-readout").textContent = `Cycle ${e.cycles[i]} of ${e.cycles[e.cycles.length - 1]}`;
  $("cycle").value = i;
}

// -------------------------------------------------------------- detail ----

function failureSpans() {
  const e = state.engine;
  const start = e.in_failure_window.indexOf(true);
  return start < 0 ? [] : [{ from: e.cycles[start], to: e.cycles[e.cycles.length - 1], color: "var(--crit-wash)" }];
}

function renderEngineDetail() {
  const e = state.engine;
  const i = state.cycleIndex;
  const p = probabilityAt(i);
  const alert = isAlert(p);
  const detail = $("detail");
  detail.innerHTML = `
    <h2>Engine ${e.engine}<span class="state-pill" style="background:${alert ? "var(--crit)" : "var(--ok)"}">
      ${alert ? "Inspect" : "Normal"}</span></h2>
    <p class="muted">Recorded test trajectory, cycle ${e.cycles[i]}.</p>
    <dl class="facts">
      <div><dt>Failure risk</dt><dd>${(p * 100).toFixed(1)}%</dd></div>
      <div><dt>True life left</dt><dd>${e.true_rul[i]}</dd></div>
      <div><dt>Alert level</dt><dd>${(state.threshold * 100).toFixed(0)}%</dd></div>
    </dl>
    <h3>Failure risk over the engine's run</h3>
    <div id="prob-chart"></div>
    <p class="chart-caption">Red band is where true remaining life is ${state.meta.failure_threshold} cycles or fewer, the
      window the model is meant to catch. Dashed line is the alert level.</p>
    <h3>What is driving this score</h3>
    <div id="shap-bars"><p class="muted">Calculating sensor contributions</p></div>
    <p class="chart-caption" id="shap-caption"></p>
    <h3>All sensors at this cycle</h3>
    <table class="sensor-table">
      <thead><tr><th>Sensor</th><th>Reading</th><th class="mini">Drift</th></tr></thead>
      <tbody id="sensor-rows"></tbody>
    </table>`;

  lineChart($("prob-chart"), {
    x: e.cycles, y: e.probability, color: "var(--ink)", yDomain: [0, 1], marker: i,
    refs: [{ y: state.threshold, label: "Alert level", color: "var(--crit)" }],
    spans: failureSpans(), format: (v) => `${(v * 100).toFixed(0)}%`,
    label: "Failure probability by cycle",
  });

  const rows = $("sensor-rows");
  const order = state.meta.sensors
    .filter((s) => s.monitored)
    .sort((a, b) => e.sensors[b.column].drift[i] - e.sensors[a.column].drift[i]);
  for (const s of [...order, ...state.meta.sensors.filter((x) => !x.monitored)]) {
    const severity = severityAt(s.column, i);
    const tr = document.createElement("tr");
    tr.tabIndex = 0;
    const reading = s.monitored ? e.sensors[s.column].readings[i] : e.flat_sensors[s.column];
    const drift = s.monitored ? e.sensors[s.column].drift[i] : null;
    tr.innerHTML = `
      <td><span class="sensor-name" style="--c:${SEVERITY_COLORS[severity]}"><i></i>${s.symbol}
        <span class="muted">${s.description}</span></span></td>
      <td>${formatValue(reading)}</td>
      <td class="mini">${drift == null ? '<span class="muted">Not used</span>' : driftBar(drift).replace(/<div class="drift-scale">.*?<\/div>/s, "")}</td>`;
    tr.addEventListener("click", () => selectSensor(s.column));
    tr.addEventListener("keydown", (ev) => { if (ev.key === "Enter") selectSensor(s.column); });
    rows.appendChild(tr);
  }
  renderShap();
}

function renderSensorDetail() {
  const column = state.selectedSensor;
  const meta = sensorMeta(column);
  const e = state.engine;
  const i = state.cycleIndex;
  const severity = severityAt(column, i);
  const detail = $("detail");
  const back = `<button class="back" id="back" type="button">Engine ${e.engine} overview</button>`;

  if (!meta.monitored) {
    detail.innerHTML = `${back}
      <h2>${meta.symbol}</h2>
      <p>${meta.description} (${meta.units})</p>
      <dl class="facts"><div><dt>Reading</dt><dd>${formatValue(e.flat_sensors[column])}</dd></div></dl>
      <p class="muted">This channel reads the same value across the entire fleet, so the variance check in the data
        pipeline excluded it. A sensor that never changes cannot warn about anything.</p>`;
    $("back").addEventListener("click", () => selectSensor(null));
    return;
  }

  const s = e.sensors[column];
  const trend = meta.rises_with_wear ? "rises" : "falls";
  detail.innerHTML = `${back}
    <h2>${meta.symbol}<span class="state-pill" style="background:${SEVERITY_COLORS[severity]}">${SEVERITY_TEXT[severity]}</span></h2>
    <p>${meta.description}</p>
    <dl class="facts">
      <div><dt>Reading (${meta.units})</dt><dd>${formatValue(s.readings[i])}</dd></div>
      <div><dt>Drift</dt><dd>${(s.drift[i] * 100).toFixed(0)}%</dd></div>
      <div><dt>Cycle</dt><dd>${e.cycles[i]}</dd></div>
    </dl>
    <div style="margin-top:0.6rem">${driftBar(s.drift[i])}</div>
    <p class="chart-caption">Drift places the smoothed reading between this sensor's healthy level (0%) and the level
      engines reach in their last 10 cycles (100%), both measured on training engines only.</p>

    <h3>Reading over the engine's run</h3>
    <div id="reading-chart"></div>
    <p class="chart-caption">On a degrading engine this sensor ${trend}. Dashed lines mark the healthy and failure levels.</p>

    <h3>Drift towards failure level</h3>
    <div id="drift-chart"></div>

    <h3>Effect on the failure score</h3>
    <p id="sensor-shap" class="muted">Calculating</p>
    ${meta.informative ? "" : `<p class="muted">Healthy-to-failure change is only ${meta.signal_to_noise}Ã— this sensor's
      normal noise, so its colour stays grey rather than presenting noise as a finding.</p>`}`;
  $("back").addEventListener("click", () => selectSensor(null));

  lineChart($("reading-chart"), {
    x: e.cycles, y: s.readings, color: "var(--cobalt)", marker: i,
    refs: [
      { y: s.healthy_level, label: "Healthy level", color: "var(--ok)" },
      { y: s.failure_level, label: "Failure level", color: "var(--crit)" },
    ],
    spans: failureSpans(), label: `${meta.symbol} reading by cycle`,
  });

  const { warning, critical } = state.meta.severity_levels;
  const lo = Math.min(-0.3, ...s.drift), hi = Math.max(1.3, ...s.drift);
  lineChart($("drift-chart"), {
    x: e.cycles, y: s.drift, color: "var(--ink)", marker: i, yDomain: [lo, hi],
    bands: [
      { from: lo, to: warning, color: "var(--ok-wash)" },
      { from: warning, to: critical, color: "var(--warn-wash)" },
      { from: critical, to: hi, color: "var(--crit-wash)" },
    ],
    format: (v) => `${(v * 100).toFixed(0)}%`, label: `${meta.symbol} drift by cycle`,
  });
  renderShap();
}

// SHAP is fetched per cycle; scrubbing debounces so a drag sends one request.
let explainTimer = null;
let explainSeq = 0;
function requestExplain() {
  clearTimeout(explainTimer);
  explainTimer = setTimeout(async () => {
    const seq = ++explainSeq;
    const cycle = state.engine.cycles[state.cycleIndex];
    try {
      const result = await api(`/api/engine/${state.engine.engine}/explain?model=${state.model}&cycle=${cycle}`);
      if (seq !== explainSeq) return;
      state.explain = result;
      renderShap();
    } catch (err) {
      if (seq === explainSeq) state.explain = { error: err.message };
      renderShap();
    }
  }, state.playing ? 400 : 120);
}

function renderShap() {
  const x = state.explain;
  const cycle = state.engine.cycles[state.cycleIndex];
  const fresh = x && !x.error && x.engine === state.engine.engine && x.cycle === cycle && x.model === state.model;

  const bars = $("shap-bars");
  if (bars) {
    if (x?.error) bars.innerHTML = `<p class="muted">Could not explain this cycle: ${x.error}</p>`;
    else if (!fresh) bars.innerHTML = `<p class="muted">Calculating sensor contributions</p>`;
    else {
      const top = x.contributions.slice(0, 8).map((c) => ({
        label: symbolOf(c.feature), value: c.shap,
        title: `${sensorMeta(c.feature)?.description ?? c.feature}, reading ${formatValue(c.reading)}`,
      }));
      divergingBars(bars, top, { positive: "var(--crit)", negative: "var(--ok)", format: (v) => (v >= 0 ? "+" : "") + v.toFixed(3) });
      $("shap-caption").textContent = x.units === "log-odds"
        ? `Red pushes towards failure, green away. XGBoost works in log-odds: baseline ${x.base_value.toFixed(3)} plus these bars gives ${x.output.toFixed(3)}, which is ${(x.probability * 100).toFixed(1)}% once converted.`
        : `Red pushes towards failure, green away. Random Forest works in probability: baseline ${x.base_value.toFixed(3)} plus these bars gives ${x.output.toFixed(3)}, the score shown.`;
    }
  }

  const line = $("sensor-shap");
  if (line && state.selectedSensor) {
    if (!fresh) line.textContent = x?.error ? `Could not explain this cycle: ${x.error}` : "Calculating";
    else {
      const rank = x.contributions.findIndex((c) => c.feature === state.selectedSensor);
      const c = x.contributions[rank];
      if (!c) line.textContent = "This sensor is not an input to the model.";
      else {
        const dir = c.shap >= 0 ? "towards failure" : "away from failure";
        line.innerHTML = `At cycle ${x.cycle} this sensor moved ${MODEL_LABELS[state.model]}'s score
          <b>${c.shap >= 0 ? "+" : ""}${c.shap.toFixed(3)} ${x.units}</b> ${dir}, ranking ${rank + 1} of
          ${x.contributions.length} inputs by size of effect.`;
        line.classList.remove("muted");
      }
    }
  }
}

function renderDetail() {
  if (state.selectedSensor) renderSensorDetail();
  else renderEngineDetail();
}

// -------------------------------------------------------------- actions ----

function renderCycle({ detail = true } = {}) {
  view.setSeverities(severitiesAt(state.cycleIndex));
  renderViewerHead();
  renderStations();
  renderCallout();
  if (detail) renderDetail();
  requestExplain();
}

async function selectEngine(engine) {
  stopReplay();
  state.selectedEngine = engine;
  state.selectedSensor = null;
  view.select(null);
  try {
    state.engine = await api(`/api/engine/${engine}?model=${state.model}`);
  } catch (err) {
    showBanner(`Could not load engine ${engine}: ${err.message}`);
    return;
  }
  state.cycleIndex = state.engine.cycles.length - 1;
  $("cycle").max = state.cycleIndex;
  state.explain = null;
  renderFleet();
  renderCycle();
  updateHash();
}

// The URL always names what is on screen, so a view can be bookmarked or shared.
function updateHash() {
  const parts = [`engine=${state.selectedEngine}`];
  if (state.selectedSensor) parts.push(`sensor=${state.selectedSensor}`);
  history.replaceState(null, "", `#${parts.join("&")}`);
}

function selectSensor(column) {
  state.selectedSensor = column;
  updateHash();
  view.select(column);
  renderStations();
  renderCallout();
  renderDetail();
}

async function setModel(model) {
  if (model === state.model) return;
  state.model = model;
  renderModelSwitch();
  await loadFleet();
  const sensor = state.selectedSensor;
  await selectEngine(state.selectedEngine);
  if (sensor) selectSensor(sensor);
}

async function loadFleet() {
  const result = await api(`/api/fleet?model=${state.model}`);
  state.fleet = result.engines;
  fitThresholdToModel();
  renderFleet();
}

// ---------------------------------------------------------------- replay ----

let replayTimer = null;
function startReplay() {
  if (state.cycleIndex >= state.engine.cycles.length - 1) state.cycleIndex = 0;
  state.playing = true;
  $("play-icon").setAttribute("d", "M4 2.5h3v11H4zM9 2.5h3v11H9z");
  $("play").setAttribute("aria-label", "Pause replay");
  replayTimer = setInterval(() => {
    if (state.cycleIndex >= state.engine.cycles.length - 1) { stopReplay(); renderDetail(); return; }
    state.cycleIndex += 1;
    // Rebuilding the detail charts every frame is wasteful; refresh them
    // a few times a second instead.
    renderCycle({ detail: state.cycleIndex % 5 === 0 });
  }, 1000 / REPLAY_FPS);
}
function stopReplay() {
  state.playing = false;
  clearInterval(replayTimer);
  $("play-icon").setAttribute("d", "M4 2.5v11l9-5.5z");
  $("play").setAttribute("aria-label", "Replay recorded cycles");
}

$("play").addEventListener("click", () => (state.playing ? stopReplay() : startReplay()));
$("cycle").addEventListener("input", (e) => {
  stopReplay();
  state.cycleIndex = Number(e.target.value);
  renderCycle({ detail: false });
});
$("cycle").addEventListener("change", () => renderDetail());
$("to-latest").addEventListener("click", () => {
  stopReplay();
  state.cycleIndex = state.engine.cycles.length - 1;
  renderCycle();
});

// --------------------------------------------------------------- wiring ----

$("threshold").addEventListener("input", (e) => {
  state.threshold = Number(e.target.value);
  $("threshold-value").textContent = state.threshold.toFixed(2);
  renderFleet();
  renderViewerHead();
  renderDetail();
});
$("fleet-search").addEventListener("input", renderFleet);
$("fleet-sort").addEventListener("change", renderFleet);
$("reset-view").addEventListener("click", () => selectSensor(null));
$("callout").addEventListener("click", () => $("detail").scrollTo({ top: 0, behavior: "smooth" }));
$("open-report").addEventListener("click", () => $("report").showModal());
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && state.selectedSensor && !$("report").open) selectSensor(null);
});
let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => state.engine && renderDetail(), 150);
});

// ------------------------------------------------------------ live sync ----

function setSync(kind, text) {
  $("sync").className = `sync ${kind}`;
  $("sync-text").textContent = text;
}

async function loadAll({ keepSelection = false } = {}) {
  state.meta = await api("/api/meta");
  state.token = state.meta.token;
  view.setSensors(state.meta.sensors);
  renderModelSwitch();
  renderReport();
  $("fleet-caption").textContent =
    `NASA C-MAPSS ${state.meta.subset} test fleet, recorded flights replayed from the dataset`;
  if (state.meta.quick_run) {
    showBanner("These models came from a <code>--quick</code> smoke run and their scores are not meaningful. Run <code>python -m src.train_classifier</code> for real models.");
  } else {
    showBanner("");
  }
  await loadFleet();

  const hash = new URLSearchParams(location.hash.slice(1));
  const fromHash = Number(hash.get("engine"));
  const engine = keepSelection && state.selectedEngine ? state.selectedEngine
    : state.fleet.some((e) => e.engine === fromHash) ? fromHash : state.fleet[0].engine;
  const hashSensor = state.meta.sensors.some((s) => s.column === hash.get("sensor")) ? hash.get("sensor") : null;
  const sensor = keepSelection ? state.selectedSensor : hashSensor;
  await selectEngine(engine);
  if (sensor) selectSensor(sensor);
  setSync("is-live", `Live, models trained ${new Date(state.meta.trained_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}`);
}

async function poll() {
  try {
    const s = await api("/api/signature");
    if (!s.ready) {
      setSync("is-down", "No trained models");
      showBanner("No trained models found. Train them with <code>python -m src.train_classifier</code>; this page reloads by itself when they appear.");
      state.token = null;
    } else if (s.token !== state.token) {
      const first = state.token === null && !state.meta;
      await loadAll({ keepSelection: true });
      if (!first) toast("Models were retrained. Every panel now shows the new scores.");
    } else if ($("sync").classList.contains("is-down")) {
      setSync("is-live", `Live, models trained ${new Date(state.meta.trained_at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}`);
      showBanner("");
    }
  } catch {
    setSync("is-down", "Server unreachable");
    showBanner("Lost contact with the model server. Start it with <code>python -m api.server</code>; this page reconnects by itself.");
  }
}

(async function start() {
  try {
    await loadAll();
  } catch (err) {
    setSync("is-down", err.status === 503 ? "No trained models" : "Server unreachable");
    showBanner(err.status === 503
      ? "No trained models found. Train them with <code>python -m src.train_classifier</code>; this page reloads by itself when they appear."
      : "Could not reach the model server. Start it with <code>python -m api.server</code> and reload.");
  }
  setInterval(poll, POLL_MS);
})();
