import { EngineView, SEVERITY_COLORS, MODULE_ORDER } from "./engine3d.js";
import { lineChart, divergingBars, formatValue } from "./charts.js";
import { SpaceView, AXIS_LABEL_POSITIONS, lifeColor } from "./vectorspace.js";
import { NetworkView } from "./network.js";

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
  viewMode: "engine",      // engine | space | network
  space: null,             // /api/space, loaded on first visit to the tab
  engineSpace: null,       // /api/engine/:id/space for the selected engine
  network: null,           // /api/network structure
  activations: null,       // /api/engine/:id/network at the current cycle
  simStatus: null,         // /api/sim/status, polled every second
  simRanges: null,         // training range of each channel, for overrides
  alerts: null,            // /api/alerts
  alertsSeen: 0,           // newest alert id already announced
  alertsAcked: 0,          // newest alert id the user has looked at
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
const SIM = "sim";
const isSim = () => state.selectedEngine === SIM;
const engineName = (e = state.engine) => (e?.engine === SIM ? "Simulated engine" : `Engine ${e?.engine}`);
const truthLabel = () => state.engine?.truth_label ?? "True life left";
// The simulated engine uses the server's alert rule (5-cycle average or LSTM),
// so the pill on screen always agrees with the email that was or was not sent.
const alertAt = (i) => (isSim() ? state.engine.sim?.alert_state === "alert" : isAlert(probabilityAt(i)));
const hasRul = () => Array.isArray(state.engine?.predicted_rul);
// The LSTM is trained on RUL capped at 125, so its top value means
// "125 or more", not exactly 125.
const lifeText = (v) => (v >= state.meta.rul_ceiling - 0.5 ? `${state.meta.rul_ceiling}+` : `${Math.round(v)}`);
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
      ${m.dropped_cols.join(", ")}.</p>
    ${m.rul ? rulReport(m.rul) : `<h3 style="margin-top:1.2rem">Remaining life (LSTM)</h3>
      <p class="muted">Not trained yet. Run <code>python -m src.train_lstm</code>.</p>`}`;
}

function rulReport(r) {
  const row = (name, split, s) => `<tr><td>${name}</td><td>${split}</td><td>${s.rmse.toFixed(2)}</td>
    <td>${s.mae.toFixed(2)}</td><td><b>${s.nasa_score.toFixed(1)}</b></td><td>${(s.late_fraction * 100).toFixed(0)}%</td></tr>`;
  return `
    <h3 style="margin-top:1.2rem">Remaining life (LSTM)</h3>
    <p class="muted">Predicts cycles left from the last ${r.window_size} cycles of all sensors. Weights from epoch
      ${r.best_epoch} of ${r.epochs_run}, chosen on the validation engines. Trained ${new Date(r.trained_at).toLocaleString()}.</p>
    <table class="metrics">
      <thead><tr><th>Model</th><th>Split</th><th>RMSE</th><th>MAE</th><th>NASA score</th><th>Predicted late</th></tr></thead>
      <tbody>
        ${row("LSTM", "test", r.metrics.LSTM.test)}
        ${row("LSTM", "val (every window)", r.metrics.LSTM.val_all_windows)}
        ${row("Average-life guess", "test", r.metrics.MeanBaseline.test)}
      </tbody>
    </table>
    <p class="muted" style="margin-top:0.8rem">RMSE and MAE are in cycles, on the 100 test engines at their last
      recorded cycle. The NASA score is the PHM 2008 challenge metric: it punishes predicting failure too late
      harder than too early, and lower is better. The average-life guess predicts the same number for every
      engine, so beating it shows the LSTM learned from the sensors.</p>
    ${r.quick_run ? `<p class="muted"><b>This LSTM came from a --quick smoke run; its scores are not meaningful.</b></p>` : ""}`;
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
    sort === "rul" ? (a.predicted_rul ?? a.true_rul) - (b.predicted_rul ?? b.true_rul) :
    b.probability - a.probability);

  const list = $("fleet-list");
  list.replaceChildren();
  const simRow = document.createElement("li");
  simRow.id = "sim-row";
  simRow.role = "option";
  simRow.tabIndex = isSim() ? 0 : -1;
  simRow.dataset.engine = SIM;
  simRow.addEventListener("click", () => selectEngine(SIM));
  list.appendChild(simRow);
  renderSimRow();
  for (const e of rows) {
    const li = document.createElement("li");
    li.className = "fleet-row" + (isAlert(e.probability) ? " is-alert" : "");
    li.role = "option";
    li.tabIndex = e.engine === state.selectedEngine ? 0 : -1;
    li.dataset.engine = e.engine;
    li.setAttribute("aria-selected", String(e.engine === state.selectedEngine));
    li.innerHTML = `
      <span class="fleet-id" title="${e.predicted_rul == null ? "" : `LSTM predicts ${lifeText(e.predicted_rul)} cycles left; `}true value ${e.true_rul}">#${e.engine}<small>${e.predicted_rul == null ? `${e.true_rul} true` : `~${lifeText(e.predicted_rul)} left`}</small></span>
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
    const id = rows[current].dataset.engine;
    selectEngine(id === SIM ? SIM : Number(id));
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
  const alert = alertAt(i);
  const severities = Object.values(severitiesAt(i));
  const red = severities.filter((s) => s === "critical").length;
  const amber = severities.filter((s) => s === "warning").length;
  $("engine-title").textContent = `${engineName(e)}`;
  $("engine-sub").textContent =
    `${red} sensor${red === 1 ? "" : "s"} near failure level, ${amber} drifting. Scored by ${MODEL_LABELS[state.model]}.`;
  $("gauge").innerHTML = `
    <div class="${alert ? "status-alert" : "status-ok"}"><dt>Failure risk</dt><dd>${(p * 100).toFixed(1)}%</dd></div>
    ${hasRul() ? `<div><dt>Predicted life left</dt><dd>${lifeText(e.predicted_rul[i])} cycles</dd></div>` : ""}
    <div><dt>${truthLabel()}</dt><dd>${e.true_rul[i]} cycles</dd></div>
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
  const alert = alertAt(i);
  const detail = $("detail");
  detail.innerHTML = `
    <h2>${engineName(e)}<span class="state-pill" style="background:${alert ? "var(--crit)" : "var(--ok)"}">
      ${alert ? "Inspect" : "Normal"}</span></h2>
    <p class="muted">${isSim() ? "Live simulated engine" : "Recorded test trajectory"}, cycle ${e.cycles[i]}.</p>
    <dl class="facts">
      <div><dt>Failure risk</dt><dd>${(p * 100).toFixed(1)}%</dd></div>
      <div><dt>${truthLabel()}</dt><dd>${e.true_rul[i]}</dd></div>
      ${hasRul()
        ? `<div><dt>Predicted life left</dt><dd>${lifeText(e.predicted_rul[i])}</dd></div>`
        : `<div><dt>Alert level</dt><dd>${(state.threshold * 100).toFixed(0)}%</dd></div>`}
    </dl>
    ${hasRul() ? `
    <h3>Remaining life, predicted and true</h3>
    <div id="rul-chart"></div>
    <p class="chart-caption">Solid line is the LSTM's prediction from the last ${state.meta.rul.window_size} cycles of
      sensor data; dashed line is the true remaining life. The model was trained with life capped at
      ${state.meta.rul_ceiling} cycles, because early wear does not show in the sensors yet. Early in a run it
      predicts just under ${state.meta.rul_ceiling} however long the engine really has, then tracks the true line
      once degradation becomes visible.</p>` : `
    <p class="muted" style="margin-top:1rem">No remaining-life model yet. Train it with
      <code>python -m src.train_lstm</code>; this panel fills in by itself.</p>`}
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

  if (hasRul()) {
    lineChart($("rul-chart"), {
      x: e.cycles, y: e.predicted_rul, color: "var(--cobalt)", marker: i,
      yDomain: [0, Math.max(state.meta.rul_ceiling + 10, ...e.true_rul.slice(0, 1))],
      extra: [{ y: e.true_rul, color: "var(--ink)", dash: "5 4", label: isSim() ? "projected" : "true" }],
      refs: [
        { y: state.meta.rul_ceiling, label: "Model ceiling", color: "var(--graphite)", dash: "2 3" },
        { y: state.meta.failure_threshold, label: "Failure window", color: "var(--crit)" },
      ],
      spans: failureSpans(), seriesLabel: "predicted", format: (v) => `${Math.round(v)}`,
      label: "Predicted and true remaining life by cycle",
    });
  }

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
  const back = `<button class="back" id="back" type="button">${engineName(e)} overview</button>`;

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
    ${meta.informative ? "" : `<p class="muted">Healthy-to-failure change is only ${meta.signal_to_noise}× this sensor's
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
  // The simulation refreshes every second; never wipe an input mid-edit
  const active = document.activeElement;
  if (isSim() && $("detail").contains(active) && active.matches("input")) return;
  if (state.viewMode === "engine" && state.selectedSensor) {
    renderSensorDetail();
    if (isSim()) renderOverride(state.selectedSensor);
    return;
  }
  if (state.viewMode === "space") renderSpaceDetail();
  else if (state.viewMode === "network") renderNetworkDetail();
  else if (state.selectedSensor) renderSensorDetail();
  else renderEngineDetail();
}

// ------------------------------------------------------- vector space ----

const spaceView = new SpaceView($("space-stage"), {
  onHover: (hit) => {
    const tag = $("space-tag");
    if (!hit || !state.space) { tag.hidden = true; return; }
    const sp = state.space;
    tag.textContent = `Training engine ${sp.unit[hit.index]}, cycle ${sp.cycle[hit.index]}: `
      + (sp.life_left[hit.index] === 0 ? "failed at this cycle"
        : `failed ${sp.life_left[hit.index]} cycle${sp.life_left[hit.index] === 1 ? "" : "s"} later`);
    tag.style.left = `${hit.x}px`;
    tag.style.top = `${hit.y}px`;
    tag.hidden = false;
  },
});
spaceView.setAxisLabels([0, 1, 2].map((k) => ({ el: $(`axis-${k}`), pos: AXIS_LABEL_POSITIONS[k] })));

async function ensureSpace() {
  if (!state.space) {
    state.space = await api("/api/space");
    spaceView.setReference(state.space);
    const ev = state.space.explained_variance;
    ev.forEach((v, k) => { $(`axis-${k}`).textContent = `PC${k + 1} (${(v * 100).toFixed(1)}% of variance)`; });
    $("space-note").textContent = `Each dot is one cycle of one of the ${state.space.engines} training engines, `
      + `placed by the LSTM's ${state.space.dims}-number summary of the ${state.meta.rul.window_size} cycles before it. `
      + `The dark line is the selected test engine's path; blue lines join it to its ${state.space.neighbors} nearest neighbours.`;
  }
  if (!state.engineSpace || state.engineSpace.engine !== state.selectedEngine) {
    const engine = state.selectedEngine;
    const data = await api(`/api/engine/${engine}/space`);
    if (engine !== state.selectedEngine) return;      // user moved on meanwhile
    state.engineSpace = data;
    spaceView.setEngine(data);
  }
  spaceView.setCycle(state.cycleIndex);
}

function renderSpaceDetail() {
  const detail = $("detail");
  const es = state.engineSpace, sp = state.space, e = state.engine;
  if (!es || !sp || es.engine !== e.engine) {
    detail.innerHTML = `<h2>Engines like this one</h2><p class="muted">Encoding engine ${e.engine}</p>`;
    return;
  }
  const i = state.cycleIndex;
  const rows = es.neighbors[i].map((ref, j) => {
    const life = sp.life_left[ref];
    return `<tr><td><span class="nb-swatch" style="background:#${lifeColor(life).getHexString()}"></span>
      Engine ${sp.unit[ref]}, cycle ${sp.cycle[ref]}</td><td>${life}</td><td>${es.distances[i][j].toFixed(2)}</td></tr>`;
  }).join("");
  const base = state.meta.rul.metrics.MeanBaseline.test.rmse;
  const kept = sp.explained_variance.reduce((a, b) => a + b, 0);
  detail.innerHTML = `
    <h2>Engines like this one</h2>
    <p class="muted">${engineName(e)} at cycle ${e.cycles[i]}, compared with every cycle of the
      ${sp.engines} training engines, which all ran to failure.</p>
    <dl class="facts">
      <div><dt>Neighbours' average</dt><dd>${lifeText(es.knn_life[i])}</dd></div>
      <div><dt>LSTM predicts</dt><dd>${lifeText(e.predicted_rul[i])}</dd></div>
      <div><dt>${truthLabel()}</dt><dd>${e.true_rul[i]}</dd></div>
    </dl>
    <h3>${sp.neighbors} nearest neighbours</h3>
    <table class="nb-table">
      <thead><tr><th>Training engine at that moment</th><th>Cycles it had left</th><th>Distance</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="chart-caption">Distance is measured in the full ${sp.dims}-number space, not the 3D picture.
      Neighbour averages are capped at ${state.meta.rul_ceiling}, like the LSTM's target.</p>
    <h3>Does nearness mean anything?</h3>
    <p>Averaging the ${sp.neighbors} nearest neighbours' remaining life predicts the 100 test engines with an
      RMSE of <b>${sp.knn_test_rmse} cycles</b>, against ${sp.lstm_test_rmse} for the LSTM itself and
      ${base.toFixed(2)} for guessing the average. Engines the LSTM places close together really are at a
      similar stage of wear.</p>
    <p class="muted">The 3D view is a PCA projection that keeps ${(kept * 100).toFixed(1)}% of the variance. The
      first axis alone holds ${(sp.explained_variance[0] * 100).toFixed(0)}%, so each axis is stretched to the same
      length to keep the cloud readable. Search is exact; an approximate index such as a vector database uses is
      only worth it at millions of vectors, and this is ${sp.n.toLocaleString()}.</p>`;
}

// ----------------------------------------------------- neural network ----

const netView = new NetworkView($("net-stage"), {
  onHover: (hit) => {
    const tag = $("net-tag");
    if (!hit) { tag.hidden = true; return; }
    tag.textContent = hit.text;
    tag.style.left = `${hit.x}px`;
    tag.style.top = `${hit.y}px`;
    tag.hidden = false;
  },
});
$("net-replay").addEventListener("click", () => netView.replayFlow());

async function ensureNetwork() {
  if (!state.network) {
    state.network = await api("/api/network");
    netView.setStructure(state.network);
  }
  requestActivations();
}

let activationTimer = null;
let activationSeq = 0;
function requestActivations() {
  clearTimeout(activationTimer);
  activationTimer = setTimeout(async () => {
    const seq = ++activationSeq;
    const cycle = state.engine.cycles[state.cycleIndex];
    try {
      const acts = await api(`/api/engine/${state.engine.engine}/network?cycle=${cycle}`);
      if (seq !== activationSeq) return;
      state.activations = acts;
      netView.setActivations(acts, { fast: state.playing });
      if (!state.playing) renderNetworkDetail();
    } catch (err) {
      showBanner(`Could not compute activations: ${err.message}`);
    }
  }, state.playing ? 60 : 120);
}

function renderNetworkDetail() {
  const detail = $("detail");
  const n = state.network, a = state.activations, e = state.engine;
  if (!n) { detail.innerHTML = `<h2>Inside the LSTM</h2><p class="muted">Loading the network</p>`; return; }
  const cycle = a ? a.cycle : e.cycles[state.cycleIndex];
  const [l1, l2, d1] = n.layers;
  detail.innerHTML = `
    <h2>Inside the LSTM</h2>
    <p class="muted">${engineName(e)} at cycle ${cycle}. Every coloured neuron is the number that layer really
      computed for this engine; drag the cycle slider or press play to watch it change.</p>
    <dl class="facts">
      <div><dt>Predicted life left</dt><dd>${a ? lifeText(a.predicted_rul) : "–"}</dd></div>
      <div><dt>Trained parameters</dt><dd>${n.parameters.toLocaleString()}</dd></div>
      <div><dt>${truthLabel()}</dt><dd>${e.true_rul[state.cycleIndex]}</dd></div>
    </dl>
    <h3>How a prediction flows</h3>
    <ol class="layer-list">
      <li><b>Input window</b>The last ${n.window_size} cycles of ${n.features.length} sensor and setting values,
        scaled 0 to 1. Rows run from oldest (top) to latest (bottom); the outlined row is the latest cycle.</li>
      <li><b>LSTM, ${l1.units} units</b>Reads the window one cycle at a time and carries a memory forward, so it
        can see a trend, not just a snapshot. Shown: its output after the last cycle.</li>
      <li><b>LSTM, ${l2.units} units</b>Compresses that into ${l2.units} numbers, the model's summary of the
        engine's state. These are the vectors plotted in the Vector space tab.</li>
      <li><b>Dense, ${d1.units} units (ReLU)</b>Combines the summary; negative results are cut to zero.</li>
      <li><b>Output</b>One number between 0 and 1, multiplied by ${n.rul_ceiling} to give cycles left.</li>
    </ol>
    <p class="chart-caption">Lines show each neuron's strongest trained incoming connections: 3 per neuron into
      the first LSTM and 4 into the second, so the structure is readable; the two Dense layers are drawn in full.
      Each LSTM unit has four gates, summed into one strength. Recurrent connections, where a unit feeds itself
      from one cycle to the next, are not drawn.</p>`;
}

// --------------------------------------------------------- view switch ----

function setView(mode) {
  if (mode !== "engine" && !state.meta?.rul) mode = "engine";
  state.viewMode = mode;
  for (const tab of document.querySelectorAll(".view-tabs [role=tab]")) {
    tab.setAttribute("aria-selected", String(tab.dataset.view === mode));
    tab.tabIndex = tab.dataset.view === mode ? 0 : -1;
  }
  $("stage").hidden = mode !== "engine";
  $("stations").hidden = mode !== "engine";
  $("space-stage").hidden = mode !== "space";
  $("net-stage").hidden = mode !== "network";
  if (mode === "space") spaceView.start(); else spaceView.stop();
  updateHash();
  renderDetail();
  if (!state.engine) return;
  if (mode === "space") ensureSpace().then(renderDetail).catch((err) => showBanner(`Vector space: ${err.message}`));
  if (mode === "network") ensureNetwork().catch((err) => showBanner(`Network: ${err.message}`));
}

for (const tab of document.querySelectorAll(".view-tabs [role=tab]")) {
  tab.addEventListener("click", () => setView(tab.dataset.view));
}
document.querySelector(".view-tabs").addEventListener("keydown", (event) => {
  if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  const tabs = [...document.querySelectorAll(".view-tabs [role=tab]:not(:disabled)")];
  const next = tabs[(tabs.indexOf(document.activeElement) + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
  next.focus();
  setView(next.dataset.view);
});
$("space-reset").addEventListener("click", () => spaceView.resetView());

function syncViewAvailability() {
  const ready = Boolean(state.meta?.rul);
  for (const id of ["tab-space", "tab-network"]) {
    $(id).disabled = !ready;
    $(id).title = ready ? "" : "Train the LSTM first: python -m src.train_lstm";
  }
}

// -------------------------------------------------------------- actions ----

function renderCycle({ detail = true } = {}) {
  view.setSeverities(severitiesAt(state.cycleIndex));
  renderViewerHead();
  renderStations();
  renderCallout();
  if (state.viewMode === "space" && state.engineSpace?.engine === state.engine.engine) {
    spaceView.setCycle(state.cycleIndex);
  }
  if (state.viewMode === "network" && state.network) requestActivations();
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
  state.activations = null;
  $("sim-bar").hidden = !isSim();
  document.querySelector(".replay").hidden = isSim();
  if (isSim()) renderSimBar();
  renderFleet();
  renderCycle();
  updateHash();
  if (state.viewMode === "space") ensureSpace().then(renderDetail).catch((err) => showBanner(`Vector space: ${err.message}`));
}

// The URL always names what is on screen, so a view can be bookmarked or shared.
function updateHash() {
  const parts = [`engine=${state.selectedEngine}`];
  if (state.selectedSensor) parts.push(`sensor=${state.selectedSensor}`);
  if (state.viewMode !== "engine") parts.push(`view=${state.viewMode}`);
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
  syncAlertRule();
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
  syncAlertRule();
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

// ------------------------------------------------------------ simulation ----

const SIM_POLL_MS = 1000;

async function apiPost(path, body = {}) {
  const response = await fetch(path, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw Object.assign(new Error(data.error || response.statusText), { status: response.status });
  return data;
}

async function simControl(action, extra = {}) {
  try {
    await apiPost("/api/sim/control", { action, ...extra });
    await pollSim({ force: true });
  } catch (err) {
    toast(`Simulation: ${err.message}`);
  }
}

async function pollSim({ force = false } = {}) {
  if (!state.meta) return;
  try {
    const st = await api(`/api/sim/status?model=${state.model}`);
    const changed = force || !state.simStatus || st.version !== state.simStatus.version;
    state.simStatus = st;
    if (changed) renderSimRow();
    if (st.last_alert_id > state.alertsSeen || $("alerts-dialog").open) await refreshAlerts({ announce: true });
    if (changed && isSim()) await refreshSimEngine();
  } catch {
    /* the main poll reports a lost server */
  }
}

async function refreshSimEngine() {
  const data = await api(`/api/engine/${SIM}?model=${state.model}`);
  if (!isSim()) return;
  state.engine = data;
  state.cycleIndex = data.cycles.length - 1;     // the simulation is always live
  $("cycle").max = state.cycleIndex;
  renderSimBar();
  if (state.viewMode === "space" && state.space) {
    state.engineSpace = null;                      // the path grew; fetch it again
    ensureSpace().then(renderDetail).catch(() => {});
  }
  renderCycle();
}

function renderSimRow() {
  const row = $("sim-row");
  const st = state.simStatus;
  if (!row) return;
  const alert = st?.alert_state === "alert";
  row.className = "fleet-row sim-row" + (alert ? " is-alert" : "") + (st?.running ? " is-running" : "");
  row.setAttribute("aria-selected", String(isSim()));
  const risk = st?.probability == null ? null : st.probability;
  const status = !st ? "Loading" : st.failed ? "Reached end of life" : st.running ? `Running, cycle ${st.cycle}`
    : st.cycle > 1 ? `Paused at cycle ${st.cycle}` : "Ready to start";
  row.innerHTML = `
    <span class="fleet-id"><span class="live-dot"></span>Live<small>${st?.predicted_rul == null ? "simulated" : `~${lifeText(st.predicted_rul)} left`}</small></span>
    <span class="prob">
      <span class="prob-track"><span class="prob-fill" style="width:${(risk ?? 0) * 100}%;${alert ? "" : "background:var(--cobalt)"}"></span></span>
      <span class="prob-text">${risk == null ? "" : `${(risk * 100).toFixed(1)}% risk, `}${status}</span>
    </span>
    <span class="dots">${alert ? '<b style="color:var(--crit)">Alert</b>' : ""}</span>`;
}

function renderSimBar() {
  const sim = state.engine?.sim;
  if (!sim) return;
  $("sim-play-icon").setAttribute("d", sim.running ? "M4 2.5h3v11H4zM9 2.5h3v11H9z" : "M4 2.5v11l9-5.5z");
  $("sim-play").setAttribute("aria-label", sim.running ? "Pause simulation" : sim.failed ? "Start a new engine" : "Start simulation");

  const segment = (box, values, current, label, action) => {
    box.replaceChildren(...values.map((v) => {
      const b = document.createElement("button");
      b.type = "button";
      b.role = "radio";
      b.textContent = label(v);
      b.setAttribute("aria-checked", String(v === current));
      b.addEventListener("click", () => simControl(action, { value: v }));
      return b;
    }));
  };
  segment($("sim-speed"), sim.speeds, sim.speed, (v) => `${v}×`, "speed");
  segment($("sim-wear"), sim.wear_presets, sim.wear, (v) => v[0].toUpperCase() + v.slice(1), "wear");
  $("sim-fault").setAttribute("aria-pressed", String(sim.fault));
  $("sim-fault").textContent = sim.fault ? "HPC fault active" : "Inject HPC fault";

  // Share of the engine's expected life already used, at the current wear
  // rate. Raw damage grows exponentially, so it would read "47%" with only
  // 15 cycles to go.
  const usedPct = sim.failed ? 100 : Math.min(100, (sim.cycle / (sim.cycle + sim.projected_life)) * 100);
  $("sim-wear-fill").style.width = `${usedPct}%`;
  $("sim-status").textContent = sim.failed
    ? `End of life at cycle ${sim.cycle}. Press play to fit a new engine.`
    : `Cycle ${sim.cycle}, ${usedPct.toFixed(0)}% of life used, about ${Math.round(sim.projected_life)} cycles left at this wear rate`;
  const last = sim.events[sim.events.length - 1];
  // Event text names channels by column; show the symbols used everywhere else
  $("sim-event").textContent = last
    ? `Cycle ${last.cycle}: ${last.text.replace(/sensor_\d+/g, (c) => symbolOf(c))}` : "";
}

$("sim-play").addEventListener("click", () => {
  const sim = state.engine?.sim;
  simControl(sim?.running ? "pause" : "start");
});
$("sim-fault").addEventListener("click", () => simControl("fault", { value: !state.engine?.sim?.fault }));
$("sim-reset").addEventListener("click", () => simControl("reset"));
$("sim-demo").addEventListener("click", () => {
  setView("engine");
  simControl("demo");
  toast("Demo started: a healthy engine, then an HPC fault, then the alert. About a minute.");
});

// Manual override panel, shown under a sensor's detail when the simulation is selected
function renderOverride(column) {
  const meta = sensorMeta(column);
  const detail = $("detail");
  if (!meta.monitored) {
    detail.insertAdjacentHTML("beforeend", `<h3>Set this sensor by hand</h3>
      <p class="muted">The models do not use ${meta.symbol}, so changing it would not affect any prediction.</p>`);
    return;
  }
  const r = state.simRanges?.[column];
  if (!r) return;
  const current = state.engine.sim.overrides[column];
  const latest = state.engine.sensors[column].readings[state.cycleIndex];
  const value = current ?? latest;
  const span = r.high - r.low;
  const lo = r.low - span * 0.5, hi = r.high + span * 0.5;
  detail.insertAdjacentHTML("beforeend", `
    <h3>Set this sensor by hand</h3>
    <p class="muted">${current == null ? "The engine is generating this sensor's readings." :
      `<span class="is-overridden">Held at ${formatValue(current)} ${meta.units}.</span>`}
      Pick a value and the simulated engine reads it every cycle, plus normal noise, until you release it.</p>
    <div class="override">
      <input type="range" id="ov-range" min="${lo}" max="${hi}" step="${span / 400}" value="${value}"
        aria-label="${meta.symbol} value">
      <input type="number" id="ov-num" step="${span / 400}" value="${value.toFixed(3)}" aria-label="${meta.symbol} value">
    </div>
    <div class="override-actions">
      <button class="ghost small" id="ov-apply" type="button">Apply</button>
      <button class="ghost small" id="ov-clear" type="button" ${current == null ? "disabled" : ""}>Release</button>
      <button class="ghost small" id="ov-fail" type="button">Set to failure level</button>
    </div>
    <p class="muted" style="margin-top:0.4rem">Healthy level ${formatValue(r.healthy)}, failure level ${formatValue(r.failure)}.
      Training engines recorded between ${formatValue(r.low)} and ${formatValue(r.high)} ${meta.units}.</p>
    <p class="ov-warn" id="ov-warn" hidden>Outside anything in the training data. The models were never shown a value
      like this, so their response to it is not reliable.</p>`);

  const range = $("ov-range"), num = $("ov-num"), warn = $("ov-warn");
  const showWarn = (v) => { warn.hidden = v >= r.low && v <= r.high; };
  showWarn(value);
  range.addEventListener("input", () => { num.value = Number(range.value).toFixed(3); showWarn(Number(range.value)); });
  num.addEventListener("input", () => { range.value = num.value; showWarn(Number(num.value)); });
  const apply = (v) => {
    document.activeElement.blur();
    simControl("override", { column, value: v });
    toast(`${meta.symbol} held at ${formatValue(v)} ${meta.units}`);
  };
  $("ov-apply").addEventListener("click", () => apply(Number(num.value)));
  num.addEventListener("keydown", (e) => { if (e.key === "Enter") apply(Number(num.value)); });
  $("ov-fail").addEventListener("click", () => apply(r.failure));
  $("ov-clear").addEventListener("click", () => {
    simControl("override", { column, value: null });
    toast(`${meta.symbol} released back to simulated readings`);
  });
}

// ---------------------------------------------------------------- alerts ----

let ruleTimer = null;
function syncAlertRule() {
  clearTimeout(ruleTimer);
  ruleTimer = setTimeout(() => apiPost("/api/alerts/rule", { threshold: state.threshold, model: state.model })
    .catch(() => {}), 250);
}

async function refreshAlerts({ announce }) {
  const data = await api("/api/alerts");
  const fresh = data.entries.filter((e) => e.id > state.alertsSeen && e.kind !== "test");
  state.alerts = data;
  if (announce) {
    for (const e of fresh.reverse()) {
      const where = e.status === "not_configured" ? "logged here; email is not set up"
        : `emailing ${e.recipients.join(", ")}`;
      toast(e.kind === "alert" ? `Alert: ${e.engine} needs inspection (${where})` : `${e.engine} is back to normal (${where})`);
    }
  }
  state.alertsSeen = Math.max(state.alertsSeen, data.last_id);
  if (!announce && !state.alertsAcked) state.alertsAcked = data.last_id;
  const unread = data.entries.filter((e) => e.id > state.alertsAcked && e.kind === "alert").length;
  $("alert-badge").textContent = unread;
  $("alert-badge").hidden = unread === 0;
  if ($("alerts-dialog").open) renderAlerts();
}

const STATUS_TEXT = {
  sent: (e) => `<span style="color:var(--ok)">Email sent</span> to ${e.recipients.join(", ")}${e.send_seconds ? ` in ${e.send_seconds}s` : ""}.`,
  sending: () => `Sending email…`,
  failed: (e) => `<span style="color:var(--crit)">Email failed.</span> ${e.error}`,
  not_configured: () => `Logged here only. Email is not set up yet.`,
};

function renderAlerts() {
  const data = state.alerts;
  if (!data) return;
  const em = data.email;
  const box = em.configured
    ? `<div class="email-box"><b>Email alerts are on.</b> Messages go from ${em.sender} to ${em.recipients.join(", ")}
        through ${em.host}:${em.port}.
        <div style="margin-top:0.5rem"><button class="ghost small" id="send-test" type="button">Send test email</button></div></div>`
    : `<div class="email-box is-off"><b>Email is not set up yet.</b> Alerts still appear here and as pop-ups.
        To get them by email, copy <code>.env.example</code> to <code>.env</code> in the project folder and fill in
        ${em.missing.map((m) => `<code>${m}</code>`).join(", ")}. No restart is needed.</div>`;
  const kindColor = { alert: "var(--crit)", recovered: "var(--ok)", test: "var(--cobalt)" };
  const kindText = { alert: "Alert", recovered: "Recovered", test: "Test" };
  const items = data.entries.length ? data.entries.map((e) => `
    <li class="alert-entry">
      <div class="alert-entry-head">
        <span class="alert-kind" style="background:${kindColor[e.kind]}">${kindText[e.kind]}</span>
        <b>${e.engine}</b>${e.cycle ? `<span class="muted">cycle ${e.cycle}</span>` : ""}
        <span class="alert-time">${new Date(e.time).toLocaleString([], { dateStyle: "medium", timeStyle: "medium" })}</span>
      </div>
      ${e.reason ? `<div>${e.reason}</div>` : ""}
      <div class="alert-status">${(STATUS_TEXT[e.status] || (() => e.status))(e)}</div>
      <details><summary>Email: ${e.subject}</summary><pre>${e.body.replace(/</g, "&lt;")}</pre></details>
    </li>`).join("") : `<li class="muted" style="padding:0.6rem 0">No alerts yet. Start the simulated engine, or press Run demo.</li>`;
  $("alerts-body").innerHTML = `${box}
    <p class="muted" style="margin:0.8rem 0 0">${data.rule.text} Each engine alerts once, then again only after it
      has recovered.</p>
    <ol class="alert-list">${items}</ol>`;
  $("send-test")?.addEventListener("click", async () => {
    const entry = await apiPost("/api/alerts/test");
    toast(entry.status === "not_configured" ? "Email is not set up yet" : "Sending test email…");
    await refreshAlerts({ announce: false });
  });
}

$("open-alerts").addEventListener("click", async () => {
  await refreshAlerts({ announce: false });
  state.alertsAcked = state.alerts?.last_id ?? 0;
  $("alert-badge").hidden = true;
  renderAlerts();
  $("alerts-dialog").showModal();
});

// ------------------------------------------------------------ live sync ----

function setSync(kind, text) {
  $("sync").className = `sync ${kind}`;
  $("sync-text").textContent = text;
}

async function loadAll({ keepSelection = false } = {}) {
  state.meta = await api("/api/meta");
  state.token = state.meta.token;
  // A retrain changes the space and the weights, so drop what was cached
  state.space = state.engineSpace = state.network = state.activations = null;
  syncViewAvailability();
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
  state.simRanges = await api("/api/sim/ranges");
  await refreshAlerts({ announce: false });
  syncAlertRule();

  const hash = new URLSearchParams(location.hash.slice(1));
  const fromHash = hash.get("engine") === SIM ? SIM : Number(hash.get("engine"));
  const engine = keepSelection && state.selectedEngine ? state.selectedEngine
    : fromHash === SIM || state.fleet.some((e) => e.engine === fromHash) ? fromHash : state.fleet[0].engine;
  const hashSensor = state.meta.sensors.some((s) => s.column === hash.get("sensor")) ? hash.get("sensor") : null;
  const sensor = keepSelection ? state.selectedSensor : hashSensor;
  await selectEngine(engine);
  if (sensor) selectSensor(sensor);
  const viewParam = keepSelection ? state.viewMode : hash.get("view");
  setView(["space", "network"].includes(viewParam) ? viewParam : "engine");
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
    state.token = null;
    console.error(err);
    setSync("is-down", err.status === 503 ? "No trained models" : "Server unreachable");
    showBanner(err.status === 503
      ? "No trained models found. Train them with <code>python -m src.train_classifier</code>; this page reloads by itself when they appear."
      : "Could not reach the model server. Start it with <code>python -m api.server</code> and reload.");
  }
  setInterval(poll, POLL_MS);
  setInterval(pollSim, SIM_POLL_MS);
})();
