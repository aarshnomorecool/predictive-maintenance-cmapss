// Small SVG chart helpers. No library: every chart here is a line over
// cycles with reference lines and shaded bands, which is a few dozen lines
// of SVG, and it keeps the dashboard working with no network.

const NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}, parent) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (parent) parent.appendChild(node);
  return node;
}

function niceTicks(min, max, count = 4) {
  const span = max - min || 1;
  const step = 10 ** Math.floor(Math.log10(span / count));
  const err = (count * step) / span;
  const mult = err <= 0.15 ? 10 : err <= 0.35 ? 5 : err <= 0.75 ? 2 : 1;
  const nice = step * mult;
  const ticks = [];
  for (let v = Math.ceil(min / nice) * nice; v <= max + 1e-9; v += nice) ticks.push(+v.toFixed(10));
  return ticks;
}

export function formatValue(v, digits) {
  if (!Number.isFinite(v)) return "–";
  const d = digits ?? (Math.abs(v) >= 1000 ? 1 : Math.abs(v) >= 100 ? 2 : 3);
  return v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
}

/**
 * Line chart over cycles.
 *
 * options:
 *   x: number[]                        cycles
 *   y: number[]                        values
 *   color: string
 *   yDomain: [min, max] | undefined    defaults to data range with padding
 *   refs: [{ y, label, color, dash }]  horizontal reference lines
 *   bands: [{ from, to, color }]       horizontal shaded bands (y ranges)
 *   spans: [{ from, to, color, label }] vertical shaded spans (x ranges)
 *   marker: number                     index of the current cycle
 *   format: (v) => string              value formatter for the hover readout
 *   height: number
 */
export function lineChart(container, options) {
  const { x, y, color = "#1f5fad", refs = [], bands = [], spans = [],
          marker, format = (v) => formatValue(v), height = 168 } = options;
  container.replaceChildren();
  const width = Math.max(container.clientWidth, 240);
  const m = { top: 10, right: 12, bottom: 24, left: 52 };
  const w = width - m.left - m.right;
  const h = height - m.top - m.bottom;

  let [lo, hi] = options.yDomain || [Math.min(...y), Math.max(...y)];
  if (!options.yDomain) {
    for (const r of refs) { lo = Math.min(lo, r.y); hi = Math.max(hi, r.y); }
    const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
    lo -= pad; hi += pad;
  }
  const x0 = x[0], x1 = x[x.length - 1] === x0 ? x0 + 1 : x[x.length - 1];
  const sx = (v) => m.left + ((v - x0) / (x1 - x0)) * w;
  const sy = (v) => m.top + (1 - (v - lo) / (hi - lo)) * h;
  const clampY = (v) => Math.min(hi, Math.max(lo, v));

  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, class: "chart", role: "img" });
  if (options.label) el("title", {}, svg).textContent = options.label;

  for (const b of bands) {
    const top = sy(clampY(b.to)), bottom = sy(clampY(b.from));
    el("rect", { x: m.left, y: top, width: w, height: Math.max(0, bottom - top), fill: b.color, class: "chart-band" }, svg);
  }
  for (const s of spans) {
    el("rect", { x: sx(s.from), y: m.top, width: Math.max(0, sx(s.to) - sx(s.from)), height: h, fill: s.color, class: "chart-span" }, svg);
  }

  for (const t of niceTicks(lo, hi)) {
    el("line", { x1: m.left, x2: m.left + w, y1: sy(t), y2: sy(t), class: "chart-grid" }, svg);
    el("text", { x: m.left - 6, y: sy(t) + 4, class: "chart-tick", "text-anchor": "end" }, svg).textContent = format(t);
  }
  for (const t of niceTicks(x0, x1, 5)) {
    el("text", { x: sx(t), y: height - 6, class: "chart-tick", "text-anchor": "middle" }, svg).textContent = t;
  }

  for (const r of refs) {
    el("line", { x1: m.left, x2: m.left + w, y1: sy(r.y), y2: sy(r.y), stroke: r.color, "stroke-dasharray": r.dash || "5 4", class: "chart-ref" }, svg);
    if (r.label) {
      el("text", { x: m.left + w - 4, y: sy(r.y) - 4, class: "chart-ref-label", fill: r.color, "text-anchor": "end" }, svg).textContent = r.label;
    }
  }

  const d = x.map((xv, i) => `${i ? "L" : "M"}${sx(xv).toFixed(1)},${sy(clampY(y[i])).toFixed(1)}`).join("");
  el("path", { d, fill: "none", stroke: color, class: "chart-line" }, svg);

  if (marker != null && marker >= 0 && marker < x.length) {
    const mx = sx(x[marker]);
    el("line", { x1: mx, x2: mx, y1: m.top, y2: m.top + h, class: "chart-marker" }, svg);
    el("circle", { cx: mx, cy: sy(clampY(y[marker])), r: 4, fill: color, class: "chart-dot" }, svg);
  }

  // Hover readout
  const hover = el("g", { class: "chart-hover", visibility: "hidden" }, svg);
  const hLine = el("line", { y1: m.top, y2: m.top + h, class: "chart-hover-line" }, hover);
  const hDot = el("circle", { r: 3.5, fill: color }, hover);
  const hText = el("text", { class: "chart-hover-text", y: m.top + 12 }, hover);
  el("rect", { x: m.left, y: m.top, width: w, height: h, fill: "transparent" }, svg)
    .addEventListener("pointermove", (e) => {
      const rect = svg.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * width;
      const cycle = x0 + ((px - m.left) / w) * (x1 - x0);
      let i = 0;
      while (i < x.length - 1 && Math.abs(x[i + 1] - cycle) < Math.abs(x[i] - cycle)) i++;
      const hx = sx(x[i]);
      hLine.setAttribute("x1", hx); hLine.setAttribute("x2", hx);
      hDot.setAttribute("cx", hx); hDot.setAttribute("cy", sy(clampY(y[i])));
      hText.textContent = `Cycle ${x[i]}: ${format(y[i])}`;
      const right = hx > m.left + w * 0.6;
      hText.setAttribute("x", right ? hx - 8 : hx + 8);
      hText.setAttribute("text-anchor", right ? "end" : "start");
      hover.setAttribute("visibility", "visible");
    });
  svg.lastChild.addEventListener("pointerleave", () => hover.setAttribute("visibility", "hidden"));

  container.appendChild(svg);
  return svg;
}

/**
 * Signed horizontal bars, centred on zero. Used for SHAP contributions.
 * rows: [{ label, value, title, highlight }]
 */
export function divergingBars(container, rows, { positive, negative, format } = {}) {
  container.replaceChildren();
  const max = Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9);
  const list = document.createElement("ol");
  list.className = "bars";
  for (const r of rows) {
    const li = document.createElement("li");
    li.className = "bars-row" + (r.highlight ? " is-highlight" : "");
    if (r.title) li.title = r.title;
    const pct = (Math.abs(r.value) / max) * 50;
    const side = r.value >= 0 ? "pos" : "neg";
    li.innerHTML = `
      <span class="bars-label">${r.label}</span>
      <span class="bars-track">
        <span class="bars-fill bars-${side}" style="width:${pct}%;
          ${side === "pos" ? "left:50%" : `left:${50 - pct}%`};
          background:${side === "pos" ? positive : negative}"></span>
      </span>
      <span class="bars-value">${format ? format(r.value) : r.value.toFixed(3)}</span>`;
    list.appendChild(li);
  }
  container.appendChild(list);
}
