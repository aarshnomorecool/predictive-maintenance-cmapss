// The trained LSTM drawn as a network, with real activations flowing through.
//
// Columns, left to right: the 30 x 17 input window as a heatmap, the 17
// features at the latest cycle, LSTM (64), LSTM (32), Dense (16) and the
// output neuron. Line opacity is the trained connection strength from
// /api/network; neuron colour is the value that layer actually computed for
// the selected engine and cycle, from /api/engine/:id/network.

const POS = [31, 95, 173];     // cobalt, for positive values
const NEG = [224, 138, 0];     // amber, for negative values
const INK = [27, 38, 49];

// Lines drawn per target neuron for the wide layers, strongest first. All
// 1,088 + 2,048 connections at once is a grey smear; the strongest few per
// neuron show the structure.
const TOP_PER_TARGET = [3, 4, 32, 16];

const rgba = ([r, g, b], a) => `rgba(${r},${g},${b},${a})`;

function valueColor(v, max = 1) {
  const t = Math.min(Math.abs(v) / max, 1);
  return rgba(v >= 0 ? POS : NEG, 0.12 + 0.88 * t);
}

export class NetworkView {
  constructor(container, { onHover } = {}) {
    this.container = container;
    this.onHover = onHover || (() => {});
    this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.canvas = document.createElement("canvas");
    this.canvas.setAttribute("role", "img");
    this.canvas.setAttribute("aria-label", "Diagram of the LSTM network and its activations");
    container.prepend(this.canvas);
    this.ctx = this.canvas.getContext("2d");
    this.structure = null;
    this.current = null;      // activations being shown
    this.previous = null;     // activations before the latest update, for the sweep
    this.flowStart = -Infinity;
    this.flowDuration = 1400;
    this.layout = null;
    this.canvas.addEventListener("pointermove", (e) => this.hover(e));
    this.canvas.addEventListener("pointerleave", () => this.onHover(null));
    new ResizeObserver(() => { this.computeLayout(); this.draw(); }).observe(container);
  }

  setStructure(structure) {
    this.structure = structure;
    // Precompute the strongest incoming connections per target neuron
    this.edges = structure.layers.map((layer, li) => {
      const [nIn, nOut] = layer.shape;
      const keep = TOP_PER_TARGET[li] ?? nIn;
      const list = [];
      for (let j = 0; j < nOut; j++) {
        const incoming = [];
        for (let i = 0; i < nIn; i++) incoming.push([i, layer.weights[i * nOut + j]]);
        incoming.sort((a, b) => b[1] - a[1]);
        for (const [i, w] of incoming.slice(0, keep)) list.push({ i, j, w });
      }
      return list;
    });
    this.computeLayout();
    this.draw();
  }

  // fast: replay is running, so skip the long sweep
  setActivations(acts, { fast = false } = {}) {
    this.previous = this.current;
    this.current = acts;
    this.flowDuration = fast || this.reducedMotion ? 0 : 1400;
    this.flowStart = performance.now();
    this.animate();
  }

  replayFlow() {
    if (!this.current) return;
    this.previous = null;
    this.flowDuration = this.reducedMotion ? 0 : 1600;
    this.flowStart = performance.now();
    this.animate();
  }

  animate() {
    if (this.raf) return;
    const tick = () => {
      this.draw();
      if (performance.now() - this.flowStart < this.flowDuration) {
        this.raf = requestAnimationFrame(tick);
      } else {
        this.raf = null;
        this.draw();
      }
    };
    this.raf = requestAnimationFrame(tick);
  }

  computeLayout() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h || !this.structure) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.canvas.style.width = `${w}px`;
    this.canvas.style.height = `${h}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const top = 64, bottom = h - 54;
    const sizes = [this.structure.features.length, ...this.structure.layers.map((l) => l.units)];
    // heatmap | features | 64 | 32 | 16 | 1
    const heatW = Math.min(w * 0.17, 150);
    const left = 24 + heatW + 92;
    const right = w - 70;
    const xs = sizes.map((_, k) => left + ((right - left) * k) / (sizes.length - 1));
    const columns = sizes.map((n, k) => {
      const span = Math.min(bottom - top, n * 26);
      const y0 = top + (bottom - top - span) / 2;
      const r = Math.max(2.4, Math.min(11, (span / Math.max(n, 1)) * 0.36));
      return Array.from({ length: n }, (_, i) => ({
        x: xs[k], y: n === 1 ? (top + bottom) / 2 : y0 + (span * (i + 0.5)) / n, r: k === sizes.length - 1 ? 24 : r,
      }));
    });
    this.layout = { w, h, top, bottom, heat: { x: 24, y: top, w: heatW, h: bottom - top }, columns, xs };
  }

  // Values per column: features (latest cycle), then each layer's output.
  columnValues(acts) {
    if (!acts) return null;
    const nf = acts.input_shape[1];
    const latest = acts.input.slice(acts.input.length - nf);
    return [latest, ...acts.layers.map((l) => l.values)];
  }

  draw() {
    const L = this.layout;
    const ctx = this.ctx;
    if (!L) return;
    ctx.clearRect(0, 0, L.w, L.h);
    const now = performance.now();
    const progress = this.flowDuration ? Math.min((now - this.flowStart) / this.flowDuration, 1) : 1;
    const nCols = L.columns.length;
    // Column k "lights up" when the sweep reaches it
    const reached = (k) => progress >= k / (nCols - 1) - 1e-6;

    const cur = this.columnValues(this.current);
    const prev = this.columnValues(this.previous);

    this.drawHeatmap(progress);

    // Edges
    this.edges?.forEach((list, li) => {
      const from = L.columns[li], to = L.columns[li + 1];
      const segStart = li / (nCols - 1), segEnd = (li + 1) / (nCols - 1);
      const active = progress > segStart && progress < segEnd;
      for (const { i, j, w } of list) {
        const a = from[i], b = to[j];
        ctx.strokeStyle = rgba(INK, 0.03 + 0.22 * w * w);
        ctx.lineWidth = 0.6 + 1.2 * w;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        // Pulse: a dot travelling along the line, brighter for strong edges
        // carrying a large value
        if (active && cur) {
          const t = (progress - segStart) / (segEnd - segStart);
          const strength = w * Math.min(Math.abs(cur[li][i]) * 1.5, 1);
          if (strength < 0.05) continue;
          ctx.fillStyle = rgba(cur[li][i] >= 0 ? POS : NEG, Math.min(1, 0.35 + strength));
          ctx.beginPath();
          ctx.arc(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, 2 + 3 * strength, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    });

    // Neurons
    L.columns.forEach((col, k) => {
      const values = cur && (reached(k) || !prev) ? cur[k] : prev ? prev[k] : null;
      const isOutput = k === nCols - 1;
      col.forEach((n, i) => {
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = "#ffffff";
        ctx.fill();
        if (values && !isOutput) {
          ctx.fillStyle = valueColor(values[i], k === 0 ? 1 : this.columnMax(cur, k));
          ctx.fill();
        }
        ctx.strokeStyle = rgba(INK, isOutput ? 0.9 : 0.35);
        ctx.lineWidth = isOutput ? 2 : 1;
        ctx.stroke();
      });
    });

    this.drawLabels(cur, reached(nCols - 1));
  }

  columnMax(cur, k) {
    return Math.max(1e-6, ...cur[k].map(Math.abs));
  }

  drawHeatmap(progress) {
    const L = this.layout, ctx = this.ctx, H = L.heat;
    const acts = this.current;
    ctx.strokeStyle = rgba(INK, 0.35);
    ctx.strokeRect(H.x - 0.5, H.y - 0.5, H.w + 1, H.h + 1);
    if (!acts) return;
    const [rows, cols] = acts.input_shape;
    const cw = H.w / cols, rh = H.h / rows;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = acts.input[r * cols + c];
        ctx.fillStyle = rgba(POS, 0.06 + 0.9 * Math.min(Math.max(v, 0), 1));
        ctx.fillRect(H.x + c * cw, H.y + r * rh, cw + 0.4, rh + 0.4);
      }
    }
    // Latest cycle row feeds the feature column
    ctx.strokeStyle = rgba(INK, 0.9);
    ctx.lineWidth = 1.5;
    ctx.strokeRect(H.x, H.y + (rows - 1) * rh, H.w, rh);
    const feat = L.columns[0];
    ctx.strokeStyle = rgba(INK, 0.25);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(H.x + H.w + 4, H.y + (rows - 0.5) * rh);
    ctx.lineTo(feat[0].x - 40, feat[Math.floor(feat.length / 2)].y);
    ctx.stroke();
  }

  drawLabels(cur, outputReached) {
    const L = this.layout, ctx = this.ctx, s = this.structure;
    ctx.fillStyle = rgba(INK, 1);
    ctx.textAlign = "center";
    ctx.font = "600 13px Bahnschrift, 'Segoe UI', system-ui, sans-serif";
    const heads = [
      [L.heat.x + L.heat.w / 2, `Input window`],
      [L.xs[0], "Latest cycle"],
      ...s.layers.map((l, i) => [L.xs[i + 1], i === s.layers.length - 1 ? "Output" : `${l.kind} ${l.units}`]),
    ];
    for (const [x, text] of heads) ctx.fillText(text, x, L.top - 32);
    ctx.font = "12px Bahnschrift, 'Segoe UI', system-ui, sans-serif";
    ctx.fillStyle = rgba(INK, 0.6);
    const subs = [
      [L.heat.x + L.heat.w / 2, `${s.window_size} cycles × ${s.features.length} features`],
      [L.xs[0], `${s.features.length} features`],
      [L.xs[1], "reads the sequence"],
      [L.xs[2], "32-number summary"],
      [L.xs[3], "ReLU"],
      [L.xs[4], "cycles left"],
    ];
    for (const [x, text] of subs) ctx.fillText(text, x, L.top - 16);

    // Feature names beside their neurons
    ctx.textAlign = "right";
    ctx.font = "11px Bahnschrift, 'Segoe UI', system-ui, sans-serif";
    L.columns[0].forEach((n, i) => {
      ctx.fillStyle = rgba(INK, 0.75);
      ctx.fillText(s.features[i], n.x - n.r - 6, n.y + 4);
    });

    // Output value
    const out = L.columns[L.columns.length - 1][0];
    ctx.textAlign = "center";
    if (this.current && outputReached) {
      ctx.fillStyle = rgba(INK, 1);
      ctx.font = "600 17px Bahnschrift, 'Segoe UI', system-ui, sans-serif";
      ctx.fillText(Math.round(this.current.predicted_rul), out.x, out.y + 6);
    }

    // Time axis on the heatmap
    ctx.font = "11px Bahnschrift, 'Segoe UI', system-ui, sans-serif";
    ctx.fillStyle = rgba(INK, 0.6);
    ctx.textAlign = "left";
    ctx.fillText("oldest", L.heat.x, L.bottom + 16);
    ctx.textAlign = "right";
    ctx.fillText("latest", L.heat.x + L.heat.w, L.bottom + 16);
  }

  hover(event) {
    const L = this.layout;
    if (!L || !this.current) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    const cur = this.columnValues(this.current);
    const names = ["Input", ...this.structure.layers.map((l, i) =>
      i === this.structure.layers.length - 1 ? "Output" : `${l.kind} ${l.units}`)];
    for (let k = 0; k < L.columns.length; k++) {
      for (let i = 0; i < L.columns[k].length; i++) {
        const n = L.columns[k][i];
        if (Math.hypot(n.x - x, n.y - y) <= Math.max(n.r, 5) + 2) {
          const label = k === 0 ? `${this.structure.features[i]} at the latest cycle (scaled 0 to 1)`
            : k === L.columns.length - 1 ? `Predicted life left: ${this.current.predicted_rul} cycles`
            : `${names[k]}, unit ${i + 1}`;
          const value = k === L.columns.length - 1 ? "" : `: ${cur[k][i].toFixed(3)}`;
          this.onHover({ text: label + value, x, y });
          return;
        }
      }
    }
    this.onHover(null);
  }
}
