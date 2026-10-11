// The LSTM's vector space as a 3D point cloud.
//
// Every point is one moment (one cycle) of one training engine, placed by
// the PCA projection of the LSTM's 32-number summary of the 30 cycles before
// it, and coloured by how many cycles that engine actually had left. The
// selected test engine is drawn as a path through the cloud, with lines to
// its nearest training neighbours at the current cycle.

import * as THREE from "three";
import { OrbitControls } from "../vendor/OrbitControls.js";

const LIFE_STOPS = [
  [0, new THREE.Color("#d7263d")],
  [35, new THREE.Color("#e08a00")],
  [80, new THREE.Color("#9bbf3a")],
  [125, new THREE.Color("#1e9e5a")],
];

export function lifeColor(life, ceiling = 125) {
  const v = Math.min(Math.max(life, 0), ceiling);
  for (let i = 1; i < LIFE_STOPS.length; i++) {
    const [b, cb] = LIFE_STOPS[i];
    if (v <= b) {
      const [a, ca] = LIFE_STOPS[i - 1];
      return ca.clone().lerp(cb, (v - a) / (b - a));
    }
  }
  return LIFE_STOPS[LIFE_STOPS.length - 1][1].clone();
}

function dotTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.55, "rgba(255,255,255,1)");
  grad.addColorStop(0.7, "rgba(255,255,255,0.35)");
  grad.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export class SpaceView {
  constructor(container, { onHover } = {}) {
    this.container = container;
    this.onHover = onHover || (() => {});
    this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.ref = null;
    this.engine = null;
    this.cycleIndex = 0;
    this.labels = [];

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.prepend(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 50);
    this.camera.position.set(2.5, 1.5, 2.9);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.minDistance = 0.6;
    this.controls.maxDistance = 8;
    this.controls.autoRotate = !this.reducedMotion;
    this.controls.autoRotateSpeed = 0.35;
    this.controls.addEventListener("start", () => { this.controls.autoRotate = false; });

    // Bounding frame: the projection is scaled into a [-1, 1] cube
    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)),
      new THREE.LineBasicMaterial({ color: 0x9aa6b0, transparent: true, opacity: 0.35 }));
    this.scene.add(frame);

    this.group = new THREE.Group();
    this.scene.add(this.group);

    this.raycaster = new THREE.Raycaster();
    this.raycaster.params.Points.threshold = 0.02;
    this.pointer = new THREE.Vector2();
    this.hoverPending = null;
    this.renderer.domElement.addEventListener("pointermove", (e) => {
      this.hoverPending = e;
    });
    this.renderer.domElement.addEventListener("pointerleave", () => {
      this.hoverPending = null;
      this.onHover(null);
    });

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
  }

  setReference(ref) {
    this.ref = ref;
    if (this.points) this.group.remove(this.points);
    const n = ref.n;
    const positions = new Float32Array(ref.xyz);
    const colors = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      lifeColor(ref.life_left[i], ref.rul_ceiling).toArray(colors, i * 3);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    this.points = new THREE.Points(geometry, new THREE.PointsMaterial({
      size: 0.026, vertexColors: true, map: dotTexture(), transparent: true,
      opacity: 0.55, depthWrite: false, alphaTest: 0.05,
    }));
    this.group.add(this.points);
  }

  setEngine(engineSpace) {
    this.engine = engineSpace;
    for (const obj of [this.pathFaint, this.pathSolid, this.marker, this.halo, this.neighborDots, this.links]) {
      if (obj) this.group.remove(obj);
    }
    if (!engineSpace) return;

    const pts = new Float32Array(engineSpace.xyz);
    const pathGeom = new THREE.BufferGeometry();
    pathGeom.setAttribute("position", new THREE.BufferAttribute(pts, 3));
    this.pathFaint = new THREE.Line(pathGeom, new THREE.LineBasicMaterial({
      color: 0x1b2631, transparent: true, opacity: 0.18 }));
    const solidGeom = pathGeom.clone();
    this.pathSolid = new THREE.Line(solidGeom, new THREE.LineBasicMaterial({ color: 0x1b2631 }));

    this.marker = new THREE.Mesh(new THREE.SphereGeometry(0.035, 24, 16),
      new THREE.MeshBasicMaterial({ color: 0x1b2631 }));
    this.halo = new THREE.Mesh(new THREE.RingGeometry(0.05, 0.062, 40),
      new THREE.MeshBasicMaterial({ color: 0x1b2631, transparent: true, opacity: 0.8,
        side: THREE.DoubleSide, depthWrite: false }));

    const k = engineSpace.neighbors[0].length;
    this.neighborDots = new THREE.InstancedMesh(new THREE.SphereGeometry(0.024, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xffffff }), k);
    this.links = new THREE.LineSegments(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0x1f5fad, transparent: true, opacity: 0.85 }));
    this.links.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(k * 6), 3));

    this.group.add(this.pathFaint, this.pathSolid, this.links, this.neighborDots, this.marker, this.halo);
    this.setCycle(engineSpace.cycles.length - 1);
  }

  setCycle(index) {
    if (!this.engine || !this.ref) return;
    this.cycleIndex = Math.min(Math.max(index, 0), this.engine.cycles.length - 1);
    const xyz = this.engine.xyz;
    const p = new THREE.Vector3(xyz[this.cycleIndex * 3], xyz[this.cycleIndex * 3 + 1], xyz[this.cycleIndex * 3 + 2]);
    this.marker.position.copy(p);
    this.halo.position.copy(p);
    this.pathSolid.geometry.setDrawRange(0, this.cycleIndex + 1);

    const nb = this.engine.neighbors[this.cycleIndex];
    const linkPos = this.links.geometry.attributes.position;
    const m = new THREE.Matrix4();
    nb.forEach((refIndex, j) => {
      const q = new THREE.Vector3().fromArray(this.ref.xyz, refIndex * 3);
      m.makeTranslation(q.x, q.y, q.z);
      this.neighborDots.setMatrixAt(j, m);
      this.neighborDots.setColorAt(j, lifeColor(this.ref.life_left[refIndex], this.ref.rul_ceiling));
      linkPos.setXYZ(j * 2, p.x, p.y, p.z);
      linkPos.setXYZ(j * 2 + 1, q.x, q.y, q.z);
    });
    this.neighborDots.instanceMatrix.needsUpdate = true;
    if (this.neighborDots.instanceColor) this.neighborDots.instanceColor.needsUpdate = true;
    linkPos.needsUpdate = true;
  }

  // Axis labels are HTML, so they stay crisp; positioned every frame.
  setAxisLabels(elements) {
    this.labels = elements;
  }

  resetView() {
    this.camera.position.set(2.5, 1.5, 2.9);
    this.controls.target.set(0, 0, 0);
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.resize();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  stop() {
    this.running = false;
    this.renderer.setAnimationLoop(null);
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  handleHover(event) {
    if (!this.points) return;
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    // Prefer the engine's own path and neighbours over the dense cloud
    const hit = this.raycaster.intersectObject(this.points, false)[0];
    this.onHover(hit ? { index: hit.index, x: event.clientX - rect.left, y: event.clientY - rect.top } : null);
  }

  frame() {
    if (this.hoverPending) {
      this.handleHover(this.hoverPending);
      this.hoverPending = null;
    }
    this.controls.update();
    if (this.halo) this.halo.quaternion.copy(this.camera.quaternion);
    this.renderer.render(this.scene, this.camera);

    const rect = this.renderer.domElement.getBoundingClientRect();
    for (const { el, pos } of this.labels) {
      const v = pos.clone().project(this.camera);
      el.style.transform = `translate(${(v.x * 0.5 + 0.5) * rect.width}px, ${(-v.y * 0.5 + 0.5) * rect.height}px)`;
      el.hidden = v.z > 1;
    }
  }
}

export const AXIS_LABEL_POSITIONS = [
  new THREE.Vector3(1.12, -1, -1),
  new THREE.Vector3(-1, 1.12, -1),
  new THREE.Vector3(-1, -1, 1.12),
];
