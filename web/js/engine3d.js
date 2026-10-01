// Procedural cutaway of a two-spool high-bypass turbofan, the engine class
// C-MAPSS simulates. Geometry is illustrative; the station layout (fan, LPC,
// HPC, combustor, HPT, LPT, nozzle, bypass duct) and the sensor positions
// follow the C-MAPSS module map in Saxena et al., PHM 2008.
//
// Everything is modelled around the local Y axis (flow runs towards +Y) and
// the whole engine group is then turned so the axis lies along world X.

import * as THREE from "three";
import { OrbitControls } from "../vendor/OrbitControls.js";
import { RoomEnvironment } from "../vendor/RoomEnvironment.js";

export const SEVERITY_COLORS = {
  normal: "#1e9e5a",
  warning: "#e08a00",
  critical: "#d7263d",
  low_signal: "#8c99a3",
  unmonitored: "#b9c2c9",
};

// Centre of the cutaway opening, in local lathe angle. Lathe angle 0 points
// to local +Z (towards the default camera); -PI/2 points to local -X, which
// becomes world up once the group is rotated. The opening faces up-and-out.
const CUT_CENTER = -Math.PI / 4;
const CUT_WIDTH = THREE.MathUtils.degToRad(118);

// Where each module's sensors sit: axial position and radius in local units.
const STATIONS = {
  inlet:      { y: -2.42, r: 1.27 },
  fan:        { y: -1.98, r: 1.33 },
  bypass:     { y: -0.55, r: 0.80 },
  lpc:        { y: -0.66, r: 0.60 },
  core_shaft: { y: -0.56, r: 0.25 },
  hpc:        { y: 0.36, r: 0.50 },
  combustor:  { y: 0.60, r: 0.56 },
  hpt:        { y: 0.92, r: 0.55 },
  lpt:        { y: 1.66, r: 0.64 },
  nozzle:     { y: 2.06, r: 0.50 },
};

export const MODULE_ORDER = [
  ["inlet", "Inlet"], ["fan", "Fan"], ["bypass", "Bypass duct"],
  ["lpc", "LPC"], ["core_shaft", "Core shaft"], ["hpc", "HPC"],
  ["combustor", "Combustor"], ["hpt", "HPT"], ["lpt", "LPT"], ["nozzle", "Nozzle"],
];

// ------------------------------------------------------------- geometry ----

function lathe(profile, material, { cut = false, segments = 96 } = {}) {
  const points = profile.map(([r, y]) => new THREE.Vector2(r, y));
  const geometry = cut
    ? new THREE.LatheGeometry(points, segments,
        CUT_CENTER + CUT_WIDTH / 2, Math.PI * 2 - CUT_WIDTH)
    : new THREE.LatheGeometry(points, segments);
  return new THREE.Mesh(geometry, material);
}

// A twisted aerofoil-ish blade: a thin box whose sections rotate with radius.
function bladeGeometry(span, chord, thickness, twistRoot, twistTip, taper = 0.75) {
  const g = new THREE.BoxGeometry(thickness, span, chord, 1, 10, 4);
  const pos = g.attributes.position;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    const t = (v.y + span / 2) / span;               // 0 at root, 1 at tip
    const scale = 1 - (1 - taper) * t;
    v.z *= scale;
    // camber: bow the section so it reads as an aerofoil, not a plank
    v.x += 0.18 * chord * scale * (1 - (2 * v.z / (chord * scale)) ** 2);
    const angle = twistRoot + (twistTip - twistRoot) * t;
    const x = v.x * Math.cos(angle) - v.z * Math.sin(angle);
    const z = v.x * Math.sin(angle) + v.z * Math.cos(angle);
    pos.setXYZ(i, x, v.y + span / 2, z);
  }
  g.computeVertexNormals();
  return g;
}

// A ring of blades: one InstancedMesh so a 24-blade stage is one draw call.
function bladeRing({ y, rHub, rTip, count, chord, thickness = 0.012,
                     twistRoot = 0.9, twistTip = 0.25, material }) {
  const geometry = bladeGeometry(rTip - rHub, chord, thickness, twistRoot, twistTip);
  const mesh = new THREE.InstancedMesh(geometry, material, count);
  const m = new THREE.Matrix4();
  const axial = new THREE.Vector3(0, 1, 0);
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2;
    // An explicit basis, not setFromUnitVectors: that picks an arbitrary
    // roll per blade, which scatters the chord direction around the ring.
    const radial = new THREE.Vector3(Math.sin(a), 0, Math.cos(a));
    const tangent = new THREE.Vector3(-Math.cos(a), 0, Math.sin(a));
    m.makeBasis(tangent, radial, axial);
    m.setPosition(radial.x * rHub, y, radial.z * rHub);
    mesh.setMatrixAt(i, m);
  }
  return mesh;
}

function stagesBetween(y0, y1, n, fn) {
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? 0 : i / (n - 1);
    out.push(fn(y0 + (y1 - y0) * t, t, i));
  }
  return out;
}

function buildEngine() {
  const mat = {
    nacelle: new THREE.MeshStandardMaterial({ color: 0xdfe4e8, metalness: 0.15, roughness: 0.45, side: THREE.DoubleSide }),
    duct: new THREE.MeshStandardMaterial({ color: 0x9aa6b0, metalness: 0.55, roughness: 0.5, side: THREE.DoubleSide }),
    cowl: new THREE.MeshStandardMaterial({ color: 0xc7ced4, metalness: 0.6, roughness: 0.35, side: THREE.DoubleSide }),
    titanium: new THREE.MeshStandardMaterial({ color: 0xb9c1c8, metalness: 0.9, roughness: 0.28 }),
    steel: new THREE.MeshStandardMaterial({ color: 0x8f99a3, metalness: 0.85, roughness: 0.35 }),
    hot: new THREE.MeshStandardMaterial({ color: 0x8a6a4f, metalness: 0.8, roughness: 0.4 }),
    drum: new THREE.MeshStandardMaterial({ color: 0x6d7882, metalness: 0.8, roughness: 0.45, side: THREE.DoubleSide }),
    spinner: new THREE.MeshStandardMaterial({ color: 0x2c3540, metalness: 0.6, roughness: 0.3 }),
    flame: new THREE.MeshStandardMaterial({ color: 0xb0623a, emissive: 0xff7a2a, emissiveIntensity: 0.9, metalness: 0.3, roughness: 0.5 }),
  };

  const engine = new THREE.Group();
  const lpSpool = new THREE.Group();   // fan, booster, LPT: one shaft
  const hpSpool = new THREE.Group();   // HPC, HPT: the other

  // Nacelle: outer skin and inner fan duct, both cut away.
  engine.add(lathe([[1.47, -2.62], [1.56, -2.45], [1.62, -1.9], [1.6, -0.9], [1.5, 0.1], [1.36, 1.0]], mat.nacelle, { cut: true }));
  engine.add(lathe([[1.47, -2.62], [1.36, -2.5], [1.33, -2.2]], mat.nacelle, { cut: true }));
  engine.add(lathe([[1.33, -2.2], [1.36, -1.6]], mat.duct, { cut: true }));
  engine.add(lathe([[1.36, -1.6], [1.36, -0.9], [1.3, 0.1], [1.25, 1.0]], mat.duct, { cut: true }));

  // Core cowl: the bypass duct's inner wall, also cut away.
  engine.add(lathe([[0.6, -1.12], [0.7, -0.9], [0.78, -0.3], [0.8, 0.5], [0.74, 1.3], [0.6, 1.95]], mat.cowl, { cut: true }));
  // Splitter lip between bypass and core flow
  engine.add(lathe([[0.6, -1.12], [0.56, -1.08], [0.58, -0.95]], mat.titanium, { cut: true }));

  // Inner core casing (compressor/turbine shroud), cut.
  engine.add(lathe([[0.58, -1.0], [0.6, -0.62], [0.5, -0.5], [0.46, 0.35], [0.58, 0.45], [0.6, 0.78], [0.54, 0.86], [0.56, 1.0], [0.66, 1.6], [0.62, 1.9]], mat.steel, { cut: true }));

  // Hub drums (inner flow path), complete rings.
  engine.add(lathe([[0.36, -1.02], [0.4, -0.62]], mat.drum));
  engine.add(lathe([[0.27, -0.52], [0.36, 0.38]], mat.drum));
  engine.add(lathe([[0.38, 0.8], [0.38, 0.98]], mat.drum));
  engine.add(lathe([[0.36, 1.0], [0.36, 1.62]], mat.drum));

  // Spinner and fan
  const spinner = lathe([[0.0, -2.36], [0.18, -2.25], [0.34, -2.05], [0.42, -1.86], [0.42, -1.7]], mat.spinner);
  lpSpool.add(spinner);
  const fan = bladeRing({ y: -1.92, rHub: 0.42, rTip: 1.31, count: 20, chord: 0.36,
    thickness: 0.02, twistRoot: 0.55, twistTip: 1.15, material: mat.titanium });
  lpSpool.add(fan);
  // Fan outlet guide vanes (static)
  engine.add(bladeRing({ y: -1.32, rHub: 0.7, rTip: 1.36, count: 36, chord: 0.14,
    twistRoot: 0.25, twistTip: 0.1, material: mat.steel }));

  // Booster / LPC: 3 stages on the LP spool
  lpSpool.add(...stagesBetween(-0.98, -0.7, 3, (y) =>
    bladeRing({ y, rHub: 0.38, rTip: 0.58, count: 34, chord: 0.07, material: mat.titanium })));

  // HPC: 9 stages, annulus narrowing towards the combustor
  hpSpool.add(...stagesBetween(-0.46, 0.3, 9, (y, t) =>
    bladeRing({ y, rHub: 0.28 + 0.08 * t, rTip: 0.5 - 0.04 * t, count: 44, chord: 0.05,
      twistRoot: 0.8, twistTip: 0.35, material: mat.titanium })));

  // Combustor: annular liner plus a ring of fuel nozzles
  const liner = new THREE.Mesh(new THREE.TorusGeometry(0.47, 0.075, 18, 96), mat.flame);
  liner.rotation.x = Math.PI / 2;
  liner.position.y = 0.6;
  liner.scale.set(1, 1, 2.1);
  engine.add(liner);
  for (let i = 0; i < 20; i++) {
    const a = (i / 20) * Math.PI * 2;
    const nozzle = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 0.14), mat.steel);
    nozzle.position.set(Math.sin(a) * 0.47, 0.44, Math.cos(a) * 0.47);
    engine.add(nozzle);
  }

  // HPT: 2 stages, hot-section alloy
  hpSpool.add(...stagesBetween(0.84, 0.95, 2, (y) =>
    bladeRing({ y, rHub: 0.38, rTip: 0.55, count: 46, chord: 0.06, thickness: 0.016,
      twistRoot: -0.9, twistTip: -0.5, material: mat.hot })));

  // LPT: 5 stages, annulus growing towards the exhaust
  lpSpool.add(...stagesBetween(1.06, 1.56, 5, (y, t) =>
    bladeRing({ y, rHub: 0.36, rTip: 0.55 + 0.09 * t, count: 52, chord: 0.06, thickness: 0.014,
      twistRoot: -0.9, twistTip: -0.45, material: mat.hot })));

  // Exhaust plug and core nozzle
  engine.add(lathe([[0.36, 1.62], [0.3, 1.95], [0.15, 2.3], [0.0, 2.45]], mat.drum));

  // Shafts: LP runs the length of the engine inside the hollow HP shaft
  const lpShaft = new THREE.Mesh(new THREE.CylinderGeometry(0.055, 0.055, 3.5, 24), mat.steel);
  lpShaft.position.y = -0.1;
  lpSpool.add(lpShaft);
  const hpShaft = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 1.5, 32, 1, true), mat.steel);
  hpShaft.position.y = 0.22;
  hpSpool.add(hpShaft);

  engine.add(lpSpool, hpSpool);
  engine.rotation.z = -Math.PI / 2;     // local +Y (aft) -> world +X
  return { engine, lpSpool, hpSpool };
}

// ----------------------------------------------------------------- view ----

export class EngineView {
  constructor(container, { onSelect, onHover } = {}) {
    this.container = container;
    this.onSelect = onSelect || (() => {});
    this.onHover = onHover || (() => {});
    this.markers = new Map();          // column -> { mesh, halo, severity }
    this.selected = null;
    this.reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.frameCallbacks = [];
    this.tween = null;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;

    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(-3, 5, 4);
    this.scene.add(key, new THREE.AmbientLight(0xffffff, 0.35));

    this.camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
    this.homePosition = new THREE.Vector3(-5.5, 3.2, 7.5);
    this.homeTarget = new THREE.Vector3(0.1, 0, 0);
    this.camera.position.copy(this.homePosition);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(this.homeTarget);
    this.controls.enableDamping = true;
    this.controls.minDistance = 2.2;
    this.controls.maxDistance = 18;
    this.controls.addEventListener("start", () => { this.tween = null; });

    const built = buildEngine();
    this.engine = built.engine;
    this.lpSpool = built.lpSpool;
    this.hpSpool = built.hpSpool;
    this.scene.add(this.engine);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.hovered = null;
    this.downAt = null;
    const canvas = this.renderer.domElement;
    canvas.addEventListener("pointermove", (e) => this.handleMove(e));
    canvas.addEventListener("pointerdown", (e) => { this.downAt = [e.clientX, e.clientY]; });
    canvas.addEventListener("pointerup", (e) => this.handleClick(e));

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.clock = new THREE.Clock();
    this.renderer.setAnimationLoop(() => this.frame());
  }

  // Place one marker per C-MAPSS channel at its module's station.
  setSensors(sensors) {
    for (const { mesh, halo } of this.markers.values()) this.engine.remove(mesh, halo);
    this.markers.clear();

    const byModule = {};
    for (const s of sensors) (byModule[s.module] ||= []).push(s);

    const sphere = new THREE.SphereGeometry(0.05, 24, 16);
    const ring = new THREE.RingGeometry(0.07, 0.095, 40);
    for (const [module, list] of Object.entries(byModule)) {
      const station = STATIONS[module];
      const step = THREE.MathUtils.degToRad(15);
      list.forEach((sensor, i) => {
        const angle = CUT_CENTER + (i - (list.length - 1) / 2) * step;
        const y = station.y + (i % 2 ? 0.05 : -0.05) * (list.length > 2 ? 1 : 0);
        const position = new THREE.Vector3(Math.sin(angle) * station.r, y, Math.cos(angle) * station.r);

        const material = new THREE.MeshStandardMaterial({
          color: SEVERITY_COLORS.unmonitored, emissive: SEVERITY_COLORS.unmonitored,
          emissiveIntensity: 0.25, roughness: 0.35, metalness: 0.1,
        });
        const mesh = new THREE.Mesh(sphere, material);
        mesh.position.copy(position);
        mesh.userData.column = sensor.column;
        mesh.renderOrder = 2;

        const halo = new THREE.Mesh(ring, new THREE.MeshBasicMaterial({
          color: SEVERITY_COLORS.unmonitored, transparent: true, opacity: 0,
          side: THREE.DoubleSide, depthWrite: false,
        }));
        halo.position.copy(position);
        halo.userData.column = sensor.column;

        this.engine.add(mesh, halo);
        this.markers.set(sensor.column, { mesh, halo, severity: "unmonitored", module });
      });
    }
  }

  setSeverities(severities) {
    for (const [column, marker] of this.markers) {
      const severity = severities[column] || "unmonitored";
      if (marker.severity === severity) continue;
      marker.severity = severity;
      const color = new THREE.Color(SEVERITY_COLORS[severity]);
      marker.mesh.material.color.copy(color);
      marker.mesh.material.emissive.copy(color);
      marker.mesh.material.emissiveIntensity = severity === "unmonitored" ? 0.15 : 0.55;
      marker.halo.material.color.copy(color);
    }
  }

  select(column, { focus = true } = {}) {
    this.selected = column;
    if (!focus) return;
    if (!column) {
      this.flyTo(this.homePosition, this.homeTarget);
      return;
    }
    const marker = this.markers.get(column);
    if (!marker) return;
    const target = marker.mesh.getWorldPosition(new THREE.Vector3());
    const outward = target.clone().setX(0).normalize();
    if (outward.lengthSq() === 0) outward.set(0, 0.6, 1).normalize();
    const position = target.clone()
      .add(outward.multiplyScalar(4.2))
      .add(new THREE.Vector3(-1.9, 1.1, 1.7));
    this.flyTo(position, target.clone().lerp(new THREE.Vector3(target.x, 0, 0), 0.35));
  }

  resetView() {
    this.select(null);
  }

  flyTo(position, target) {
    if (this.reducedMotion) {
      this.camera.position.copy(position);
      this.controls.target.copy(target);
      return;
    }
    this.tween = {
      fromPos: this.camera.position.clone(), toPos: position.clone(),
      fromTarget: this.controls.target.clone(), toTarget: target.clone(),
      start: performance.now(), duration: 900,
    };
  }

  // Materials are shared between modules (every compressor stage is the same
  // titanium), so tinting geometry would light up unrelated parts. Enlarging
  // the module's markers is unambiguous.
  highlightModule(module) {
    for (const marker of this.markers.values()) {
      marker.mesh.scale.setScalar(module && marker.module === module ? 1.6 : 1);
    }
  }

  // Screen-space position of a marker, for drawing HTML callouts.
  screenPosition(column) {
    const marker = this.markers.get(column);
    if (!marker) return null;
    const p = marker.mesh.getWorldPosition(new THREE.Vector3()).project(this.camera);
    const rect = this.renderer.domElement.getBoundingClientRect();
    return {
      x: (p.x * 0.5 + 0.5) * rect.width,
      y: (-p.y * 0.5 + 0.5) * rect.height,
      visible: p.z < 1,
    };
  }

  onFrame(callback) {
    this.frameCallbacks.push(callback);
  }

  pick(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      -((event.clientY - rect.top) / rect.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const meshes = [...this.markers.values()].flatMap((m) => [m.mesh, m.halo]);
    const hit = this.raycaster.intersectObjects(meshes, false)[0];
    return hit ? hit.object.userData.column : null;
  }

  handleMove(event) {
    const column = this.pick(event);
    if (column !== this.hovered) {
      this.hovered = column;
      this.renderer.domElement.style.cursor = column ? "pointer" : "grab";
      this.onHover(column);
    }
  }

  handleClick(event) {
    if (!this.downAt) return;
    const moved = Math.hypot(event.clientX - this.downAt[0], event.clientY - this.downAt[1]);
    this.downAt = null;
    if (moved > 5) return;               // that was an orbit drag, not a click
    this.onSelect(this.pick(event));
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Pull back on narrow screens so the whole engine stays in frame
    this.camera.zoom = Math.min(1, (w / h) / 1.45);
    this.camera.updateProjectionMatrix();
  }

  frame() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;

    if (!this.reducedMotion) {
      this.lpSpool.rotation.y += dt * 2.4;
      this.hpSpool.rotation.y += dt * 5.1;
    }

    if (this.tween) {
      const k = Math.min(1, (performance.now() - this.tween.start) / this.tween.duration);
      const e = 1 - (1 - k) ** 3;
      this.camera.position.lerpVectors(this.tween.fromPos, this.tween.toPos, e);
      this.controls.target.lerpVectors(this.tween.fromTarget, this.tween.toTarget, e);
      if (k === 1) this.tween = null;
    }
    this.controls.update();

    for (const [column, { halo, severity }] of this.markers) {
      halo.quaternion.copy(this.camera.quaternion);
      halo.quaternion.premultiply(this.engine.quaternion.clone().invert());
      const isSelected = column === this.selected;
      if (severity === "critical" && !this.reducedMotion) {
        const pulse = (t * 1.2) % 1;
        halo.scale.setScalar(1 + pulse * 1.4);
        halo.material.opacity = 0.75 * (1 - pulse);
      } else {
        halo.scale.setScalar(isSelected ? 1.5 : 1);
        halo.material.opacity = isSelected ? 0.9 : 0;
      }
      if (isSelected) halo.material.opacity = Math.max(halo.material.opacity, 0.9);
    }

    this.renderer.render(this.scene, this.camera);
    for (const callback of this.frameCallbacks) callback();
  }
}
