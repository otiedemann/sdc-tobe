// marker_mission_sim — live 3D arena view (Three.js, no build step).
//
// Coordinate mapping (arena -> Three.js):
//     X_three =  arena.x          (+x = right)
//     Y_three =  arena.z          (up)
//     Z_three = -arena.y          (+arena.y / front-blue wall -> -Z, "into" screen)
// Applied identically to floor zones, markers, boxes and drones.
//
// Flow: fetch /api/world once to build the static geometry, then subscribe to
// /api/world/stream (SSE) — falling back to polling /api/world — to update
// drone poses, box colours and the HUD.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ---- colours (kept in sync with app.css) ----
const COL = {
  red: 0xff5a5a,
  blue: 0x4d9bff,
  neutral: 0x707a8a,
  redZone: 0x3a1417,
  blueZone: 0x122742,
  neutralZone: 0x161a22,
  marker: 0xf2f5fa,
  wall: 0x4a5870,
  grid: 0x2a3340,
  drop: 0x55627a,
};
const ZONE_DEPTH = 5; // metres of coloured strip at each end

// arena -> three.js position
const toThree = (x, y, z) => new THREE.Vector3(x, z, -y);

// ---------------------------------------------------------------------------
// Renderer / scene / camera
// ---------------------------------------------------------------------------
const canvas = document.getElementById("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0c0f14);
scene.fog = new THREE.Fog(0x0c0f14, 30, 90);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 500);
camera.position.set(13, 16, 22);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495; // don't dip under the floor
controls.target.set(0, 1, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const key = new THREE.DirectionalLight(0xffffff, 1.0);
key.position.set(8, 18, 6);
scene.add(key);
const fill = new THREE.DirectionalLight(0x88aaff, 0.35);
fill.position.set(-10, 8, -8);
scene.add(fill);

// ---------------------------------------------------------------------------
// HTML label overlay — projected to screen each frame for crisp text.
// ---------------------------------------------------------------------------
const labels = []; // { el, world: Vector3 }
function makeLabel(text, cls) {
  const el = document.createElement("div");
  el.className = "tag " + (cls || "");
  el.textContent = text;
  document.body.appendChild(el);
  const rec = { el, world: new THREE.Vector3() };
  labels.push(rec);
  return rec;
}
function updateLabels() {
  const w = renderer.domElement.clientWidth;
  const h = renderer.domElement.clientHeight;
  const v = new THREE.Vector3();
  for (const { el, world } of labels) {
    v.copy(world).project(camera);
    const behind = v.z > 1;
    if (behind) {
      el.classList.add("hidden");
      continue;
    }
    el.classList.remove("hidden");
    el.style.left = ((v.x * 0.5 + 0.5) * w) + "px";
    el.style.top = ((-v.y * 0.5 + 0.5) * h) + "px";
  }
}

// ---------------------------------------------------------------------------
// Static geometry (built once from the first snapshot)
// ---------------------------------------------------------------------------
function buildArena(arena) {
  const W = arena.width_m;
  const D = arena.depth_m;
  const C = arena.ceiling_m;

  const zoneGeo = (depth) => new THREE.PlaneGeometry(W, depth);
  const mkZone = (depth, yCenter, color) => {
    const m = new THREE.Mesh(
      zoneGeo(depth),
      new THREE.MeshStandardMaterial({ color, roughness: 0.95, metalness: 0.0 })
    );
    m.rotation.x = -Math.PI / 2;          // lay flat
    m.position.copy(toThree(0, yCenter, 0));
    m.position.y = -0.01;                  // just under the grid
    scene.add(m);
  };
  // red strip at -y end, blue strip at +y end, neutral in the middle.
  mkZone(ZONE_DEPTH, -D / 2 + ZONE_DEPTH / 2, COL.redZone);
  mkZone(ZONE_DEPTH, D / 2 - ZONE_DEPTH / 2, COL.blueZone);
  mkZone(D - 2 * ZONE_DEPTH, 0, COL.neutralZone);

  // faint grid over the whole floor
  const grid = new THREE.GridHelper(Math.max(W, D), Math.max(W, D), COL.grid, COL.grid);
  grid.position.y = 0;
  grid.material.opacity = 0.35;
  grid.material.transparent = true;
  // clip the square helper to the arena footprint
  grid.scale.set(W / Math.max(W, D), 1, D / Math.max(W, D));
  scene.add(grid);

  // translucent perimeter walls (x = ±W/2, y = ±D/2)
  const wallMat = new THREE.MeshStandardMaterial({
    color: COL.wall, transparent: true, opacity: 0.12,
    side: THREE.DoubleSide, roughness: 1.0,
  });
  const edgeMat = new THREE.LineBasicMaterial({ color: COL.wall, transparent: true, opacity: 0.5 });
  const t = 0.08; // wall thickness
  const addWall = (sx, sz, cx, cy) => {
    const g = new THREE.BoxGeometry(sx, C, sz);
    const wall = new THREE.Mesh(g, wallMat);
    wall.position.copy(toThree(cx, cy, C / 2));
    scene.add(wall);
    const edges = new THREE.LineSegments(new THREE.EdgesGeometry(g), edgeMat);
    edges.position.copy(wall.position);
    scene.add(edges);
  };
  addWall(W, t, 0, +D / 2);  // front (blue, +y)
  addWall(W, t, 0, -D / 2);  // back (red, -y)
  addWall(t, D, +W / 2, 0);  // right
  addWall(t, D, -W / 2, 0);  // left

  return { W, D, C };
}

// Inward-facing normal per wall (arena frame): the direction a marker glued
// FLAT to that wall points — straight out into the play area, perpendicular
// to the wall, regardless of where along the wall it sits.
const WALL_INWARD = {
  front: [0, -1, 0],   // +Y wall (y=+10) faces -Y
  back:  [0,  1, 0],   // -Y wall (y=-10) faces +Y
  right: [-1, 0, 0],   // +X wall (x=+5) faces -X
  left:  [ 1, 0, 0],   // -X wall (x=-5) faces +X
};

function buildMarkers(markers) {
  const geo = new THREE.PlaneGeometry(0.35, 0.35);
  const mat = new THREE.MeshBasicMaterial({
    color: COL.marker, side: THREE.DoubleSide,
  });
  for (const m of markers) {
    const sq = new THREE.Mesh(geo, mat);
    const [x, y, z] = m.pos;
    sq.position.copy(toThree(x, y, z));
    // Orient the square FLAT against its own wall (perpendicular to the wall),
    // not toward the arena centre. The old `lookAt(centre)` rotated side-wall
    // markers (the ones not at y=0) by up to 45° toward the middle — visually
    // wrong: a marker on a flat wall points straight out from that wall. (The
    // vision sim ignores marker facing, so this is purely a view fix.)
    const n = WALL_INWARD[m.wall];
    if (n) sq.lookAt(toThree(x + n[0], y + n[1], z + n[2]));
    else   sq.lookAt(toThree(0, 0, z));   // unknown wall -> fall back to centre
    scene.add(sq);
  }
}

// ---------------------------------------------------------------------------
// Dynamic objects (created once, mutated every frame)
// ---------------------------------------------------------------------------
const boxMeshes = new Map();   // slot -> { mesh, label }
const droneMeshes = new Map(); // id   -> { group, cone, line, label }

function teamColor(team) {
  return team === "blue" ? COL.blue : COL.red;
}

function buildBoxes(boxes) {
  const size = 0.5;
  for (const b of boxes) {
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(size, size, size),
      new THREE.MeshStandardMaterial({ roughness: 0.5, metalness: 0.1 })
    );
    const [x, y, z] = b.pos;
    mesh.position.copy(toThree(x, y, z));
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(mesh.geometry),
      new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 })
    );
    mesh.add(edges);
    scene.add(mesh);

    const label = makeLabel("", "box");
    boxMeshes.set(b.slot, { mesh, label, size });
    updateBox(b);
  }
}

function updateBox(b) {
  const rec = boxMeshes.get(b.slot);
  if (!rec) return;
  rec.mesh.material.color.setHex(teamColor(b.holder));
  rec.mesh.material.emissive.setHex(teamColor(b.holder));
  rec.mesh.material.emissiveIntensity = 0.18;
  const [x, y, z] = b.pos;
  rec.label.world.copy(toThree(x, y, z));
  rec.label.world.y += rec.size * 0.5 + 0.25;
  rec.label.el.textContent = `#${b.slot} f${b.current_face_id}`;
  rec.label.el.className = "tag box " + b.holder;
}

function buildDrone(d) {
  const group = new THREE.Group();
  // cone points +Y by default; rotate so it lies flat pointing toward +Z(forward).
  const cone = new THREE.Mesh(
    new THREE.ConeGeometry(0.22, 0.7, 16),
    new THREE.MeshStandardMaterial({
      color: teamColor(d.team), roughness: 0.35, metalness: 0.2,
      emissive: teamColor(d.team), emissiveIntensity: 0.25,
    })
  );
  cone.rotation.x = Math.PI / 2;       // nose -> +Z (arena +y after mapping handled by group yaw)
  group.add(cone);
  scene.add(group);

  // vertical drop-line from the drone down to the floor (altitude cue)
  const lineGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 0),
  ]);
  const line = new THREE.Line(
    lineGeo,
    new THREE.LineDashedMaterial({ color: COL.drop, dashSize: 0.2, gapSize: 0.15, transparent: true, opacity: 0.6 })
  );
  scene.add(line);

  const label = makeLabel(d.id, d.team);
  droneMeshes.set(d.id, { group, cone, line, label });
  updateDrone(d);
}

function updateDrone(d) {
  const rec = droneMeshes.get(d.id);
  if (!rec) return;
  const [x, y, z] = d.pos;
  const p = toThree(x, y, z);
  rec.group.position.copy(p);

  // heading is CW from +Y (arena). Arena +y -> three -Z. A yaw of 0 should
  // point the nose toward -Z; rotation about three's +Y is CCW, so heading
  // (CW) maps to -heading about Y, with a +180° offset to face -Z at 0.
  const yaw = THREE.MathUtils.degToRad(-d.heading_deg) + Math.PI;
  rec.group.rotation.set(0, yaw, 0);

  // drop-line to floor
  const pos = rec.line.geometry.attributes.position;
  pos.setXYZ(0, p.x, p.y, p.z);
  pos.setXYZ(1, p.x, 0, p.z);
  pos.needsUpdate = true;
  rec.line.computeLineDistances();

  // dim while idle / landed
  const lit = d.flying ? 0.35 : 0.06;
  rec.cone.material.emissiveIntensity = lit;

  rec.label.world.copy(p);
  rec.label.world.y += 0.6;
}

// ---------------------------------------------------------------------------
// HUD
// ---------------------------------------------------------------------------
const $clock = document.getElementById("clock");
const $drones = document.getElementById("drones");
const $slots = document.getElementById("slots");
const $conn = document.getElementById("conn");

function updateHud(snap) {
  $clock.textContent = `t = ${Number(snap.t).toFixed(1)}s`;

  if (!snap.drones.length) {
    $drones.innerHTML = '<div class="hud-empty">no drones</div>';
  } else {
    $drones.innerHTML = snap.drones.map((d) => {
      const alt = (d.pos[2] ?? 0).toFixed(2);
      const spd = (d.speed_mps ?? 0).toFixed(2);
      const step = d.n_steps ? `${d.step_idx}/${d.n_steps}` : "—";
      return `<div class="drone ${d.team}">
        <div class="row1"><span class="id">${d.id}</span><span class="phase">${d.phase}</span></div>
        <div class="meta"><span>alt ${alt}m</span><span>${spd} m/s</span><span>step ${step}</span><span>${d.flying ? "flying" : "idle"}</span></div>
      </div>`;
    }).join("");
  }

  if (!snap.boxes.length) {
    $slots.innerHTML = '<div class="hud-empty">no boxes</div>';
  } else {
    $slots.innerHTML = snap.boxes.map((b) =>
      `<div class="slot ${b.holder}"><span class="n">#${b.slot}</span><span class="face">${b.holder}·f${b.current_face_id}</span></div>`
    ).join("");
  }
}

// ---------------------------------------------------------------------------
// Live updates: build static geometry once, then patch dynamics each snapshot.
// ---------------------------------------------------------------------------
let built = false;

function applySnapshot(snap) {
  if (!built) {
    buildArena(snap.arena);
    buildMarkers(snap.wall_markers || []);
    buildBoxes(snap.boxes || []);
    for (const d of snap.drones || []) buildDrone(d);
    built = true;
  } else {
    for (const b of snap.boxes || []) updateBox(b);
    for (const d of snap.drones || []) {
      if (!droneMeshes.has(d.id)) buildDrone(d);
      else updateDrone(d);
    }
  }
  updateHud(snap);
}

function setConn(state, text) {
  $conn.className = "conn " + state;
  $conn.textContent = text;
}

async function fetchOnce() {
  const r = await fetch("/api/world", { cache: "no-store" });
  if (!r.ok) throw new Error("HTTP " + r.status);
  return r.json();
}

// Prefer the SSE stream; fall back to polling if it errors out.
function startStream() {
  let usingPoll = false;
  let pollTimer = null;

  const poll = async () => {
    try {
      applySnapshot(await fetchOnce());
      setConn("live", "polling /api/world");
    } catch (e) {
      setConn("err", "feed error: " + e.message);
    }
  };
  const startPolling = () => {
    if (usingPoll) return;
    usingPoll = true;
    poll();
    pollTimer = setInterval(poll, 200);
  };

  let es;
  try {
    es = new EventSource("/api/world/stream");
  } catch {
    startPolling();
    return;
  }
  es.onmessage = (ev) => {
    try {
      applySnapshot(JSON.parse(ev.data));
      setConn("live", "live · SSE");
    } catch { /* ignore malformed frame */ }
  };
  es.onerror = () => {
    // EventSource auto-reconnects; if it can't, fall back to polling.
    setConn("err", "stream dropped — polling");
    es.close();
    if (!pollTimer) startPolling();
  };
}

// ---------------------------------------------------------------------------
// Resize + render loop
// ---------------------------------------------------------------------------
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function tick() {
  controls.update();
  renderer.render(scene, camera);
  updateLabels();
  requestAnimationFrame(tick);
}
tick();

// Kick things off: one fetch to build, then the live feed.
(async () => {
  try {
    applySnapshot(await fetchOnce());
    setConn("live", "connected");
  } catch (e) {
    setConn("err", "initial fetch failed: " + e.message);
  }
  startStream();
})();
