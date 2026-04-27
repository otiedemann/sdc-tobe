
  import * as THREE from 'three';
  import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

  let scene, camera, renderer, controls, droneMeshes = {}, markerMeshes = {}, rafId = 0;
  const ARENA_W = 20.0, ARENA_D = 10.8, ARENA_H = 6.0;
  const ARENA_OX = -10.0, ARENA_OY = 0.0;

  function init3D(container) {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b1220);

    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 200);
    camera.position.set(15, 12, 15);

    renderer = new THREE.WebGLRenderer({antialias: true});
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.5, 5.4);
    controls.update();

    // Lighting
    const ambient = new THREE.AmbientLight(0xffffff, 0.45);
    scene.add(ambient);
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(5, 12, 8);
    scene.add(dir);

    // Arena floor — 1 m grid (matches the 2D overlay)
    const grid = new THREE.GridHelper(20, 20, 0x475569, 0x1e3a5f);
    grid.position.set(0, 0, ARENA_D / 2);
    scene.add(grid);
    // Depth-direction grid (10.8 m, we round to 11 cells)
    const grid2 = new THREE.GridHelper(12, 12, 0x475569, 0x1e3a5f);
    grid2.rotation.x = Math.PI / 2;
    grid2.position.set(0, 3, 0);
    grid2.material.opacity = 0.0;
    // keep subtle — 1 grid is enough; the floor grid is what matters

    // Arena box (wireframe)
    const boxGeom = new THREE.BoxGeometry(ARENA_W, ARENA_H, ARENA_D);
    const boxMat = new THREE.LineBasicMaterial({color: 0x3b82f6, transparent: true, opacity: 0.4});
    const box = new THREE.LineSegments(new THREE.EdgesGeometry(boxGeom), boxMat);
    box.position.set(0, ARENA_H / 2, ARENA_D / 2);
    scene.add(box);

    // Arena origin marker (green corner cube)
    const originGeom = new THREE.BoxGeometry(0.3, 0.3, 0.3);
    const origin = new THREE.Mesh(originGeom, new THREE.MeshStandardMaterial({color: 0x10b981}));
    origin.position.set(0, 0.15, 0);
    scene.add(origin);

    // Helper — build a floating canvas-sprite label with a given text.
    // Canvas resolution is fixed; sprite.scale controls world size.
    function _makeLabelSprite(text, bgRGBA, fgRGB, fontPx) {
      const canvas = document.createElement('canvas');
      canvas.width = 128; canvas.height = 56;
      const c = canvas.getContext('2d');
      c.fillStyle = bgRGBA;
      // Rounded rect for a nicer badge look
      const r = 8;
      c.beginPath();
      c.moveTo(r, 0);
      c.lineTo(canvas.width - r, 0);
      c.quadraticCurveTo(canvas.width, 0, canvas.width, r);
      c.lineTo(canvas.width, canvas.height - r);
      c.quadraticCurveTo(canvas.width, canvas.height, canvas.width - r, canvas.height);
      c.lineTo(r, canvas.height);
      c.quadraticCurveTo(0, canvas.height, 0, canvas.height - r);
      c.lineTo(0, r);
      c.quadraticCurveTo(0, 0, r, 0);
      c.closePath();
      c.fill();
      c.fillStyle = fgRGB;
      c.font = 'bold ' + (fontPx || 32) + 'px monospace';
      c.textAlign = 'center';
      c.textBaseline = 'middle';
      c.fillText(String(text), canvas.width / 2, canvas.height / 2 + 2);
      const tex = new THREE.CanvasTexture(canvas);
      tex.minFilter = THREE.LinearFilter;
      const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: tex, transparent: true, depthTest: false}));
      sprite.renderOrder = 100;   // float on top of everything
      return sprite;
    }

    // Fetch + plot arena markers (from the arena_config we already fetched).
    // Each marker gets a coloured face cube AND a floating ID label so the
    // 3D scene is immediately readable without hovering / guessing.
    if (window.arenaMarkers && Object.keys(window.arenaMarkers).length) {
      for (const [id, m] of Object.entries(window.arenaMarkers)) {
        if (!m.pos) continue;
        const g = new THREE.BoxGeometry(0.5, 0.5, 0.05);
        const col = (m.wall === 'front') ? 0x6366f1
                 : (m.wall === 'back')  ? 0xa855f7
                 : (m.wall === 'left')  ? 0x06b6d4
                 : (m.wall === 'right') ? 0x10b981 : 0x94a3b8;
        // Wrap in a group so we can carry a label sprite alongside the cube.
        const grp = new THREE.Group();
        const mesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({color: col}));
        grp.add(mesh);
        const label = _makeLabelSprite(id, 'rgba(15,23,42,0.85)', '#e2e8f0', 34);
        label.scale.set(0.55, 0.24, 1);     // world units — ~55 cm wide
        label.position.set(0, 0.4, 0);      // above the cube
        grp.add(label);
        grp.position.set(m.pos[0], m.pos[2] || 2, m.pos[1]);
        scene.add(grp);
        // Keep the Mesh in markerMeshes (updateVisibleMarkers mutates the
        // material on the cube, not the group).
        mesh.userData._label = label;       // so updateVisibleMarkers can tint
        mesh.userData._group = grp;
        markerMeshes[id] = mesh;
      }
    }

    window.addEventListener('resize', () => {
      if (!renderer) return;
      const w = container.clientWidth, h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });

    // Render the initial set of target boxes as soon as the scene opens
    syncTargetBoxes();

    function loop() {
      controls.update();
      renderer.render(scene, camera);
      rafId = requestAnimationFrame(loop);
    }
    loop();
  }

  // ── Target-box 3D rendering ────────────────────────────────────
  // Kept separately so we can call it both at scene-init AND every
  // time the textarea / mission status updates the list.
  const targetBoxMeshes = {};   // index → THREE.Mesh
  function syncTargetBoxes() {
    if (!scene) return;
    const boxes = window._targetBoxes || [];
    const claimed = window._missionClaimedBoxes || {};
    const captured = new Set(window._missionCapturedBoxes || []);
    const seen = new Set();
    boxes.forEach((b, i) => {
      if (!b || b.x == null || b.y == null) return;
      const idx = (typeof b.idx === 'number') ? b.idx : i;
      seen.add(idx);
      const isCap = captured.has(idx);
      const isClaimed = Object.values(claimed).includes(idx);
      const team = (b.home_team || '').toLowerCase();
      const col = isCap     ? 0x22c55e
                : isClaimed ? 0xfacc15
                : team === 'red'  ? 0xef4444
                : team === 'blue' ? 0x3b82f6
                                  : 0x94a3b8;
      let mesh = targetBoxMeshes[idx];
      if (!mesh) {
        const group = new THREE.Group();
        // SDC box dimensions (rules §1.2): 57.5×37.5×55 cm closed,
        // up to 73 cm when open. We render 0.575×0.55×0.375 as an
        // approximation sitting on the floor.
        const body = new THREE.Mesh(
          new THREE.BoxGeometry(0.575, 0.55, 0.375),
          new THREE.MeshStandardMaterial({color: col, transparent:true, opacity:0.85}));
        body.position.y = 0.275;
        group.add(body);
        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(0.42, 0.04, 8, 24),
          new THREE.MeshStandardMaterial({color: col, transparent:true, opacity:0.55}));
        ring.rotation.x = Math.PI / 2;
        ring.position.y = 0.02;
        group.add(ring);
        // Label
        const canvas = document.createElement('canvas');
        canvas.width = 128; canvas.height = 36;
        const c = canvas.getContext('2d');
        c.fillStyle = 'rgba(0,0,0,0.7)'; c.fillRect(0,0,128,36);
        c.fillStyle = '#fff'; c.font = 'bold 20px monospace';
        c.fillText('BOX ' + (b.id ?? idx+1), 6, 25);
        const tex = new THREE.CanvasTexture(canvas);
        const sprite = new THREE.Sprite(
          new THREE.SpriteMaterial({map: tex, transparent: true}));
        sprite.scale.set(1.1, 0.32, 1);
        sprite.position.set(0, 0.95, 0);
        group.add(sprite);
        group.userData.body = body;
        group.userData.ring = ring;
        scene.add(group);
        targetBoxMeshes[idx] = group;
        mesh = group;
      }
      // Update position + colour every frame in case they moved or state changed
      mesh.position.set(Number(b.x), 0, Number(b.y));
      if (mesh.userData.body) {
        mesh.userData.body.material.color.setHex(col);
        mesh.userData.body.material.opacity = isCap ? 0.55 : 0.85;
      }
      if (mesh.userData.ring) {
        mesh.userData.ring.material.color.setHex(col);
      }
    });
    // Remove meshes for boxes no longer in the list
    for (const idx of Object.keys(targetBoxMeshes)) {
      if (!seen.has(Number(idx))) {
        scene.remove(targetBoxMeshes[idx]);
        delete targetBoxMeshes[idx];
      }
    }
  }

  function updateDrones(observers) {
    if (!scene) return;
    const DRONE_COLORS = [0xf97316, 0x38bdf8, 0xa78bfa, 0xf472b6, 0xfbbf24];
    let idx = 0;
    const seen = new Set();
    for (const [did, st] of Object.entries(observers || {})) {
      seen.add(did);
      const p = st && (st.pos || st.cam);
      if (!p || p.length < 2) { idx++; continue; }
      let mesh = droneMeshes[did];
      if (!mesh) {
        const col = DRONE_COLORS[idx % DRONE_COLORS.length];
        // Drone = small sphere with a forward nose-cone
        const group = new THREE.Group();
        const body = new THREE.Mesh(
          new THREE.SphereGeometry(0.18, 16, 12),
          new THREE.MeshStandardMaterial({color: col}));
        group.add(body);
        const nose = new THREE.Mesh(
          new THREE.ConeGeometry(0.08, 0.3, 8),
          new THREE.MeshStandardMaterial({color: 0xffffff}));
        nose.rotation.x = Math.PI / 2;
        nose.position.set(0, 0, 0.2);
        group.add(nose);
        // Label (using a sprite)
        const canvas = document.createElement('canvas');
        canvas.width = 128; canvas.height = 32;
        const c = canvas.getContext('2d');
        c.fillStyle = 'rgba(0,0,0,0.7)'; c.fillRect(0,0,128,32);
        c.fillStyle = '#fff'; c.font = 'bold 18px monospace';
        c.fillText(did, 8, 22);
        const tex = new THREE.CanvasTexture(canvas);
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({map: tex, transparent: true}));
        sprite.scale.set(0.9, 0.22, 1);
        sprite.position.set(0, 0.4, 0);
        group.add(sprite);
        scene.add(group);
        mesh = group;
        droneMeshes[did] = mesh;
      }
      // Map ArUco-arena coords → Three.js world coords:
      //   ArUco  x = arena horizontal  →  Three.js X
      //   ArUco  y = arena depth       →  Three.js Z (forward)
      //   ArUco  z = altitude          →  Three.js Y (up)
      // Use a typeof check instead of `||` so a legit z=0 reading isn't
      // silently replaced by the 1.5 m fallback (which was hiding grounded
      // or pre-takeoff drones on the cover of the arena).
      const ax = Number(p[0]) || 0;
      const ay = Number(p[1]) || 0;
      const az = (typeof p[2] === 'number' && isFinite(p[2]))
                   ? p[2]
                   : (Number(st.altitude_m) || 1.5);
      mesh.position.set(ax, Math.max(0, az), ay);
      // Heading: prefer the ArUco-derived direction vector (dx,dy) in the
      // arena frame because it matches the pose we just plotted. Fall back
      // to the compass yaw from drone telemetry if the positioner is stale.
      if (Array.isArray(st.dir) && st.dir.length >= 2 &&
          (st.dir[0]*st.dir[0] + st.dir[1]*st.dir[1]) > 1e-6) {
        const hdg = Math.atan2(st.dir[0], st.dir[1]);   // rad from +Y axis
        mesh.rotation.y = -hdg;
      } else if (typeof st.yaw === 'number') {
        mesh.rotation.y = -st.yaw * Math.PI / 180;
      }
      // Flash opacity if the pose is stale so operators see the drone is
      // running on IMU dead-reckoning rather than live vision.
      const staleAlpha = (st.pos_stale === true) ? 0.55 : 1.0;
      mesh.traverse(o => {
        if (o.material && 'opacity' in o.material) {
          o.material.transparent = staleAlpha < 1.0;
          o.material.opacity = staleAlpha;
        }
      });
      idx++;
    }
    // Remove meshes for drones that disappeared
    for (const did of Object.keys(droneMeshes)) {
      if (!seen.has(did)) {
        scene.remove(droneMeshes[did]);
        delete droneMeshes[did];
      }
    }
  }

  function teardown3D() {
    if (rafId) { cancelAnimationFrame(rafId); rafId = 0; }
    if (renderer) {
      renderer.dispose();
      if (renderer.domElement && renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement);
      }
      renderer = null;
    }
    scene = null; camera = null; controls = null;
    droneMeshes = {}; markerMeshes = {};
    // Remove the HUD overlay DIV so it doesn't show stale coordinates
    // when the 3D view gets re-enabled later.
    if (_hudEl && _hudEl.parentNode) {
      _hudEl.parentNode.removeChild(_hudEl);
    }
    _hudEl = null;
  }

  // ── Visible-marker highlight for the 3D arena ─────────────────────
  // Mirrors the 2D halo/white-border on seen markers. We cache each
  // marker's base colour + emissive on first create and mutate only
  // on state changes — cheaper than rebuilding materials per frame.
  function updateVisibleMarkers(seenIds, refIds) {
    if (!scene) return;
    const seen = new Set((seenIds || []).map(String));
    const refs = new Set((refIds  || []).map(String));
    for (const [id, mesh] of Object.entries(markerMeshes)) {
      if (!mesh || !mesh.material) continue;
      if (mesh.userData._baseColor == null) {
        // Cache originals the first time we touch this mesh.
        mesh.userData._baseColor    = mesh.material.color.getHex();
        mesh.userData._baseEmissive = mesh.material.emissive
                                        ? mesh.material.emissive.getHex() : 0;
        mesh.userData._baseScale    = mesh.scale.x;
      }
      const isSeen = seen.has(String(id));
      const isRef  = refs.has(String(id));
      const lbl = mesh.userData._label;
      if (isSeen) {
        // Bright yellow (or green if actually used for pose fusion).
        const highlightColor = isRef ? 0x22c55e : 0xfbbf24;
        mesh.material.color.setHex(highlightColor);
        if (mesh.material.emissive) {
          mesh.material.emissive.setHex(highlightColor);
          mesh.material.emissiveIntensity = 0.6;
        }
        mesh.scale.setScalar(mesh.userData._baseScale * 1.35);
        if (lbl) lbl.scale.set(0.75, 0.32, 1);  // grow label too
      } else {
        mesh.material.color.setHex(mesh.userData._baseColor);
        if (mesh.material.emissive) {
          mesh.material.emissive.setHex(mesh.userData._baseEmissive);
          mesh.material.emissiveIntensity = 0;
        }
        mesh.scale.setScalar(mesh.userData._baseScale);
        if (lbl) lbl.scale.set(0.55, 0.24, 1);
      }
    }
  }

  // ── Drone position HUD — floating DIV above the 3D container ──────
  // Three.js scenes lack an obvious "show me the active drone's xyz"
  // readout. Add a small top-left overlay that shows per-drone
  // coordinates; falls out of the container's bottom if there are many
  // drones, which is fine for up to ~5.
  let _hudEl = null;
  function _ensureHUD() {
    if (_hudEl) return _hudEl;
    const wrap = document.getElementById('arena3d_wrap');
    if (!wrap) return null;
    _hudEl = document.createElement('div');
    _hudEl.id = 'arena3d_hud';
    _hudEl.style.cssText = [
      'position:absolute', 'top:6px', 'left:6px', 'z-index:10',
      'padding:4px 8px',
      'background:rgba(15,23,42,0.8)',
      'color:#e2e8f0', 'font-family:monospace', 'font-size:11px',
      'line-height:1.4', 'border:1px solid #334155', 'border-radius:4px',
      'pointer-events:none', 'max-width:240px',
    ].join(';') + ';';
    wrap.appendChild(_hudEl);
    return _hudEl;
  }
  function updateDronePositionHUD(observers) {
    const el = _ensureHUD();
    if (!el) return;
    const lines = [];
    const DRONE_COLORS = ['#f97316', '#38bdf8', '#a78bfa', '#f472b6', '#fbbf24'];
    let idx = 0;
    for (const [did, st] of Object.entries(observers || {})) {
      const p = st && (st.pos || st.cam);
      const col = DRONE_COLORS[idx % DRONE_COLORS.length];
      idx++;
      if (!Array.isArray(p) || p.length < 2) {
        lines.push('<span style="color:#64748b;">drone ' + did + ': no fix</span>');
        continue;
      }
      const x = Number(p[0]).toFixed(2);
      const y = Number(p[1]).toFixed(2);
      const z = (typeof p[2] === 'number' && isFinite(p[2])) ? Number(p[2]).toFixed(2) : '–';
      const stale = st.pos_stale ? ' <span style="color:#fbbf24;">stale</span>' : '';
      lines.push(
        '<span style="color:' + col + ';font-weight:700;">drone ' + did + '</span> ' +
        '<span style="color:#38bdf8;">x=' + x + '</span> ' +
        '<span style="color:#4ade80;">y=' + y + '</span> ' +
        '<span style="color:#fb923c;">z=' + z + '</span>' + stale
      );
    }
    if (!lines.length) lines.push('<span style="color:#64748b;">no drones</span>');
    el.innerHTML = lines.join('<br/>');
  }

  // ── Safety-margin box in the 3D arena ─────────────────────────────
  // A translucent red wireframe cuboid indicating the Pi-side arena
  // guard boundary. Created lazily the first time updateSafetyMargin
  // is called; subsequent calls just rescale it.
  let safetyBoxMesh = null;
  function updateSafetyMargin(marginM, engaged) {
    if (!scene) return;
    if (marginM == null || marginM <= 0) {
      if (safetyBoxMesh) { scene.remove(safetyBoxMesh); safetyBoxMesh = null; }
      return;
    }
    // Dimensions: arena minus 2×margin on each horizontal axis; height
    // matches the ceiling (MAX_ALTITUDE_M comes from the safety bar).
    const w = Math.max(0.2, ARENA_W - 2 * marginM);
    const d = Math.max(0.2, ARENA_D - 2 * marginM);
    const ceilM = (window._arenaSafety && window._arenaSafety.ceiling_m)
                    || parseFloat((document.getElementById('ceiling_input') || {}).value)
                    || 5.0;
    const h = Math.max(0.5, ceilM);
    if (!safetyBoxMesh) {
      const geom = new THREE.BoxGeometry(w, h, d);
      const edges = new THREE.EdgesGeometry(geom);
      const mat = new THREE.LineBasicMaterial({
        color: 0xef4444, transparent: true, opacity: 0.35,
      });
      safetyBoxMesh = new THREE.LineSegments(edges, mat);
      scene.add(safetyBoxMesh);
    } else {
      // Rebuild geometry on dimension change — simpler than scaling.
      safetyBoxMesh.geometry.dispose();
      const geom = new THREE.BoxGeometry(w, h, d);
      safetyBoxMesh.geometry = new THREE.EdgesGeometry(geom);
    }
    // Centre of arena: x=0, depth=arena_depth/2, y=height/2.
    safetyBoxMesh.position.set(0, h / 2, ARENA_D / 2);
    // Brighter + thicker line when guard is actively clamping.
    if (safetyBoxMesh.material) {
      safetyBoxMesh.material.opacity = engaged ? 0.85 : 0.35;
      safetyBoxMesh.material.color.setHex(engaged ? 0xfca5a5 : 0xef4444);
    }
  }

  // Expose for fleetPoll()
  window._arena3d = {updateDrones, syncTargetBoxes, updateVisibleMarkers,
                     updateDronePositionHUD, updateSafetyMargin};

  // Toggle wiring — 3D view is shown BY DEFAULT below the 2D canvas.
  // The 2D canvas stays visible the whole time so operators have the
  // top-down reference; unchecking simply tears down the 3D scene.
  const cb = document.getElementById('arena_show_3d');
  const wrap = document.getElementById('arena3d_wrap');
  const container = document.getElementById('arena3d_container');
  function apply3DState() {
    if (cb.checked) {
      wrap.style.display = '';
      if (!scene) {
        try { init3D(container); }
        catch (e) { console.error('[3D] init failed:', e); cb.checked = false; wrap.style.display = 'none'; }
      }
    } else {
      wrap.style.display = 'none';
      teardown3D();
    }
  }
  if (cb && wrap && container) {
    cb.addEventListener('change', apply3DState);
    // Start in the default state — 3D active alongside the 2D canvas.
    // Defer a tick so the container has layout dimensions before THREE initialises.
    setTimeout(apply3DState, 0);
  }
