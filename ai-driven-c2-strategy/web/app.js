
const viewport = document.getElementById('viewport');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x070b12);

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 1000);
const cameraTarget = new THREE.Vector3(0, 5, 1.6);
const cameraState = {
  radius: 22,
  theta: 0,
  phi: Math.PI / 3,
};

const renderer = new THREE.WebGLRenderer({ antialias: true });
viewport.appendChild(renderer.domElement);

const root = new THREE.Group();
scene.add(root);

const objects = new Map();
const commandLines = new Map();
const pointerState = {
  active: false,
  mode: 'rotate',
  pointerId: null,
  lastX: 0,
  lastY: 0,
  touches: new Map(),
  pinchDistance: 0,
};

scene.add(new THREE.AmbientLight(0xffffff, 0.9));
const light = new THREE.DirectionalLight(0xffffff, 0.8);
light.position.set(0, -6, 10);
scene.add(light);

function resize() {
  const rect = viewport.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height);
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();
updateCamera();

function updateCamera() {
  const sinPhi = Math.sin(cameraState.phi);
  camera.position.set(
    cameraTarget.x + cameraState.radius * sinPhi * Math.sin(cameraState.theta),
    cameraTarget.y - cameraState.radius * sinPhi * Math.cos(cameraState.theta),
    cameraTarget.z + cameraState.radius * Math.cos(cameraState.phi)
  );
  camera.lookAt(cameraTarget);
}

function clampCamera() {
  cameraState.radius = Math.min(Math.max(cameraState.radius, 6), 60);
  cameraState.phi = Math.min(Math.max(cameraState.phi, 0.18), Math.PI / 2 - 0.05);
  cameraTarget.x = Math.min(Math.max(cameraTarget.x, -16), 16);
  cameraTarget.y = Math.min(Math.max(cameraTarget.y, -5), 15);
  cameraTarget.z = Math.min(Math.max(cameraTarget.z, 0), 8);
}

function rotateView(dx, dy) {
  cameraState.theta -= dx * 0.006;
  cameraState.phi -= dy * 0.004;
  clampCamera();
  updateCamera();
}

function panView(dx, dy) {
  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const up = new THREE.Vector3().crossVectors(right, forward).normalize();
  const scale = cameraState.radius * 0.0018;
  cameraTarget.addScaledVector(right, -dx * scale);
  cameraTarget.addScaledVector(up, dy * scale);
  clampCamera();
  updateCamera();
}

function zoomView(delta) {
  cameraState.radius *= Math.exp(delta * 0.001);
  clampCamera();
  updateCamera();
}

function pointerMode(event) {
  if (event.button === 1 || event.button === 2 || event.shiftKey || event.ctrlKey) {
    return 'pan';
  }
  return 'rotate';
}

renderer.domElement.addEventListener('contextmenu', (event) => event.preventDefault());
renderer.domElement.addEventListener('wheel', (event) => {
  event.preventDefault();
  zoomView(event.deltaY);
}, { passive: false });

renderer.domElement.addEventListener('pointerdown', (event) => {
  renderer.domElement.setPointerCapture(event.pointerId);
  pointerState.active = true;
  pointerState.pointerId = event.pointerId;
  pointerState.lastX = event.clientX;
  pointerState.lastY = event.clientY;
  pointerState.mode = pointerMode(event);
  pointerState.touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
  pointerState.pinchDistance = currentPinchDistance();
});

renderer.domElement.addEventListener('pointermove', (event) => {
  if (pointerState.touches.has(event.pointerId)) {
    pointerState.touches.set(event.pointerId, { x: event.clientX, y: event.clientY });
  }

  if (pointerState.touches.size >= 2) {
    const distance = currentPinchDistance();
    if (pointerState.pinchDistance > 0 && distance > 0) {
      zoomView((pointerState.pinchDistance - distance) * 4);
    }
    pointerState.pinchDistance = distance;
    return;
  }

  if (!pointerState.active || pointerState.pointerId !== event.pointerId) return;
  const dx = event.clientX - pointerState.lastX;
  const dy = event.clientY - pointerState.lastY;
  pointerState.lastX = event.clientX;
  pointerState.lastY = event.clientY;

  if (pointerState.mode === 'pan') {
    panView(dx, dy);
  } else {
    rotateView(dx, dy);
  }
});

renderer.domElement.addEventListener('pointerup', endPointer);
renderer.domElement.addEventListener('pointercancel', endPointer);

function endPointer(event) {
  pointerState.touches.delete(event.pointerId);
  if (pointerState.pointerId === event.pointerId) {
    pointerState.active = false;
    pointerState.pointerId = null;
  }
  pointerState.pinchDistance = currentPinchDistance();
}

function currentPinchDistance() {
  const touches = Array.from(pointerState.touches.values());
  if (touches.length < 2) return 0;
  const dx = touches[0].x - touches[1].x;
  const dy = touches[0].y - touches[1].y;
  return Math.hypot(dx, dy);
}

function makeLine(points, color) {
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  return new THREE.Line(geometry, new THREE.LineBasicMaterial({ color }));
}

function drawArena() {
  const floor = new THREE.GridHelper(20, 20, 0x2a3955, 0x172239);
  floor.rotation.x = Math.PI / 2;
  floor.position.set(0, 5, 0);
  root.add(floor);

  const box = new THREE.BoxGeometry(20, 10, 6);
  const edges = new THREE.EdgesGeometry(box);
  const wire = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: 0x5d789f }));
  wire.position.set(0, 5, 3);
  root.add(wire);

  addZone(-7.5, 5, 0.01, 5, 10, 0x173a78);
  addZone(0, 5, 0.015, 10, 10, 0x273145);
  addZone(7.5, 5, 0.02, 5, 10, 0x6c1c24);
}

function addZone(x, y, z, width, depth, color) {
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(width, depth),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.22, side: THREE.DoubleSide })
  );
  mesh.position.set(x, y, z);
  root.add(mesh);
}

drawArena();

function upsertSphere(key, radius, color) {
  let mesh = objects.get(key);
  if (!mesh) {
    mesh = new THREE.Mesh(
      new THREE.SphereGeometry(radius, 24, 16),
      new THREE.MeshStandardMaterial({ color, roughness: 0.45 })
    );
    root.add(mesh);
    objects.set(key, mesh);
  }
  mesh.material.color.setHex(color);
  return mesh;
}

function upsertTarget(key, color) {
  let mesh = objects.get(key);
  if (!mesh) {
    mesh = new THREE.Mesh(
      new THREE.BoxGeometry(0.7, 0.45, 0.55),
      new THREE.MeshStandardMaterial({ color, roughness: 0.35 })
    );
    root.add(mesh);
    objects.set(key, mesh);
  }
  mesh.material.color.setHex(color);
  return mesh;
}

function colorForOwner(owner) {
  return owner === 'blue' ? 0x2c7dff : 0xf13b4a;
}

function updateLine(key, from, to, color) {
  const old = commandLines.get(key);
  if (old) root.remove(old);
  const line = makeLine([
    new THREE.Vector3(from.x, from.y, from.z),
    new THREE.Vector3(to.x, to.y, to.z),
  ], color);
  root.add(line);
  commandLines.set(key, line);
}

function updateScene(data) {
  const match = data.match_state;
  const commandsByDrone = new Map((data.plan.commands || []).map((cmd) => [cmd.drone_id, cmd]));

  for (const target of match.targets) {
    const mesh = upsertTarget(`target:${target.id}`, colorForOwner(target.owner));
    mesh.position.set(target.position.x, target.position.y, 0.28);
  }

  for (const drone of match.own_drones) {
    const mesh = upsertSphere(`drone:${drone.id}`, 0.22, 0x48a0ff);
    mesh.position.set(drone.pose.x, drone.pose.y, drone.pose.z);
    if (drone.command_position) {
      updateLine(`line:${drone.id}`, drone.pose, drone.command_position, 0x7bbcff);
      const ghost = upsertSphere(`ghost:${drone.id}`, 0.11, 0x9bc9ff);
      ghost.position.set(drone.command_position.x, drone.command_position.y, drone.command_position.z);
    }
  }

  for (const drone of match.enemy_drones) {
    const mesh = upsertSphere(`drone:${drone.id}`, 0.22, 0xff4a5c);
    mesh.position.set(drone.pose.x, drone.pose.y, drone.pose.z);
  }

  document.getElementById('time').textContent = `${match.match_time_seconds.toFixed(1)}s / ${match.time_remaining_seconds.toFixed(0)}s left`;
  document.getElementById('score').textContent = `${match.score.own} : ${match.score.opponent}`;
  document.getElementById('planner').textContent = `${data.plan.source} / ${data.plan.mode}`;
  document.getElementById('summary').textContent = data.plan.strategy_summary || '-';

  const events = document.getElementById('events');
  events.innerHTML = '';
  for (const event of match.recent_events.slice(-8).reverse()) {
    const li = document.createElement('li');
    li.textContent = `${event.time_seconds.toFixed(1)}s: ${event.message}`;
    events.appendChild(li);
  }
}

function connectStateStream() {
  if (!window.EventSource) {
    document.getElementById('summary').textContent = 'This browser does not support Server-Sent Events.';
    return;
  }

  const events = new EventSource('/events');
  events.addEventListener('state', (event) => {
    updateScene(JSON.parse(event.data));
  });
  events.onerror = () => {
    document.getElementById('summary').textContent = 'SSE connection interrupted; reconnecting...';
  };
}

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}

connectStateStream();
animate();
