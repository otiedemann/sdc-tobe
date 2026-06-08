#!/usr/bin/env python3
"""Box marker display + control panel — a tiny Flask web app on port 4000.

Two pages:

  /         Full-page ArUco marker for one box (``?id=1..6``). The marker is
            sized so that, on a fullscreen browser, the printed pattern is
            exactly 18 cm — computed from the ``screensize`` (diagonal inches)
            and ``format`` (aspect ratio, e.g. ``16x9``) query parameters.

  /control  Control panel showing the six boxes, 1-3 on the left and 4-6 on
            the right. A box turns green while a marker page for it is open
            ("subscribed"). Clicking a box toggles the first digit of its
            ArUco id between 3 and 4 (blue <-> red team), which live-updates
            any open marker page for that box.

Box -> ArUco id mapping at startup (second digit is the box number):
    box 1-3 -> 31, 32, 33   (prefix 3)
    box 4-6 -> 44, 45, 46   (prefix 4)

The dictionary is ``DICT_4X4_100`` to match the rest of the arena tooling
(see tools/sphinx-arena/generate_aruco_pngs.py).

Usage
-----
    python3 tools/boxes.py
    # then open http://localhost:4000/control
    #          http://localhost:4000/?id=1&screensize=27&format=16x9
"""
from __future__ import annotations

import argparse
import json
import queue
import threading

import cv2
from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    render_template_string,
    request,
    stream_with_context,
)

ARUCO_DICT_NAME = "DICT_4X4_100"
BOXES = (1, 2, 3, 4, 5, 6)
MARKER_CM = 18.0  # printed marker pattern size when shown fullscreen

# Offered in the marker page's dropdowns (the current value is always added
# too, so a hand-edited URL still shows up as a selectable option).
SCREEN_SIZES = [13.0, 14.0, 15.6, 17.0, 20.0, 21.5, 24.0, 27.0, 32.0, 43.0, 50.0, 55.0, 65.0, 75.0, 86.0]
FORMATS = ["16x9", "16x10", "4x3", "3x2", "21x9", "1x1"]


def fmt_num(x: float) -> str:
    """Drop the trailing .0 from whole numbers (27.0 -> "27", 15.6 -> "15.6")."""
    return str(int(x)) if float(x).is_integer() else str(x)


# --------------------------------------------------------------------------- #
# Shared state + a minimal pub/sub broker for Server-Sent Events.
# --------------------------------------------------------------------------- #
class Broker:
    """Holds the per-box state and fans events out to all connected pages."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # First digit (3 or 4) of each box's ArUco id. Second digit == box id.
        self.prefix = {b: (3 if b <= 3 else 4) for b in BOXES}
        # How many marker pages are currently open ("subscribed") per box.
        self.subscribers = {b: 0 for b in BOXES}
        self._listeners: list[queue.Queue] = []

    def aruco_id(self, box: int) -> int:
        return self.prefix[box] * 10 + box

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "boxes": [
                    {
                        "box": b,
                        "prefix": self.prefix[b],
                        "aruco": self.aruco_id(b),
                        "active": self.subscribers[b] > 0,
                        "count": self.subscribers[b],
                    }
                    for b in BOXES
                ]
            }

    def _broadcast(self, event: dict) -> None:
        for q in list(self._listeners):
            q.put(event)

    def listen(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            self._listeners.append(q)
        return q

    def drop(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._listeners:
                self._listeners.remove(q)

    def toggle(self, box: int) -> dict:
        with self._lock:
            self.prefix[box] = 4 if self.prefix[box] == 3 else 3
            event = {
                "type": "prefix",
                "box": box,
                "prefix": self.prefix[box],
                "aruco": self.aruco_id(box),
            }
        self._broadcast(event)
        return event

    def subscribe(self, box: int) -> None:
        with self._lock:
            self.subscribers[box] += 1
            count = self.subscribers[box]
        self._broadcast(
            {"type": "subscription", "box": box, "count": count, "active": True}
        )

    def unsubscribe(self, box: int) -> None:
        with self._lock:
            self.subscribers[box] = max(0, self.subscribers[box] - 1)
            count = self.subscribers[box]
        self._broadcast(
            {
                "type": "subscription",
                "box": box,
                "count": count,
                "active": count > 0,
            }
        )


broker = Broker()
app = Flask(__name__)

_aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT_NAME))
_marker_cache: dict[int, bytes] = {}
_marker_cache_lock = threading.Lock()


def render_marker_png(aruco_id: int, side_px: int = 1000) -> bytes:
    """Return PNG bytes of the marker pattern (no extra quiet zone)."""
    with _marker_cache_lock:
        cached = _marker_cache.get(aruco_id)
    if cached is not None:
        return cached
    img = cv2.aruco.generateImageMarker(_aruco_dict, aruco_id, side_px)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"failed to encode marker {aruco_id}")
    data = buf.tobytes()
    with _marker_cache_lock:
        _marker_cache[aruco_id] = data
    return data


def sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


# --------------------------------------------------------------------------- #
# Marker page
# --------------------------------------------------------------------------- #
MARKER_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>Box {{ box }} — ArUco {{ aruco }}</title>
<style>
  html, body {
    margin: 0; padding: 0; height: 100%; width: 100%;
    background: #fff; overflow: hidden;
    display: flex; align-items: center; justify-content: center;
    font-family: system-ui, sans-serif;
  }
  #marker {
    image-rendering: pixelated;
    image-rendering: crisp-edges;
    /* a white quiet zone helps detection; it does not count toward 18 cm */
    background: #fff; padding: 24px; box-sizing: content-box;
  }
  #bar {
    position: fixed; top: 12px; left: 50%; transform: translateX(-50%);
    display: flex; align-items: center; gap: 18px;
    background: rgba(255,255,255,0.92); padding: 8px 14px;
    border: 2px solid #000; border-radius: 8px;
    font-size: 14px; color: #000; z-index: 10;
  }
  #bar .group { display: flex; align-items: center; gap: 6px; }
  #bar label { font-weight: 700; }
  #bar select { font-size: 14px; padding: 3px 4px; }
  #bar button {
    font-size: 18px; font-weight: 800; line-height: 1;
    width: 32px; height: 32px; cursor: pointer;
    border: 1px solid #000; border-radius: 6px; background: #f3f3f3;
  }
  #bar button:disabled { opacity: 0.35; cursor: default; }
  #idval { min-width: 88px; text-align: center; font-weight: 700; }
  #idval .aruco { font-size: 18px; }
  #idval .sub { font-size: 11px; color: #666; }
  #hint {
    position: fixed; bottom: 10px; right: 12px;
    font-size: 12px; color: #888;
  }
</style>
</head>
<body>
  <img id="marker" alt="ArUco marker">
  <div id="bar">
    <div class="group">
      <label>ID</label>
      <button id="iddown" title="lower box id">&#9660;</button>
      <span id="idval"><span class="aruco">{{ aruco }}</span><br><span class="sub">box {{ box }}</span></span>
      <button id="idup" title="higher box id">&#9650;</button>
    </div>
    <div class="group">
      <label for="screensize">Screen</label>
      <select id="screensize">
        {% for o in screensize_opts %}
        <option value="{{ o.v }}"{{ ' selected' if o.sel else '' }}>{{ o.label }}</option>
        {% endfor %}
      </select>
    </div>
    <div class="group">
      <label for="format">Format</label>
      <select id="format">
        {% for o in format_opts %}
        <option value="{{ o.v }}"{{ ' selected' if o.sel else '' }}>{{ o.v }}</option>
        {% endfor %}
      </select>
    </div>
  </div>
  <div id="hint">target 18&nbsp;cm · press F11 for fullscreen</div>
<script>
  const MARKER_CM = {{ marker_cm }};
  const MIN_ID = {{ min_id }}, MAX_ID = {{ max_id }};
  let box = {{ box }};
  let aruco = {{ aruco }};
  let screensize = {{ screensize }};
  let fmtW = {{ fmt_w }}, fmtH = {{ fmt_h }};

  // Physical width of the screen (cm) from diagonal + aspect ratio, then
  // CSS-pixels-per-cm. window.screen.width CSS px span the full physical
  // width regardless of devicePixelRatio, so this maps cm -> CSS px.
  function markerPx() {
    const diagCm = screensize * 2.54;
    const widthCm = diagCm * fmtW / Math.hypot(fmtW, fmtH);
    const pxPerCm = window.screen.width / widthCm;
    return Math.round(MARKER_CM * pxPerCm);
  }

  const img = document.getElementById('marker');
  const idDown = document.getElementById('iddown');
  const idUp = document.getElementById('idup');

  function applySize() {
    const px = markerPx();
    img.style.width = px + 'px';
    img.style.height = px + 'px';
  }
  function applyMarker() {
    img.src = '/marker/' + aruco + '.png?t=' + Date.now();
    document.querySelector('#idval .aruco').textContent = aruco;
    document.querySelector('#idval .sub').textContent = 'box ' + box;
    document.title = 'Box ' + box + ' — ArUco ' + aruco;
  }
  function updateUrl() {
    const fmt = fmtW + 'x' + fmtH;
    const qs = '?id=' + box + '&screensize=' + screensize + '&format=' + fmt;
    history.replaceState(null, '', qs);
  }
  function syncButtons() {
    idDown.disabled = box <= MIN_ID;
    idUp.disabled = box >= MAX_ID;
  }

  // Subscribe: opening this stream marks the box "active" on the control
  // panel; closing it (tab close or switching box) drops the subscription.
  let es = null;
  function connect() {
    if (es) es.close();
    es = new EventSource('/events/marker/' + box);
    es.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      // hello and prefix both carry the box's current aruco id.
      if ((msg.type === 'hello' || msg.type === 'prefix') && msg.box === box) {
        aruco = msg.aruco;
        applyMarker();
      }
    };
  }

  function changeId(delta) {
    const next = Math.min(MAX_ID, Math.max(MIN_ID, box + delta));
    if (next === box) return;
    box = next;
    syncButtons();
    applyMarker();   // refresh label immediately; aruco confirmed by hello
    updateUrl();
    connect();       // re-subscribe to the new box
  }
  idDown.addEventListener('click', () => changeId(-1));
  idUp.addEventListener('click', () => changeId(+1));

  document.getElementById('screensize').addEventListener('change', (e) => {
    screensize = parseFloat(e.target.value);
    applySize();
    updateUrl();
  });
  document.getElementById('format').addEventListener('change', (e) => {
    const parts = e.target.value.split(/[xX:/]/);
    fmtW = parseInt(parts[0], 10);
    fmtH = parseInt(parts[1], 10);
    applySize();
    updateUrl();
  });

  syncButtons();
  applySize();
  applyMarker();
  connect();
  window.addEventListener('resize', applySize);
</script>
</body>
</html>
"""


def parse_format(fmt: str) -> tuple[int, int]:
    for sep in ("x", "X", ":", "/"):
        if sep in fmt:
            a, b = fmt.split(sep, 1)
            return int(a), int(b)
    raise ValueError(fmt)


@app.route("/")
def marker_page():
    try:
        box = int(request.args.get("id", 1))
    except ValueError:
        abort(400, "id must be an integer 1-6")
    if box not in BOXES:
        abort(400, "id must be 1-6")
    try:
        screensize = float(request.args.get("screensize", 27))
    except ValueError:
        abort(400, "screensize must be a number (inches)")
    fmt = request.args.get("format", "16x9")
    try:
        fmt_w, fmt_h = parse_format(fmt)
    except ValueError:
        abort(400, "format must look like 16x9 or 4x3")

    # Build dropdown options, always including the current value.
    sizes = sorted(set(SCREEN_SIZES) | {screensize})
    screensize_opts = [
        {"v": fmt_num(s), "label": fmt_num(s) + '"', "sel": s == screensize}
        for s in sizes
    ]
    fmt_norm = f"{fmt_w}x{fmt_h}"
    formats = list(dict.fromkeys(FORMATS + [fmt_norm]))
    format_opts = [{"v": f, "sel": f == fmt_norm} for f in formats]

    return render_template_string(
        MARKER_HTML,
        box=box,
        aruco=broker.aruco_id(box),
        screensize=fmt_num(screensize),
        fmt_w=fmt_w,
        fmt_h=fmt_h,
        marker_cm=MARKER_CM,
        min_id=min(BOXES),
        max_id=max(BOXES),
        screensize_opts=screensize_opts,
        format_opts=format_opts,
    )


@app.route("/marker/<int:aruco_id>.png")
def marker_png(aruco_id: int):
    if not (0 <= aruco_id < 100):
        abort(404)
    return Response(render_marker_png(aruco_id), mimetype="image/png")


@app.route("/events/marker/<int:box>")
def events_marker(box: int):
    if box not in BOXES:
        abort(404)

    @stream_with_context
    def gen():
        q = broker.listen()
        broker.subscribe(box)
        try:
            yield sse({"type": "hello", "box": box, "aruco": broker.aruco_id(box)})
            while True:
                try:
                    yield sse(q.get(timeout=4))
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            broker.unsubscribe(box)
            broker.drop(q)

    return Response(gen(), mimetype="text/event-stream")


# --------------------------------------------------------------------------- #
# Control panel
# --------------------------------------------------------------------------- #
CONTROL_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Box control panel</title>
<style>
  :root { color-scheme: light dark; }
  body {
    margin: 0; font-family: system-ui, sans-serif; background: #1e1e1e;
    color: #eee; min-height: 100vh;
  }
  h1 { text-align: center; font-size: 20px; padding: 16px 0 4px; margin: 0; }
  p.sub { text-align: center; margin: 0 0 20px; color: #aaa; font-size: 13px; }
  .grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
    max-width: 720px; margin: 0 auto; padding: 0 16px 32px;
  }
  .box {
    border: 3px solid #555; border-radius: 12px; padding: 22px 16px;
    background: #2b2b2b; cursor: pointer; user-select: none;
    transition: background .15s, border-color .15s, transform .05s;
    text-align: center;
  }
  .box:hover { border-color: #888; }
  .box:active { transform: scale(0.98); }
  .box.active { border-color: #2ecc40; background: #13361a; }
  .box .name { font-size: 15px; color: #bbb; }
  .box .aruco { font-size: 40px; font-weight: 800; margin: 6px 0; letter-spacing: 1px; }
  .box .state { font-size: 12px; color: #888; }
  .box.active .state { color: #2ecc40; }
  .box .team { font-size: 12px; font-weight: 700; }
  .box .team.t3 { color: #4aa3ff; }
  .box .team.t4 { color: #ff5a5a; }
</style>
</head>
<body>
  <h1>Box control panel</h1>
  <p class="sub">Green = a marker page is open · click to toggle ArUco 3↔4</p>
  <div class="grid" id="grid"></div>
<script>
  const BOXES = [1, 2, 3, 4, 5, 6];
  const state = {};

  function render(b) {
    const el = document.getElementById('box-' + b.box);
    el.classList.toggle('active', b.active);
    el.querySelector('.aruco').textContent = b.aruco;
    const team = b.prefix === 3 ? 'blue' : 'red';
    const teamEl = el.querySelector('.team');
    teamEl.textContent = team.toUpperCase() + ' team';
    teamEl.className = 'team t' + b.prefix;
    el.querySelector('.state').textContent =
      b.active ? ('subscribed' + (b.count > 1 ? ' ×' + b.count : '')) : 'not open';
  }

  function buildGrid(boxes) {
    const grid = document.getElementById('grid');
    grid.innerHTML = '';
    // Left column 1-3, right column 4-6: interleave so the CSS grid lays
    // them out as two side-by-side columns row by row.
    const left = boxes.filter(b => b.box <= 3);
    const right = boxes.filter(b => b.box >= 4);
    for (let i = 0; i < 3; i++) {
      for (const b of [left[i], right[i]]) {
        state[b.box] = b;
        const el = document.createElement('div');
        el.className = 'box';
        el.id = 'box-' + b.box;
        el.innerHTML =
          '<div class="name">Box ' + b.box + '</div>' +
          '<div class="aruco"></div>' +
          '<div class="team"></div>' +
          '<div class="state"></div>';
        el.addEventListener('click', () => {
          fetch('/toggle/' + b.box, { method: 'POST' });
        });
        grid.appendChild(el);
        render(b);
      }
    }
  }

  fetch('/state').then(r => r.json()).then(s => buildGrid(s.boxes));

  const es = new EventSource('/events/control');
  es.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (!('box' in msg) || !(msg.box in state)) return;
    const b = state[msg.box];
    if (msg.type === 'prefix') { b.prefix = msg.prefix; b.aruco = msg.aruco; }
    else if (msg.type === 'subscription') { b.active = msg.active; b.count = msg.count; }
    render(b);
  };
</script>
</body>
</html>
"""


@app.route("/control")
def control_page():
    return render_template_string(CONTROL_HTML)


@app.route("/state")
def state():
    return jsonify(broker.snapshot())


@app.route("/toggle/<int:box>", methods=["POST"])
def toggle(box: int):
    if box not in BOXES:
        abort(404)
    return jsonify(broker.toggle(box))


@app.route("/events/control")
def events_control():
    @stream_with_context
    def gen():
        q = broker.listen()
        try:
            # Send the current snapshot so a freshly opened panel is in sync.
            for b in broker.snapshot()["boxes"]:
                yield sse({"type": "subscription", **b})
            while True:
                try:
                    yield sse(q.get(timeout=4))
                except queue.Empty:
                    yield ": keep-alive\n\n"
        finally:
            broker.drop(q)

    return Response(gen(), mimetype="text/event-stream")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=4000)
    args = p.parse_args()
    print(f"boxes.py — control: http://localhost:{args.port}/control")
    print(f"           marker:  http://localhost:{args.port}/?id=1&screensize=27&format=16x9")
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
