"""C2 V2 dashboard — Phase 1: read-only live view of the five drones.

Shows, per drone: its arena position (from ArUco vision) on a top-down map,
connection + telemetry + battery, the FC phase, and what it's doing. No
commands yet (arming, roles, and the coordinator land in later phases).

Self-contained Flask app (plain HTML + canvas + fetch, no build step) so it
runs offline at the venue.
"""
from __future__ import annotations

from typing import Any, Dict

from flask import Flask, Response, jsonify

# Reuse the live arena geometry (real/gvz) for the map + box layout.
from marker_mission_c2.strategy import arena_state

from .pool import FCPool


def _no_cache(resp: Response) -> Response:
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


def build_app(pool: FCPool) -> Flask:
    app = Flask(__name__)

    @app.route("/api/state")
    def api_state():
        snap = pool.snapshot()
        try:
            snap["arena"] = arena_state.to_client_dict()
        except Exception as exc:  # arena registry missing -> still serve drones
            snap["arena"] = {"error": f"{type(exc).__name__}: {exc}"}
        return _no_cache(jsonify(snap))

    @app.route("/")
    def index():
        return _no_cache(Response(_PAGE, mimetype="text/html"))

    return app


_PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>C2 V2 — live</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { --bg:#0d1117; --panel:#161b22; --panel2:#21262d; --fg:#e6edf3;
          --muted:#8b949e; --green:#2ea043; --red:#f85149; --blue:#388bfd;
          --warn:#d29922; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:13px/1.4 -apple-system,Segoe UI,Roboto,sans-serif; }
  header { display:flex; align-items:center; gap:10px; padding:10px 14px;
           background:var(--panel); border-bottom:1px solid #30363d; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  .pill { background:var(--panel2); border-radius:10px; padding:2px 8px;
          font-size:12px; color:var(--muted); }
  .pill.ok { color:var(--green); } .pill.bad { color:var(--red); }
  main { display:grid; grid-template-columns: 1fr 1fr; gap:12px; padding:12px; }
  @media (max-width:900px){ main { grid-template-columns:1fr; } }
  section { background:var(--panel); border:1px solid #30363d; border-radius:8px;
            padding:10px; }
  h2 { font-size:13px; margin:0 0 8px; color:var(--muted); font-weight:600; }
  canvas { background:#0b0f14; border:1px solid #30363d; border-radius:6px;
           width:100%; height:auto; display:block; }
  .drones { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
            gap:10px; }
  .drone { background:var(--panel2); border:1px solid #30363d; border-radius:8px;
           padding:8px; }
  .drone.off { opacity:.5; }
  .drone .name { font-weight:600; display:flex; justify-content:space-between; }
  .dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
  .dot.ok{background:var(--green);} .dot.bad{background:var(--red);}
  .kv { display:grid; grid-template-columns:auto 1fr; gap:1px 8px; margin-top:6px;
        font-size:12px; }
  .kv .k { color:var(--muted); } .kv .v { text-align:right; }
  .bat { color:var(--green); } .bat.low { color:var(--red); }
  .plan { margin-top:6px; font-size:12px; color:var(--blue); word-break:break-word; }
  .small { font-size:11px; color:var(--muted); }
</style></head>
<body>
<header>
  <h1>C2 V2 <span class="small">· read-only live view</span></h1>
  <span class="pill" id="arena-pill">arena: –</span>
  <span class="pill" id="conn-pill">– / – connected</span>
  <span class="pill" id="age-pill"></span>
</header>
<main>
  <section>
    <h2>Arena <span class="small">(top-down · live ArUco positions)</span></h2>
    <canvas id="map" width="520" height="280"></canvas>
  </section>
  <section>
    <h2>Drones</h2>
    <div class="drones" id="drones"></div>
  </section>
</main>
<script>
const TEAM_COL = { red:"#f85149", blue:"#388bfd" };
async function tick(){
  let s;
  try { s = await (await fetch("/api/state",{cache:"no-store"})).json(); }
  catch(e){ document.getElementById("conn-pill").textContent = "fetch failed"; return; }
  renderHeader(s); renderDrones(s); renderMap(s);
}
function renderHeader(s){
  const a = s.arena||{};
  document.getElementById("arena-pill").textContent = "arena: " + (a.name||"?");
  const cp = document.getElementById("conn-pill");
  cp.textContent = s.connected_count + " / " + s.fc_count + " connected";
  cp.className = "pill " + (s.connected_count===s.fc_count ? "ok" : (s.connected_count? "" : "bad"));
}
function renderDrones(s){
  const root = document.getElementById("drones");
  root.innerHTML = "";
  for(const d of s.drones){
    const el = document.createElement("div");
    el.className = "drone" + (d.connected? "":" off");
    const pos = d.position_m ? d.position_m.map(x=>x.toFixed(1)).join(", ") : "—";
    const bat = d.battery_pct==null? "—" : Math.round(d.battery_pct)+"%";
    el.innerHTML = `
      <div class="name">${d.name}
        <span><span class="dot ${d.connected?'ok':'bad'}"></span></span></div>
      <div class="kv">
        <div class="k">pos (x,y,z)</div><div class="v">${pos}</div>
        <div class="k">height</div><div class="v">${d.height_m==null?'—':d.height_m+' m'}</div>
        <div class="k">battery</div><div class="v bat ${d.battery_low?'low':''}">${bat}</div>
        <div class="k">drone link</div><div class="v">${d.drone_connected?'ok':'—'}${d.flying?' · flying':''}</div>
        <div class="k">markers</div><div class="v">${(d.visible_marker_ids||[]).join(',')||'—'}</div>
        <div class="k">wifi</div><div class="v small">${d.wifi_ssid||'—'}</div>
        <div class="k">serial</div><div class="v small">${d.serial||'—'}</div>
      </div>
      <div class="plan">${d.plan||''}</div>
      ${d.note? `<div class="small">${d.note}</div>`:''}
      ${d.last_error && !d.connected? `<div class="small" style="color:var(--red)">${d.last_error}</div>`:''}
    `;
    root.appendChild(el);
  }
}
function renderMap(s){
  const a = s.arena; const c = document.getElementById("map");
  const ctx = c.getContext("2d"); ctx.clearRect(0,0,c.width,c.height);
  if(!a || a.x_min==null){ return; }
  // World: x in [x_min,x_max] (width, short axis), y in [y_min,y_max] (length).
  // Draw length (y) horizontally so the 20x10 field is landscape.
  const pad = 16;
  const W = c.width - 2*pad, H = c.height - 2*pad;
  const yspan = a.y_max - a.y_min, xspan = a.x_max - a.x_min;
  const sx = (y)=> pad + ( (y - a.y_min)/yspan )*W;   // world y -> screen x
  const sy = (x)=> pad + ( (a.x_max - x)/xspan )*H;    // world x -> screen y
  // field
  ctx.strokeStyle="#30363d"; ctx.strokeRect(pad,pad,W,H);
  // home zones (red at y<0 end, blue at y>0 end), 5m deep
  const rh = a.red_home_y||[a.y_min, a.y_min+5];
  const bh = a.blue_home_y||[a.y_max-5, a.y_max];
  ctx.fillStyle="rgba(248,81,73,.10)"; ctx.fillRect(sx(rh[0]),pad,sx(rh[1])-sx(rh[0]),H);
  ctx.fillStyle="rgba(56,139,253,.10)"; ctx.fillRect(sx(bh[0]),pad,sx(bh[1])-sx(bh[0]),H);
  // boxes
  const boxes = a.boxes||{};
  ctx.font="10px sans-serif";
  for(const slot in boxes){
    const [bx,by]=boxes[slot];
    ctx.fillStyle = (slot<=3)?"#f85149":"#388bfd";
    ctx.fillRect(sx(by)-4, sy(bx)-4, 8, 8);
    ctx.fillStyle="#8b949e"; ctx.fillText(slot, sx(by)+6, sy(bx)+3);
  }
  // drones
  for(const d of s.drones){
    if(!d.position_m) continue;
    const [x,y]=d.position_m;
    const X=sx(y), Y=sy(x);
    ctx.beginPath(); ctx.arc(X,Y,6,0,7);
    ctx.fillStyle = d.connected? "#2ea043":"#8b949e"; ctx.fill();
    ctx.fillStyle="#e6edf3"; ctx.font="10px sans-serif";
    ctx.fillText(d.name.replace("flightctrl","fc"), X+8, Y+3);
    // yaw arrow
    if(d.drone_yaw_deg!=null){
      const r=12, th=(d.drone_yaw_deg)*Math.PI/180;
      ctx.beginPath(); ctx.moveTo(X,Y);
      ctx.lineTo(X+r*Math.sin(th), Y-r*Math.cos(th));
      ctx.strokeStyle="#2ea043"; ctx.stroke();
    }
  }
}
tick(); setInterval(tick, 1000);
</script>
</body></html>
"""
