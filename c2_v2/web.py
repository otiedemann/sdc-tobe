"""C2 V2 dashboard — Phase 1: read-only live view of the five drones.

Shows, per drone: its arena position (from ArUco vision) on a top-down map,
connection + telemetry + battery, the FC phase, and what it's doing. No
commands yet (arming, roles, and the coordinator land in later phases).

Self-contained Flask app (plain HTML + canvas + fetch, no build step) so it
runs offline at the venue.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from flask import Flask, Response, jsonify, request

# Reuse the live arena geometry (real/gvz) for the map + box layout.
from marker_mission_c2.strategy import arena_state

from .pool import FCPool


def _no_cache(resp: Response) -> Response:
    resp.headers["Cache-Control"] = "no-store, max-age=0"
    return resp


def build_app(pool: FCPool, match: Any = None, runner: Any = None,
              loop: Any = None) -> Flask:
    app = Flask(__name__)

    def _run_coro(coro, timeout: float = 8.0):
        """Run an async runner/pool coroutine from the Flask thread on the
        asyncio loop."""
        import asyncio
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result(timeout)

    @app.route("/api/state")
    def api_state():
        snap = pool.snapshot()
        try:
            snap["arena_geom"] = arena_state.to_client_dict()
        except Exception as exc:  # arena registry missing -> still serve drones
            snap["arena_geom"] = {"error": f"{type(exc).__name__}: {exc}"}
        snap["match"] = match.to_client() if match is not None else None
        snap["actions"] = runner.last_actions() if runner is not None else {}
        return _no_cache(jsonify(snap))

    @app.route("/api/arm", methods=["POST"])
    def api_arm():
        body = request.get_json(silent=True) or {}
        if match is None:
            return _no_cache(jsonify({"ok": False})), 400
        match.set_armed(bool(body.get("armed", False)))
        return _no_cache(jsonify({"ok": True, "armed": match.armed}))

    @app.route("/api/auto", methods=["POST"])
    def api_auto():
        body = request.get_json(silent=True) or {}
        if match is None:
            return _no_cache(jsonify({"ok": False})), 400
        match.set_auto(bool(body.get("auto", False)))
        return _no_cache(jsonify({"ok": True, "auto": match.auto}))

    @app.route("/api/emergency-land", methods=["POST"])
    def api_emergency_land():
        # Land everyone NOW + disarm. Disarm first (cheap, always works), then
        # land. NEVER report success unless drones were actually commanded down.
        if match is not None:
            match.set_armed(False)
        if loop is None:
            return _no_cache(jsonify({
                "ok": False, "error": "no asyncio loop — disarmed only; "
                "use the physical kill switch"})), 503
        try:
            # Prefer the runner (also records per-drone action); fall back to the
            # pool directly so a land still happens even without a runner.
            if runner is not None:
                res = _run_coro(runner.emergency_land(), timeout=12.0)
            else:
                res = _run_coro(pool.emergency_land_all(), timeout=12.0)
        except Exception as exc:
            return _no_cache(jsonify({"ok": False, "error": str(exc)})), 502
        return _no_cache(jsonify({"ok": True, "results": res}))

    @app.route("/api/drone/<fc>/role", methods=["POST"])
    def api_role(fc):
        body = request.get_json(silent=True) or {}
        if match is None or not match.set_role(fc, str(body.get("role", ""))):
            return _no_cache(jsonify({"ok": False, "error": "bad role/fc"})), 400
        return _no_cache(jsonify({"ok": True}))

    @app.route("/api/drone/<fc>/enabled", methods=["POST"])
    def api_enabled(fc):
        body = request.get_json(silent=True) or {}
        if match is None or not match.set_enabled(fc, bool(body.get("enabled", True))):
            return _no_cache(jsonify({"ok": False, "error": "bad fc"})), 400
        return _no_cache(jsonify({"ok": True}))

    @app.route("/api/tune", methods=["POST"])
    def api_tune():
        body = request.get_json(silent=True) or {}
        if match is None or not match.set_tunable(str(body.get("key", "")),
                                                  body.get("value")):
            return _no_cache(jsonify({"ok": False, "error": "bad tunable"})), 400
        return _no_cache(jsonify({"ok": True}))

    @app.route("/api/team", methods=["POST"])
    def api_team():
        body = request.get_json(silent=True) or {}
        if match is None or not match.set_team(str(body.get("team", ""))):
            return _no_cache(jsonify({"ok": False, "error": "team must be red|blue"})), 400
        return _no_cache(jsonify({"ok": True, "our_team": match.our_team}))

    @app.route("/api/arena", methods=["POST"])
    def api_arena():
        body = request.get_json(silent=True) or {}
        name = str(body.get("name", ""))
        if match is None or not match.set_arena(name):
            return _no_cache(jsonify({"ok": False, "error": f"unknown arena {name!r}"})), 400
        return _no_cache(jsonify({"ok": True, "arena": arena_state.active_name()}))

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
           background:var(--panel); border-bottom:1px solid #30363d; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0; font-weight:600; }
  select { background:var(--panel2); color:var(--fg); border:1px solid #30363d;
           border-radius:6px; padding:3px 6px; font-size:12px; }
  .pill { background:var(--panel2); border-radius:10px; padding:2px 8px;
          font-size:12px; color:var(--muted); }
  .pill.ok { color:var(--green); } .pill.bad { color:var(--red); }
  .pill.win { color:#0d1117; background:var(--warn); font-weight:700; }
  .pill.red { color:var(--red); } .pill.blue { color:var(--blue); }
  .pill.armed { color:#0d1117; background:var(--green); font-weight:700; }
  button { background:var(--panel2); color:var(--fg); border:1px solid #30363d;
           border-radius:6px; padding:4px 10px; font-size:12px; cursor:pointer; }
  button.armed { background:var(--green); color:#0d1117; font-weight:700; border-color:var(--green); }
  button.danger { background:var(--red); color:#fff; font-weight:700; border-color:var(--red); }
  .spacer { flex:1; }
  main { display:grid; grid-template-columns: 1fr 1fr; gap:12px; padding:12px; }
  @media (max-width:900px){ main { grid-template-columns:1fr; } }
  section { background:var(--panel); border:1px solid #30363d; border-radius:8px;
            padding:10px; }
  section.wide { grid-column: 1 / -1; }
  h2 { font-size:13px; margin:0 0 8px; color:var(--muted); font-weight:600; }
  canvas { background:#0b0f14; border:1px solid #30363d; border-radius:6px;
           width:100%; height:auto; display:block; }
  .drones { display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr)); gap:10px; }
  .drone { background:var(--panel2); border:1px solid #30363d; border-radius:8px; padding:8px; }
  .drone.off { opacity:.5; }
  .drone .name { font-weight:600; display:flex; justify-content:space-between; }
  .dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
  .dot.ok{background:var(--green);} .dot.bad{background:var(--red);}
  .kv { display:grid; grid-template-columns:auto 1fr; gap:1px 8px; margin-top:6px; font-size:12px; }
  .kv .k { color:var(--muted); } .kv .v { text-align:right; }
  .bat { color:var(--green); } .bat.low { color:var(--red); }
  .plan { margin-top:6px; font-size:12px; color:var(--blue); word-break:break-word; }
  .small { font-size:11px; color:var(--muted); }
  .boxes { display:grid; grid-template-columns:repeat(6,1fr); gap:8px; }
  @media (max-width:700px){ .boxes { grid-template-columns:repeat(3,1fr); } }
  .box { border:1px solid #30363d; border-radius:8px; padding:8px; text-align:center;
         background:var(--panel2); }
  .box .id { font-weight:700; font-size:14px; }
  .box .hold { margin-top:4px; font-weight:600; border-radius:6px; padding:2px 4px; }
  .hold.red { background:rgba(248,81,73,.20); color:var(--red); }
  .hold.blue { background:rgba(56,139,253,.20); color:var(--blue); }
  .hold.unknown { background:var(--panel); color:var(--muted); }
  .box .tag { margin-top:4px; font-size:11px; }
  .tag.ours { color:var(--green); } .tag.enemy { color:var(--red); }
  .box .lock { color:var(--warn); font-size:11px; }
</style></head>
<body>
<header>
  <h1>C2 V2</h1>
  <span class="small">team</span>
  <select id="team-sel"><option value="red">red</option><option value="blue">blue</option></select>
  <span class="small">arena</span>
  <select id="arena-sel"></select>
  <button id="auto-btn" title="AUTO: the coordinator assigns roles from the game state. MANUAL: you assign roles.">Go AUTO</button>
  <button id="arm-btn">Arm</button>
  <button id="land-btn" class="danger">EMERGENCY LAND</button>
  <span class="pill" id="armed-pill">disarmed</span>
  <span class="pill" id="mode-pill">MANUAL</span>
  <span class="spacer"></span>
  <span class="pill" id="conn-pill">– / – connected</span>
  <span class="pill" id="boxes-pill">boxes –</span>
  <span class="pill" id="score-pill">score –</span>
  <span class="pill" id="special-pill" style="display:none">special</span>
</header>
<div id="warn-banner" style="display:none;background:var(--red);color:#fff;padding:6px 14px;font-weight:600"></div>
<main>
  <section>
    <h2>Arena <span class="small">(top-down · live ArUco positions · box colour = current holder)</span></h2>
    <canvas id="map" width="520" height="280"></canvas>
  </section>
  <section>
    <h2>Drones <span class="small" id="coord-line"></span></h2>
    <div class="drones" id="drones"></div>
  </section>
  <section class="wide">
    <h2>Targets <span class="small">(6 boxes · holder from the scout's view)</span></h2>
    <div class="boxes" id="boxes"></div>
  </section>
  <section class="wide">
    <details>
      <summary style="cursor:pointer;color:var(--muted)">Tuning <span class="small">(live — takes effect on next launch)</span></summary>
      <div class="tune" id="tune" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px;margin-top:8px"></div>
    </details>
    <div class="small" style="margin-top:8px;color:var(--muted)">
      Safety: C2 V2's 5 Hz polling is the v7 §3.3 heartbeat — the FC auto-lands
      a drone within 2 s of losing C2 requests (REMOTE_TIMEOUT_S=2.0). '?' or
      EMERGENCY LAND lands everyone immediately.
    </div>
  </section>
</main>
<script>
const HOLD_COL = { red:"#f85149", blue:"#388bfd", unknown:"#6e7681" };
let arenaInit = false;
async function api(path, body){
  return fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
                      body: JSON.stringify(body)});
}
async function tick(){
  let s;
  try { s = await (await fetch("/api/state",{cache:"no-store"})).json(); }
  catch(e){ document.getElementById("conn-pill").textContent = "fetch failed"; return; }
  renderHeader(s); renderDrones(s); renderBoxes(s); renderMap(s);
  renderTune(s); renderWarn(s);
}
function renderWarn(s){
  const b = document.getElementById("warn-banner");
  const m = s.match||{};
  if(landError){ b.style.display=""; b.textContent = "⚠ " + landError; return; }
  const lost = s.drones.filter(d => !d.connected).map(d=>d.name.replace("flightctrl","fc"));
  if(m.armed && lost.length){
    b.style.display="";
    b.textContent = `⚠ LINK LOST: ${lost.join(", ")} — the FC auto-lands each ~2 s after losing C2 polling. Reconnect or EMERGENCY LAND.`;
  } else { b.style.display="none"; }
}
let tuneBuilt = false;
function renderTune(s){
  const m = s.match||{}; const spec = m.tune_spec; const vals = m.tunables;
  if(!spec || !vals) return;
  const root = document.getElementById("tune");
  if(!tuneBuilt){
    root.innerHTML = spec.map(([key,label,lo,hi,st])=>
      `<label class="small" style="display:flex;justify-content:space-between;gap:6px;align-items:center">
         <span>${label}</span>
         <input type="number" data-tune="${key}" min="${lo}" max="${hi}" step="${st}" style="width:5.5em"></label>`).join("");
    tuneBuilt = true;
  }
  for(const inp of root.querySelectorAll("[data-tune]")){
    const k = inp.getAttribute("data-tune");
    if(document.activeElement !== inp && vals[k]!=null) inp.value = vals[k];
  }
}
function renderHeader(s){
  const m = s.match||{};
  const cp = document.getElementById("conn-pill");
  cp.textContent = s.connected_count + " / " + s.fc_count + " connected";
  cp.className = "pill " + (s.connected_count===s.fc_count ? "ok" : (s.connected_count? "" : "bad"));
  // team select (don't clobber while focused)
  const ts = document.getElementById("team-sel");
  if(document.activeElement!==ts && m.our_team) ts.value = m.our_team;
  // arena select — populate once
  const as = document.getElementById("arena-sel");
  if(!arenaInit && m.arena_options){
    as.innerHTML = Object.entries(m.arena_options).map(([k,v])=>`<option value="${k}">${v}</option>`).join("");
    arenaInit = true;
  }
  if(document.activeElement!==as && m.arena) as.value = m.arena;
  // boxes / score / special
  document.getElementById("boxes-pill").textContent =
    `ours ${m.boxes_ours==null?'–':m.boxes_ours} · enemy ${m.boxes_enemy==null?'–':m.boxes_enemy}`;
  const sc = (m.score)||{};
  document.getElementById("score-pill").textContent =
    `score r:${sc.red||0} b:${sc.blue||0} (est)`;
  const sp = document.getElementById("special-pill"); const ms = m.special||{};
  if(ms.all_ours){
    sp.style.display="";
    sp.textContent = ms.won? "★ INSTANT WIN" : `ALL 6 OURS ${ms.dwell_s.toFixed(1)}/${ms.threshold_s}s`;
    sp.className = "pill " + (ms.won? "win":"");
  } else { sp.style.display="none"; }
  // armed state
  const armed = !!m.armed; armedNow = armed;
  const ap = document.getElementById("armed-pill");
  ap.textContent = armed? "ARMED" : "disarmed";
  ap.className = "pill " + (armed? "armed":"");
  const ab = document.getElementById("arm-btn");
  ab.textContent = armed? "Disarm" : "Arm";
  ab.className = armed? "armed":"";
  // auto/manual mode
  autoNow = !!m.auto;
  const mp = document.getElementById("mode-pill");
  mp.textContent = autoNow? "AUTO" : "MANUAL";
  mp.className = "pill " + (autoNow? "armed":"");
  document.getElementById("auto-btn").textContent = autoNow? "Go MANUAL" : "Go AUTO";
  document.getElementById("coord-line").textContent =
    autoNow? ("· " + (m.coord_summary||"…")) : "";
}
function roleOpts(cur){
  return ["idle","scout","attacker","defender"].map(r=>
    `<option value="${r}" ${r===cur?"selected":""}>${r}</option>`).join("");
}
// Build each drone card ONCE; on later ticks update fields in place. Rebuilding
// innerHTML every second would tear down the role <select> while the operator
// has it open, snapping the dropdown shut before they can pick — so the panel
// is built once and the controls are never recreated (mirrors renderTune).
const droneCards = {};  // name -> {el, refs}
// Names whose role <select> the operator is currently interacting with. While a
// name is in here we do NOT reconcile its <select>.value from the server: writing
// select.value while the native dropdown popup is open CANCELS the operator's
// pick (macOS/Chrome), so the change event never fires and the role never
// switches — which is exactly why the dropdown appeared "stuck" on the old role.
const editingRoles = new Set();
function buildDroneCard(name){
  const el = document.createElement("div");
  el.className = "drone";
  el.innerHTML = `
    <div class="name">${name}<span><span class="dot bad" data-f="dot"></span></span></div>
    <div style="display:flex;gap:6px;align-items:center;margin-top:4px">
      <select data-role="${name}">${roleOpts('idle')}</select>
      <label class="small"><input type="checkbox" data-enabled="${name}"> on</label>
    </div>
    <div class="kv">
      <div class="k">pos (x,y,z)</div><div class="v" data-f="pos">—</div>
      <div class="k">height</div><div class="v" data-f="height">—</div>
      <div class="k">battery</div><div class="v bat" data-f="bat">—</div>
      <div class="k">drone link</div><div class="v" data-f="link">—</div>
      <div class="k">markers</div><div class="v" data-f="markers">—</div>
      <div class="k">wifi</div><div class="v small" data-f="wifi">—</div>
    </div>
    <div class="plan" data-f="plan"></div>
    <div class="small" data-f="act" style="display:none"></div>
    <div class="small" data-f="err" style="display:none;color:var(--red)"></div>`;
  const refs = {};
  el.querySelectorAll("[data-f]").forEach(n=>{ refs[n.getAttribute("data-f")] = n; });
  refs.role = el.querySelector("[data-role]");
  refs.enabled = el.querySelector("[data-enabled]");
  // Mark the select as "being edited" the moment the operator touches it (the
  // pointerdown/focus fire BEFORE the native popup opens), and keep it marked
  // briefly after blur so the change->POST->refresh round-trip can land before
  // the per-tick reconciliation resumes.
  const hold = ()=> editingRoles.add(name);
  refs.role.addEventListener("pointerdown", hold);
  refs.role.addEventListener("focus", hold);
  refs.role.addEventListener("blur", ()=> setTimeout(()=> editingRoles.delete(name), 600));
  return {el, refs};
}
function renderDrones(s){
  const root = document.getElementById("drones");
  const cfgs = (s.match&&s.match.drones)||{}; const acts=s.actions||{};
  const auto = !!(s.match&&s.match.auto);
  const seen = new Set();
  for(const d of s.drones){
    seen.add(d.name);
    let card = droneCards[d.name];
    if(!card){ card = droneCards[d.name] = buildDroneCard(d.name); root.appendChild(card.el); }
    const el = card.el, r = card.refs;
    const cfg = cfgs[d.name]||{};
    el.className = "drone" + (d.connected? "":" off");
    // role select: reconcile to the server value ONLY when the operator is not
    // interacting with it AND it actually differs. The differs-check matters as
    // much as the editing guard: in steady state value===want, so we never write
    // and never disturb an open popup; in AUTO the coordinator owns roles so the
    // select is disabled (and a disabled select fires no focus, so reconciliation
    // still reflects coordinator changes).
    const wantRole = cfg.role || 'idle';
    if(!editingRoles.has(d.name) && r.role.value !== wantRole) r.role.value = wantRole;
    r.role.disabled = auto;
    r.role.title = auto? "AUTO assigns roles" : "";
    const wantOn = cfg.enabled !== false;
    if(document.activeElement !== r.enabled && r.enabled.checked !== wantOn) r.enabled.checked = wantOn;
    r.dot.className = "dot " + (d.connected?'ok':'bad');
    r.pos.textContent = d.position_m ? d.position_m.map(x=>x.toFixed(1)).join(", ") : "—";
    r.height.textContent = d.height_m==null?'—':d.height_m+' m';
    r.bat.textContent = d.battery_pct==null? "—" : Math.round(d.battery_pct)+"%";
    r.bat.className = "v bat " + (d.battery_low?'low':'');
    r.link.textContent = (d.drone_connected?'ok':'—') + (d.flying?' · flying':'');
    r.markers.textContent = (d.visible_marker_ids||[]).join(',')||'—';
    r.wifi.textContent = d.wifi_ssid||'—';
    r.plan.textContent = d.plan||'';
    const a = acts[d.name];
    if(a){ r.act.style.display=""; r.act.textContent = "↳ " + a; } else { r.act.style.display="none"; }
    if(d.last_error && !d.connected){ r.err.style.display=""; r.err.textContent = d.last_error; }
    else { r.err.style.display="none"; }
  }
  // drop cards for drones that vanished from the snapshot
  for(const name of Object.keys(droneCards)){
    if(!seen.has(name)){ droneCards[name].el.remove(); delete droneCards[name]; editingRoles.delete(name); }
  }
}
function renderBoxes(s){
  const root = document.getElementById("boxes"); root.innerHTML="";
  const m = s.match||{}; const boxes = m.boxes||{};
  for(let slot=1; slot<=6; slot++){
    const b = boxes[slot]; const el=document.createElement("div"); el.className="box";
    if(!b){ el.innerHTML = `<div class="id">${slot}</div><div class="hold unknown">?</div>`; root.appendChild(el); continue; }
    const homeTeam = slot<=3? "red":"blue";
    const tag = b.captured_by_us? `<div class="tag ours">OURS</div>`
              : b.captured_by_enemy? `<div class="tag enemy">ENEMY</div>`
              : `<div class="tag small">${homeTeam} home</div>`;
    const age = b.last_seen_age_s==null? "never" : b.last_seen_age_s.toFixed(0)+"s ago";
    el.innerHTML = `
      <div class="id">box ${slot} <span class="small">(${homeTeam} home)</span></div>
      <div class="hold ${b.holder}">${b.holder.toUpperCase()}</div>
      ${tag}
      ${b.locked? `<div class="lock">🔒 ${b.lock_remaining_s.toFixed(1)}s</div>`:''}
      <div class="small">seen ${age}${b.last_seen_by? ' · '+b.last_seen_by.replace('flightctrl','fc'):''}</div>`;
    root.appendChild(el);
  }
}
function renderMap(s){
  const a = s.arena_geom; const m = s.match||{}; const c = document.getElementById("map");
  const ctx = c.getContext("2d"); ctx.clearRect(0,0,c.width,c.height);
  if(!a || a.x_min==null){ return; }
  const pad = 16, W = c.width-2*pad, H = c.height-2*pad;
  const yspan = a.y_max-a.y_min, xspan = a.x_max-a.x_min;
  const sx = (y)=> pad + ((y-a.y_min)/yspan)*W;     // world y -> screen x
  const sy = (x)=> pad + ((a.x_max-x)/xspan)*H;     // world x -> screen y
  ctx.strokeStyle="#30363d"; ctx.strokeRect(pad,pad,W,H);
  const rh=a.red_home_y||[a.y_min,a.y_min+5], bh=a.blue_home_y||[a.y_max-5,a.y_max];
  ctx.fillStyle="rgba(248,81,73,.10)"; ctx.fillRect(sx(rh[0]),pad,sx(rh[1])-sx(rh[0]),H);
  ctx.fillStyle="rgba(56,139,253,.10)"; ctx.fillRect(sx(bh[0]),pad,sx(bh[1])-sx(bh[0]),H);
  const boxes=a.boxes||{}, bstate=(m.boxes)||{};
  ctx.font="10px sans-serif";
  for(const slot in boxes){
    const [bx,by]=boxes[slot]; const st=bstate[slot];
    const holder = st? st.holder : "unknown";
    ctx.fillStyle = HOLD_COL[holder]||HOLD_COL.unknown;
    ctx.fillRect(sx(by)-5, sy(bx)-5, 10, 10);
    ctx.strokeStyle="#0b0f14"; ctx.strokeRect(sx(by)-5, sy(bx)-5, 10, 10);
    ctx.fillStyle="#8b949e"; ctx.fillText(slot, sx(by)+7, sy(bx)+3);
  }
  for(const d of s.drones){
    if(!d.position_m) continue;
    const [x,y]=d.position_m; const X=sx(y), Y=sy(x);
    ctx.beginPath(); ctx.arc(X,Y,6,0,7);
    ctx.fillStyle = d.connected? "#2ea043":"#8b949e"; ctx.fill();
    ctx.fillStyle="#e6edf3"; ctx.fillText(d.name.replace("flightctrl","fc"), X+8, Y+3);
    if(d.drone_yaw_deg!=null){
      const r=12, th=d.drone_yaw_deg*Math.PI/180;
      ctx.beginPath(); ctx.moveTo(X,Y); ctx.lineTo(X+r*Math.sin(th), Y-r*Math.cos(th));
      ctx.strokeStyle="#2ea043"; ctx.stroke();
    }
  }
}
document.getElementById("team-sel").addEventListener("change", async (e)=>{
  await api("/api/team", {team: e.target.value}); tick();
});
document.getElementById("arena-sel").addEventListener("change", async (e)=>{
  await api("/api/arena", {name: e.target.value}); tick();
});
let armedNow = false, autoNow = false;
document.getElementById("arm-btn").addEventListener("click", async ()=>{
  if(!armedNow && !confirm("ARM the swarm? Enabled drones will launch their role and fly.")) return;
  await api("/api/arm", {armed: !armedNow}); tick();
});
document.getElementById("auto-btn").addEventListener("click", async ()=>{
  await api("/api/auto", {auto: !autoNow}); tick();
});
let landError = "";
async function emergencyLand(){
  // Fire immediately — land must always win, no confirm.
  landError = "";
  try {
    const r = await api("/api/emergency-land", {});
    const j = await r.json().catch(()=>({}));
    if(!r.ok || j.ok===false){ landError = "EMERGENCY LAND FAILED: " + (j.error||r.status) + " — USE THE PHYSICAL KILL SWITCH"; }
  } catch(e){
    landError = "EMERGENCY LAND request failed: " + e + " — USE THE PHYSICAL KILL SWITCH";
  }
  if(landError){ console.error(landError); alert(landError); }
  tick();
}
document.getElementById("land-btn").addEventListener("click", emergencyLand);
// '?' keyboard kill switch — lands ALL drones immediately, any focus.
window.addEventListener("keydown", (e)=>{ if(e.key==="?") emergencyLand(); });
// per-drone role + enabled
document.addEventListener("change", async (e)=>{
  const t=e.target;
  if(t.matches("[data-role]")){ await api(`/api/drone/${encodeURIComponent(t.getAttribute("data-role"))}/role`, {role:t.value}); tick(); }
  if(t.matches("[data-enabled]")){ await api(`/api/drone/${encodeURIComponent(t.getAttribute("data-enabled"))}/enabled`, {enabled:t.checked}); tick(); }
  if(t.matches("[data-tune]")){ await api("/api/tune", {key:t.getAttribute("data-tune"), value:parseFloat(t.value)}); tick(); }
});
tick(); setInterval(tick, 1000);
</script>
</body></html>
"""
