/* SDC26 C2 operator console — vanilla JS, no build step, no deps.
 *
 * Renders the live WorldSnapshot (target slots, drones, events, mode, score)
 * and forwards operator actions to the C2 API. Live updates arrive over
 * Server-Sent Events with an automatic fall-back to 1 Hz polling.
 */
"use strict";

(function () {
  // ---- DOM handles ------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const el = {
    teamBadge: $("team-badge"),
    connBadge: $("conn-badge"),
    scoreUs: $("score-us"),
    scoreThem: $("score-them"),
    modeManual: $("mode-manual"),
    modeAuto: $("mode-auto"),
    slotsGrid: $("slots-grid"),
    slotsSub: $("slots-sub"),
    droneCount: $("drone-count"),
    dronesBody: $("drones-body"),
    eventLog: $("event-log"),
    eventsSub: $("events-sub"),
    droneSelect: $("drone-select"),
    attackSlot: $("attack-slot"),
    defendSlot: $("defend-slot"),
    roleSelect: $("role-select"),
    btnAttack: $("btn-attack"),
    btnDefend: $("btn-defend"),
    btnRole: $("btn-role"),
    toast: $("toast"),
  };

  // ---- State ------------------------------------------------------------
  let last = null;             // last snapshot
  let ourTeam = null;
  let selectedDrone = null;    // sticky drone selection across refreshes
  const N_SLOTS = 6;           // contract: always render 6 slots

  // ======================================================================
  //  Rendering
  // ======================================================================
  function fmtAge(age) {
    if (age == null) return "—";
    if (age < 10) return age.toFixed(1) + "s";
    if (age < 100) return age.toFixed(0) + "s";
    return Math.round(age) + "s";
  }
  function fmtXY(pos) {
    if (!pos) return "xy —";
    return "xy " + pos[0].toFixed(1) + "," + pos[1].toFixed(1);
  }

  function renderHeader(s) {
    ourTeam = s.our_team;
    el.teamBadge.textContent = "team " + s.our_team;
    el.teamBadge.className = "badge team-" + s.our_team;
    el.scoreUs.textContent = s.score_us;
    el.scoreThem.textContent = s.score_them;
    const manual = s.mode === "manual";
    el.modeManual.classList.toggle("active", manual);
    el.modeAuto.classList.toggle("active", !manual);
  }

  function renderSlots(s) {
    const slots = s.slots || {};
    const frag = document.createDocumentFragment();
    // ALWAYS render all 6 slots, even ones never observed.
    for (let i = 1; i <= N_SLOTS; i++) {
      const slot = slots[String(i)] || null;
      const color = slot ? slot.color : "unknown"; // red|blue|unknown
      const homeTeam = slot ? slot.home_team : (i <= 3 ? "blue" : "red");
      const mine = ourTeam && homeTeam === ourTeam;

      const tile = document.createElement("div");
      tile.className = "slot-tile owner-" + color;

      // freshness
      let freshCls = "never", freshTxt = "never";
      if (slot && slot.marker_id_seen != null) {
        if (slot.fresh) { freshCls = "fresh"; freshTxt = "fresh"; }
        else { freshCls = "stale"; freshTxt = "stale"; }
      }
      const age = slot ? slot.age_s : null;
      const marker = slot && slot.marker_id_seen != null ? slot.marker_id_seen : "—";

      tile.innerHTML =
        '<div class="slot-top">' +
          '<span class="slot-idx">' + i + '</span>' +
          '<span class="slot-home' + (mine ? " mine" : "") + '">' +
            (mine ? "ours · " : "enemy · ") + homeTeam +
          '</span>' +
        '</div>' +
        '<div class="owner">' + color + '</div>' +
        '<div><span class="fresh-pill ' + freshCls + '">' + freshTxt + '</span> ' +
          '<span class="muted">' + fmtAge(age) + '</span></div>' +
        '<div class="slot-meta">' +
          '<div>marker ' + marker + '</div>' +
          '<div>' + fmtXY(slot ? slot.position : null) + '</div>' +
        '</div>';
      frag.appendChild(tile);
    }
    el.slotsGrid.replaceChildren(frag);
  }

  function battClass(p) {
    if (p == null) return "";
    if (p <= 20) return "low";
    if (p <= 40) return "mid";
    return "";
  }

  function renderDrones(s) {
    const drones = s.drones || {};
    const ids = Object.keys(drones).sort();
    el.droneCount.textContent = ids.length;

    if (!ids.length) {
      el.dronesBody.innerHTML =
        '<tr class="empty"><td colspan="8">no drones configured</td></tr>';
      return;
    }
    const frag = document.createDocumentFragment();
    for (const id of ids) {
      const d = drones[id];
      const z = d.position ? d.position[2] : 0;
      const alt = (d.base_altitude_m + (d.position ? z : 0));
      const tr = document.createElement("tr");
      tr.dataset.drone = id;
      if (id === selectedDrone) tr.className = "selected";

      const linkDot = d.connected
        ? (d.flying ? '<span class="dot on"></span>flying'
                    : '<span class="dot warn"></span>idle')
        : '<span class="dot off"></span>down';
      const batt = d.battery_pct == null ? "—"
        : '<span class="batt ' + battClass(d.battery_pct) + '">' +
          Math.round(d.battery_pct) + '%</span>';

      tr.innerHTML =
        '<td class="mono">' + id + '</td>' +
        '<td><span class="pill role-' + d.role + '">' + d.role + '</span></td>' +
        '<td>' + d.phase + (d.pos_stale ? ' <span class="muted">·stale</span>' : '') + '</td>' +
        '<td class="mono">' + alt.toFixed(1) + '</td>' +
        '<td>' + linkDot + '</td>' +
        '<td>' + batt + '</td>' +
        '<td class="mono">' + (d.assigned_slot != null ? d.assigned_slot : '—') + '</td>' +
        '<td class="muted">' + (d.note || '') + '</td>';
      tr.addEventListener("click", () => {
        selectedDrone = id;
        el.droneSelect.value = id;
        syncSelectedRow();
      });
      frag.appendChild(tr);
    }
    el.dronesBody.replaceChildren(frag);
  }

  function syncSelectedRow() {
    for (const tr of el.dronesBody.querySelectorAll("tr")) {
      tr.classList.toggle("selected", tr.dataset.drone === selectedDrone);
    }
  }

  function renderDroneSelect(s) {
    const ids = Object.keys(s.drones || {}).sort();
    const prev = el.droneSelect.value || selectedDrone;
    const frag = document.createDocumentFragment();
    for (const id of ids) {
      const o = document.createElement("option");
      o.value = id; o.textContent = id;
      frag.appendChild(o);
    }
    el.droneSelect.replaceChildren(frag);
    if (ids.length) {
      if (prev && ids.includes(prev)) el.droneSelect.value = prev;
      selectedDrone = el.droneSelect.value;
    } else {
      selectedDrone = null;
    }
  }

  function renderSlotSelects() {
    // Attack: enemy slots first but allow any 1..6. Defend: any of ours.
    function fill(sel, keepEmpty) {
      const prev = sel.value;
      const frag = document.createDocumentFragment();
      if (keepEmpty) {
        const o = document.createElement("option");
        o.value = ""; o.textContent = "— any —";
        frag.appendChild(o);
      }
      for (let i = 1; i <= N_SLOTS; i++) {
        const o = document.createElement("option");
        o.value = String(i);
        const home = i <= 3 ? "blue" : "red";
        o.textContent = "slot " + i + " (" + home + ")";
        frag.appendChild(o);
      }
      sel.replaceChildren(frag);
      if (prev) sel.value = prev;
    }
    // Only build once (options are static); guard by child count.
    if (el.attackSlot.children.length !== N_SLOTS) fill(el.attackSlot, false);
    if (el.defendSlot.children.length !== N_SLOTS + 1) fill(el.defendSlot, true);
  }

  function renderEvents(s) {
    const events = s.events || [];
    el.eventsSub.textContent = events.length;
    if (!events.length) {
      el.eventLog.innerHTML = '<li class="muted">no events yet</li>';
      return;
    }
    const frag = document.createDocumentFragment();
    // newest last in the source -> show newest first.
    for (let i = events.length - 1; i >= 0; i--) {
      const li = document.createElement("li");
      li.textContent = events[i];
      frag.appendChild(li);
    }
    el.eventLog.replaceChildren(frag);
  }

  function render(s) {
    last = s;
    renderHeader(s);
    renderSlots(s);
    renderDrones(s);
    renderDroneSelect(s);
    renderSlotSelects();
    renderEvents(s);
  }

  // ======================================================================
  //  Live link: SSE with polling fallback
  // ======================================================================
  let es = null;
  let pollTimer = null;

  function setConn(state) {
    el.connBadge.textContent =
      state === "live" ? "live" : state === "polling" ? "polling" : "offline";
    el.connBadge.className = "badge " + state;
  }

  function startPolling() {
    if (pollTimer) return;
    setConn("polling");
    const tick = async () => {
      try {
        const r = await fetch("/api/state", { cache: "no-store" });
        if (r.ok) { render(await r.json()); }
        else { setConn("offline"); }
      } catch (_) { setConn("offline"); }
    };
    tick();
    pollTimer = setInterval(tick, 1000);
  }
  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  function connectSSE() {
    if (!window.EventSource) { startPolling(); return; }
    try {
      es = new EventSource("/api/state/stream");
    } catch (_) {
      startPolling();
      return;
    }
    es.onopen = () => { stopPolling(); setConn("live"); };
    es.onmessage = (ev) => {
      if (!ev.data) return;          // heartbeat comment, ignore
      try { render(JSON.parse(ev.data)); setConn("live"); }
      catch (_) { /* ignore malformed frame */ }
    };
    es.onerror = () => {
      // Browser auto-reconnects EventSource; meanwhile fall back to polling
      // so the operator never goes blind.
      setConn("offline");
      if (es && es.readyState === EventSource.CLOSED) {
        es = null;
        startPolling();
        // Retry SSE after a short delay.
        setTimeout(connectSSE, 3000);
      } else {
        startPolling();
      }
    };
  }

  // ======================================================================
  //  Commands
  // ======================================================================
  let toastTimer = null;
  function toast(msg, ok) {
    el.toast.textContent = msg;
    el.toast.className = "toast show " + (ok ? "ok" : "err");
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { el.toast.className = "toast"; }, 4000);
  }

  async function postCommand(cmd) {
    try {
      const r = await fetch("/api/command", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cmd),
      });
      const body = await r.json().catch(() => ({}));
      const ok = r.ok && body.ok !== false;
      const label = cmd.type + (cmd.drone_id ? " " + cmd.drone_id : "") +
        (cmd.slot != null ? " slot " + cmd.slot : "") +
        (cmd.role ? " " + cmd.role : "");
      toast((ok ? "OK  " : "ERR  ") + label +
        (body.error ? " — " + body.error : (body.msg ? " — " + body.msg : "")), ok);
    } catch (e) {
      toast("ERR  network: " + e, false);
    }
  }

  async function postMode(mode) {
    try {
      const r = await fetch("/api/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode }),
      });
      const body = await r.json().catch(() => ({}));
      const ok = r.ok && body.ok !== false;
      toast((ok ? "OK  mode " : "ERR  mode ") + mode +
        (body.error ? " — " + body.error : ""), ok);
    } catch (e) {
      toast("ERR  network: " + e, false);
    }
  }

  function needDrone() {
    const id = el.droneSelect.value;
    if (!id) { toast("ERR  no drone selected", false); return null; }
    return id;
  }

  function wireConsole() {
    // Mode toggle.
    el.modeManual.addEventListener("click", () => postMode("manual"));
    el.modeAuto.addEventListener("click", () => postMode("auto"));

    // Keep sticky selection in sync with the dropdown.
    el.droneSelect.addEventListener("change", () => {
      selectedDrone = el.droneSelect.value;
      syncSelectedRow();
    });

    // Per-drone + global simple commands (data-type buttons).
    document.querySelectorAll("button.cmd[data-type]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const type = btn.dataset.type;
        if (btn.dataset.all) {
          if (type === "emergency" &&
              !confirm("EMERGENCY: cut motors on ALL drones?")) return;
          postCommand({ type, drone_id: "all" });
        } else {
          const id = needDrone();
          if (!id) return;
          postCommand({ type, drone_id: id });
        }
      });
    });

    // Attack.
    el.btnAttack.addEventListener("click", () => {
      const id = needDrone();
      if (!id) return;
      const slot = parseInt(el.attackSlot.value, 10);
      postCommand({ type: "attack", drone_id: id, slot });
    });

    // Defend (optional slot).
    el.btnDefend.addEventListener("click", () => {
      const id = needDrone();
      if (!id) return;
      const cmd = { type: "defend", drone_id: id };
      if (el.defendSlot.value) cmd.slot = parseInt(el.defendSlot.value, 10);
      postCommand(cmd);
    });

    // Assign role.
    el.btnRole.addEventListener("click", () => {
      const id = needDrone();
      if (!id) return;
      postCommand({ type: "assign_role", drone_id: id, role: el.roleSelect.value });
    });
  }

  // ======================================================================
  //  Boot
  // ======================================================================
  function boot() {
    wireConsole();
    // Render an immediate skeleton (6 empty slots) before any data lands.
    render({
      t: 0, mode: "manual", our_team: "red",
      drones: {}, slots: {}, score_us: 0, score_them: 0, events: [],
    });
    setConn("offline");
    connectSSE();
    // Safety net: if SSE never opened within 2 s, start polling.
    setTimeout(() => {
      if (!es || es.readyState !== EventSource.OPEN) startPolling();
    }, 2000);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
