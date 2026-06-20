
  (function(){
    let misDronesKnown = {};

    async function misLoadDrones() {
      try {
        const r = await fetch('/proxy/drones');
        const d = await r.json();
        misDronesKnown = d.drones || {};
        const cont = document.getElementById('mis_drones');
        const prev = {};
        cont.querySelectorAll('input[type=checkbox]').forEach(cb => { prev[cb.dataset.id] = cb.checked; });
        cont.innerHTML = '';
        const ids = Object.keys(misDronesKnown).sort();
        ids.forEach(id => {
          const info = misDronesKnown[id];
          const wrap = document.createElement('label');
          wrap.style.cssText = 'display:flex;gap:4px;align-items:center;font-size:12px;color:#e2e8f0;cursor:pointer;';
          const checked = (id in prev) ? prev[id] : true;
          wrap.innerHTML = '<input type="checkbox" data-id="'+id+'"'+(checked?' checked':'')+' /> '+info.name+' <span style="color:#64748b;">#'+id+'</span>';
          cont.appendChild(wrap);
        });
      } catch {}
    }

    function misSelectedDroneIds() {
      return Array.from(document.querySelectorAll('#mis_drones input[type=checkbox]'))
        .filter(cb => cb.checked).map(cb => cb.dataset.id);
    }

    function misBadge(phase) {
      const low = (phase || 'idle').toLowerCase();
      let cls = 'idle';
      if (low === 'search') cls = 'scan';
      else if (low === 'approach') cls = 'approach';
      else if (low === 'hover') cls = 'hover';
      else if (low === 'done') cls = 'done';
      else if (low === 'error') cls = 'error';
      return '<span class="mis-badge '+cls+'">'+phase+'</span>';
    }

    function misRenderStatus(st) {
      const title = document.getElementById('mis_title_status');
      const prog = document.getElementById('mis_progress');
      const panel = document.getElementById('mis_status');

      if (!st.has_mission) {
        title.textContent = 'idle';
        title.style.color = '#94a3b8';
        prog.textContent = '—';
        document.getElementById('mis_scanned').textContent = '—';
        document.getElementById('mis_remaining').textContent = '—';
        panel.textContent = 'idle — no mission running';
        return;
      }
      title.textContent = st.active ? '● running' : '○ stopped';
      title.style.color = st.active ? '#4ade80' : '#94a3b8';
      prog.textContent = 'Progress: ' + st.progress;
      document.getElementById('mis_scanned').textContent = (st.scanned || []).join(', ') || '—';
      document.getElementById('mis_remaining').textContent = (st.remaining || []).join(', ') || '—';

      let html = '';
      const drones = st.drones || {};
      Object.keys(drones).sort().forEach(did => {
        const d = drones[did];
        const name = (misDronesKnown[did] && misDronesKnown[did].name) || ('Drone '+did);
        const tgt = d.target != null ? 'target '+d.target : '—';
        html += '<div class="mis-drone-line"><b>'+name+' #'+did+'</b> '+misBadge(d.phase)+
                ' <span style="color:#64748b;">['+tgt+']</span> <span style="color:#cbd5e1;">'+(d.note||'')+'</span></div>';
      });
      if (st.error) html += '<div style="color:#fca5a5;margin-top:6px;">error: '+st.error+'</div>';
      panel.innerHTML = html || 'no drones assigned';
    }

    async function misPoll() {
      try {
        const r = await fetch('/proxy/missions/status');
        const st = await r.json();
        misRenderStatus(st);
        // If a capture-targets mission is running, pull its boxes into the
        // global so the arena views (2D + 3D) can render them.
        if (st && st.has_mission && st.target_boxes &&
            Array.isArray(st.target_boxes) && st.target_boxes.length) {
          window._targetBoxes = st.target_boxes;
        }
        // Capture state → colour the boxes in the arena view
        window._missionClaimedBoxes = (st && st.claimed) || {};
        window._missionCapturedBoxes = (st && st.captured) || [];
      } catch {}
    }

    // Parse the mission's target-boxes JSON textarea live and expose it
    // globally. Both the 2D canvas drawArena() and the Three.js scene
    // read window._targetBoxes to render the boxes on the floor even
    // before the mission starts.
    function _parseTargetBoxesInput() {
      const el = document.getElementById('mis_boxes_json');
      if (!el) return;
      try {
        const v = JSON.parse(el.value);
        if (Array.isArray(v)) {
          window._targetBoxes = v;
          el.style.borderColor = '#334155';
        } else {
          el.style.borderColor = '#ef4444';
        }
      } catch {
        el.style.borderColor = '#ef4444';
      }
    }
    (function(){
      const el = document.getElementById('mis_boxes_json');
      if (el) {
        el.addEventListener('input', _parseTargetBoxesInput);
        _parseTargetBoxesInput();  // initial parse so defaults render
      }
    })();

    // Click-to-arm pattern for Start Mission — no native confirm() dialog
    // (some browsers auto-dismiss rapid-fire dialogs, making the mission
    // never start with no visible feedback).
    let _misArmedUntil = 0;
    document.getElementById('mis_start').onclick = async () => {
      const drone_ids = misSelectedDroneIds();
      const misErr = document.getElementById('mis_err');
      const misOk  = document.getElementById('mis_ok');
      function misShowWarn(msg) {
        if (!misErr) return;
        misErr.textContent = msg;
        misErr.style.background = '#78350f';
        misErr.style.color = '#fde68a';
        misErr.style.borderColor = '#f59e0b';
        misErr.style.display = 'inline-block';
      }
      function misClearMsgs() {
        if (misErr) misErr.style.display = 'none';
        if (misOk)  misOk.style.display  = 'none';
      }
      if (!drone_ids.length) {
        misShowWarn('✗ Select at least one drone first');
        return;
      }
      const missionType = document.getElementById('mis_type').value;
      const auto_takeoff = document.getElementById('mis_auto_takeoff').checked;
      let endpoint, payload;
      if (missionType === 'capture_targets') {
        // Parse target-boxes JSON
        let boxes = [];
        try { boxes = JSON.parse(document.getElementById('mis_boxes_json').value); }
        catch (e) { misShowWarn('✗ target_boxes JSON invalid: ' + e.message); return; }
        if (!Array.isArray(boxes) || boxes.length === 0) {
          misShowWarn('✗ target_boxes must be a non-empty JSON array'); return;
        }
        const home_xy = [
          parseFloat(document.getElementById('mis_home_x').value) || 0,
          parseFloat(document.getElementById('mis_home_y').value) || 0,
        ];
        const face_xy = [
          parseFloat(document.getElementById('mis_face_x').value) || 0,
          parseFloat(document.getElementById('mis_face_y').value) || 0,
        ];
        const alt = parseFloat(document.getElementById('mis_alt').value) || 1.5;
        const hv  = parseFloat(document.getElementById('mis_cap_hover_s').value) || 4.0;
        endpoint = '/proxy/missions/capture_targets/start';
        payload = {
          drone_ids, target_boxes: boxes,
          home_xy, arena_face_xy: face_xy,
          hover_above_m: alt, hover_seconds: hv,
          auto_takeoff,
        };
      } else {
        // scan_all (default)
        const markers = document.getElementById('mis_markers').value;
        const hover_seconds = parseFloat(document.getElementById('mis_hover_s').value) || 3.0;
        const tol = parseFloat(document.getElementById('mis_tol_m').value) || 0.35;
        const skew_tol_el = document.getElementById('mis_skew_tol');
        const skew_tol = skew_tol_el ? (parseFloat(skew_tol_el.value) || 0.08) : 0.08;
        endpoint = '/proxy/missions/scan_all/start';
        payload = {
          drone_ids, target_markers: markers,
          hover_seconds, approach_tolerance_m: tol,
          approach_skew_tol: skew_tol,
          auto_takeoff,
        };
      }
      const btn = document.getElementById('mis_start');
      const now = Date.now();
      if (now >= _misArmedUntil) {
        // First click — arm for 3s, show summary inline
        _misArmedUntil = now + 3000;
        const origLabel = btn.textContent;
        btn._origLabel = origLabel;
        btn.textContent = '⚠ Click again to launch';
        btn.style.animation = 'arcpulse 0.6s ease-in-out infinite';
        const summary = 'Drones: ' + drone_ids.join(', ') +
                        '   •   Markers: ' + markers +
                        '   •   Hover ' + hover_seconds + 's' +
                        (auto_takeoff ? '   •   ⚠ AUTO-TAKEOFF' : '');
        misShowWarn('⚠ Starting mission — ' + summary + '. Click Start again within 3 s.');
        setTimeout(() => {
          if (Date.now() >= _misArmedUntil) {
            _misArmedUntil = 0;
            btn.textContent = btn._origLabel || '▶ Start mission';
            btn.style.animation = '';
            misClearMsgs();
          }
        }, 3100);
        return;
      }
      // Armed — proceed with launch
      _misArmedUntil = 0;
      btn.style.animation = '';
      btn.textContent = btn._origLabel || '▶ Start mission';
      misClearMsgs();
      const origLabel = btn.textContent;
      btn.disabled = true; btn.textContent = '… starting';
      let j = {}; let httpStatus = 0;
      try {
        const r = await fetch(endpoint, {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify(payload)});
        httpStatus = r.status;
        try { j = await r.json(); } catch { j = {}; }
      } catch (err) {
        j = { ok: false, error: 'network error: ' + err };
      } finally {
        btn.disabled = false; btn.textContent = origLabel;
      }
      console.log('[mission] start response:', httpStatus, j);
      if (!j.ok) {
        const msg = (j.error || j.message || 'unknown error') + (httpStatus ? ' (HTTP ' + httpStatus + ')' : '');
        if (misErr) {
          misErr.textContent = '✗ Mission start refused: ' + msg;
          misErr.style.display = 'inline-block';
        }
      } else {
        const msg = j.message || 'mission started';
        const warn = j.status && j.status.error ? ' — warning: ' + j.status.error : '';
        console.log('[mission] started:', msg, warn);
        if (misOk) {
          misOk.textContent = '✓ ' + msg + warn;
          misOk.style.display = 'inline-block';
        }
      }
      misPoll();
    };

    document.getElementById('mis_stop').onclick = async () => {
      await fetch('/proxy/missions/stop', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({land:false})});
      misPoll();
    };

    document.getElementById('mis_stop_land').onclick = async () => {
      if (!confirm('Stop mission AND land all participating drones?')) return;
      await fetch('/proxy/missions/stop', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({land:true})});
      misPoll();
    };

    misLoadDrones();
    // If the drone config changes, the drones list in the ArUco section
    // repopulates via /proxy/drones poll — mirror that for missions.
    setInterval(misLoadDrones, 5000);
    setInterval(misPoll, 1500);  // was 500ms → 1.5s; mission status barely changes between ticks
    misPoll();

    // Show/hide the capture-targets-specific rows based on mission type
    const misTypeSel = document.getElementById('mis_type');
    const misCaptureRows = document.getElementById('mis_capture_rows');
    const misScanRows = document.querySelector('#missions_panel .mis-row:nth-of-type(3)');
    function syncMissionUI() {
      const t = misTypeSel.value;
      if (misCaptureRows) misCaptureRows.style.display = (t === 'capture_targets') ? '' : 'none';
      if (misScanRows)    misScanRows.style.display    = (t === 'capture_targets') ? 'none' : '';
    }
    if (misTypeSel) { misTypeSel.addEventListener('change', syncMissionUI); syncMissionUI(); }

    // ── Mission-as-code + preset management ─────────────────────────
    // Each mission type has its own JSON block. The editor is the
    // canonical source of truth: "Run from code" parses it and starts
    // directly. "From form" rebuilds the JSON from the form fields so
    // the editor stays in sync with traditional click-around editing.
    // Presets are persisted server-side in mission_presets.json.
    (function wireMissionCode(){
      const code  = document.getElementById('mis_code');
      const sel   = document.getElementById('mis_preset_sel');
      const nameI = document.getElementById('mis_preset_name');
      const psBtn = document.getElementById('mis_preset_save');
      const pdBtn = document.getElementById('mis_preset_delete');
      const plBtn = document.getElementById('mis_preset_load');
      const ffBtn = document.getElementById('mis_code_from_form');
      const tfBtn = document.getElementById('mis_code_to_form');
      const runBtn = document.getElementById('mis_code_run');
      const psStatus = document.getElementById('mis_preset_status');
      const cStatus  = document.getElementById('mis_code_status');
      if (!code) return;
      let _presets = {};

      function flash(el, msg, col) {
        if (!el) return;
        el.textContent = msg;
        el.style.color = col || '#64748b';
        setTimeout(() => { if (el.textContent === msg) el.textContent = ''; }, 3000);
      }
      function currentType() {
        return (misTypeSel && misTypeSel.value) || 'scan_all';
      }
      function renderPresetList() {
        if (!sel) return;
        const t = currentType();
        const names = Object.keys((_presets[t]) || {}).sort();
        sel.innerHTML = '';
        if (!names.length) {
          const opt = document.createElement('option');
          opt.value = ''; opt.textContent = '(no presets)';
          sel.appendChild(opt);
          return;
        }
        names.forEach(n => {
          const opt = document.createElement('option');
          opt.value = n; opt.textContent = n;
          sel.appendChild(opt);
        });
        if (names.includes('default')) sel.value = 'default';
      }
      async function loadPresets() {
        try {
          const j = await (await fetch('/proxy/missions/presets')).json();
          _presets = j.presets || {};
          renderPresetList();
          // Auto-populate the code editor with the default preset on first load
          if (!code.value.trim()) loadIntoEditor('default');
        } catch (e) { console.warn('[mis] preset load failed', e); }
      }
      function loadIntoEditor(name) {
        const t = currentType();
        const params = (_presets[t] || {})[name];
        if (!params) return;
        code.value = JSON.stringify(params, null, 2);
        if (nameI) nameI.value = name;
        flash(psStatus, '✓ loaded "' + name + '"', '#22c55e');
      }
      function buildFromForm() {
        // Reuse the same logic as mis_start but return the payload
        // instead of POSTing. Everything below mirrors the existing
        // build in mis_start.
        const drone_ids = misSelectedDroneIds();
        const auto_takeoff = document.getElementById('mis_auto_takeoff').checked;
        if (currentType() === 'capture_targets') {
          let boxes = [];
          try { boxes = JSON.parse(document.getElementById('mis_boxes_json').value); }
          catch {}
          return {
            drone_ids, target_boxes: boxes,
            home_xy: [
              parseFloat(document.getElementById('mis_home_x').value) || 0,
              parseFloat(document.getElementById('mis_home_y').value) || 0,
            ],
            arena_face_xy: [
              parseFloat(document.getElementById('mis_face_x').value) || 0,
              parseFloat(document.getElementById('mis_face_y').value) || 0,
            ],
            hover_above_m: parseFloat(document.getElementById('mis_alt').value) || 1.5,
            hover_seconds: parseFloat(document.getElementById('mis_cap_hover_s').value) || 4.0,
            auto_takeoff,
          };
        }
        return {
          drone_ids,
          target_markers: document.getElementById('mis_markers').value,
          hover_seconds: parseFloat(document.getElementById('mis_hover_s').value) || 3.0,
          approach_tolerance_m: parseFloat(document.getElementById('mis_tol_m').value) || 0.30,
          approach_skew_tol: parseFloat(document.getElementById('mis_skew_tol').value) || 0.12,
          auto_takeoff,
        };
      }
      function applyCodeToForm() {
        try {
          const p = JSON.parse(code.value);
          if (currentType() === 'capture_targets') {
            if (Array.isArray(p.target_boxes))
              document.getElementById('mis_boxes_json').value = JSON.stringify(p.target_boxes, null, 2);
            if (Array.isArray(p.home_xy)) {
              document.getElementById('mis_home_x').value = p.home_xy[0];
              document.getElementById('mis_home_y').value = p.home_xy[1];
            }
            if (Array.isArray(p.arena_face_xy)) {
              document.getElementById('mis_face_x').value = p.arena_face_xy[0];
              document.getElementById('mis_face_y').value = p.arena_face_xy[1];
            }
            if (p.hover_above_m != null) document.getElementById('mis_alt').value = p.hover_above_m;
            if (p.hover_seconds != null) document.getElementById('mis_cap_hover_s').value = p.hover_seconds;
          } else {
            if (p.target_markers != null) document.getElementById('mis_markers').value = p.target_markers;
            if (p.hover_seconds != null) document.getElementById('mis_hover_s').value = p.hover_seconds;
            if (p.approach_tolerance_m != null) document.getElementById('mis_tol_m').value = p.approach_tolerance_m;
            if (p.approach_skew_tol != null) document.getElementById('mis_skew_tol').value = p.approach_skew_tol;
          }
          if (typeof p.auto_takeoff === 'boolean')
            document.getElementById('mis_auto_takeoff').checked = p.auto_takeoff;
          flash(cStatus, '✓ form populated from JSON', '#22c55e');
        } catch (e) { flash(cStatus, '✗ JSON parse error: ' + e.message, '#ef4444'); }
      }
      // Wire buttons
      if (ffBtn) ffBtn.onclick = () => {
        code.value = JSON.stringify(buildFromForm(), null, 2);
        flash(cStatus, '✓ JSON rebuilt from form', '#22c55e');
      };
      if (tfBtn) tfBtn.onclick = applyCodeToForm;
      if (plBtn) plBtn.onclick = () => { if (sel && sel.value) loadIntoEditor(sel.value); };
      if (psBtn) psBtn.onclick = async () => {
        const name = (nameI && nameI.value.trim()) || (sel && sel.value) || '';
        if (!name) { flash(psStatus, '✗ enter a preset name', '#ef4444'); return; }
        let params;
        try { params = JSON.parse(code.value); }
        catch (e) { flash(psStatus, '✗ invalid JSON: ' + e.message, '#ef4444'); return; }
        try {
          const r = await fetch('/proxy/missions/presets', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify({
              mission_type: currentType(), name, params,
            }),
          });
          const j = await r.json();
          if (j.ok) { flash(psStatus, '✓ saved "' + name + '"', '#22c55e'); await loadPresets(); }
          else flash(psStatus, '✗ ' + (j.error || 'save failed'), '#ef4444');
        } catch (e) { flash(psStatus, '✗ ' + e, '#ef4444'); }
      };
      if (pdBtn) pdBtn.onclick = async () => {
        const name = (sel && sel.value) || '';
        if (!name) return;
        if (!confirm('Delete preset "' + name + '" for ' + currentType() + '?')) return;
        try {
          const u = '/proxy/missions/presets?mission_type=' + encodeURIComponent(currentType())
                    + '&name=' + encodeURIComponent(name);
          const r = await fetch(u, {method:'DELETE'});
          const j = await r.json();
          if (j.ok) { flash(psStatus, '✓ deleted "' + name + '"', '#22c55e'); await loadPresets(); }
          else flash(psStatus, '✗ ' + (j.error || 'delete failed'), '#ef4444');
        } catch (e) { flash(psStatus, '✗ ' + e, '#ef4444'); }
      };
      if (runBtn) runBtn.onclick = async () => {
        let payload;
        try { payload = JSON.parse(code.value); }
        catch (e) { flash(cStatus, '✗ JSON parse error: ' + e.message, '#ef4444'); return; }
        // Fill drone_ids from selector if JSON didn't provide any
        if (!Array.isArray(payload.drone_ids) || !payload.drone_ids.length) {
          payload.drone_ids = misSelectedDroneIds();
        }
        if (!payload.drone_ids || !payload.drone_ids.length) {
          flash(cStatus, '✗ drone_ids missing — select drone(s) or include in JSON', '#ef4444');
          return;
        }
        const endpoint = (currentType() === 'capture_targets')
          ? '/proxy/missions/capture_targets/start'
          : '/proxy/missions/scan_all/start';
        try {
          const r = await fetch(endpoint, {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(payload),
          });
          const j = await r.json();
          if (r.ok && j.ok) flash(cStatus, '✓ mission started', '#22c55e');
          else flash(cStatus, '✗ ' + (j.error || j.message || ('HTTP ' + r.status)), '#ef4444');
          misPoll();
        } catch (e) { flash(cStatus, '✗ ' + e, '#ef4444'); }
      };
      // Re-render preset list when the user flips mission type
      if (misTypeSel) misTypeSel.addEventListener('change', () => {
        renderPresetList();
        // Also auto-load the default preset of the new type so the
        // editor shows something relevant instead of stale JSON.
        const def = (_presets[currentType()] || {}).default;
        if (def) code.value = JSON.stringify(def, null, 2);
      });
      loadPresets();
    })();

    // Mission panel click counter — same idea as the ArUco one. Proves the
    // Start / Stop buttons receive the click event, independent of what
    // the server does with the request afterwards.
    const misHdr = document.querySelector('#missions_panel h3');
    if (misHdr) {
      const misClickTag = document.createElement('span');
      misClickTag.id = 'mis_click_counter';
      misClickTag.style.cssText = 'font-size:10px;color:#fbbf24;margin-left:8px;font-weight:400;';
      misClickTag.textContent = 'clicks: 0';
      misHdr.appendChild(misClickTag);
      let _misClicks = 0;
      ['mis_start','mis_stop','mis_stop_land'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('click', () => {
          _misClicks += 1;
          misClickTag.textContent = 'clicks: ' + _misClicks + ' (' + id + ')';
          console.log('[mission] click #' + _misClicks + ' from ' + id);
        });
      });
    }
  })();
  