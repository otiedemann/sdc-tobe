
  (function(){
    const PGROUPS = ['Mission target','Camera filter / deadbands','P gains (camera)','D gains (IMU damping)','Output clamps','Drawing'];
    const SLIDERS = [
      ['hover_distance_m','Hover distance (m)',          0.5, 4.0, 0.05, 0],
      ['fb_max',          'Approach speed (fwd RC %)',   0,100, 1, 0],
      ['fb_back_max',     'Retreat speed (back RC %)',   0,100, 1, 0],
      ['dist_p',          'Approach aggressiveness (P · distance)', 0, 60, 0.5, 0],
      ['ema_alpha',       'EMA α (smoothing)',           0.05,0.95, 0.05, 1],
      ['deadband_x',      'Deadband err_x',              0.00,0.30, 0.01, 1],
      ['deadband_y',      'Deadband err_y',              0.00,0.30, 0.01, 1],
      ['deadband_skew',   'Deadband skew',               0.00,0.30, 0.01, 1],
      ['deadband_dist_m', 'Deadband distance (m)',       0.00,1.00, 0.05, 1],
      ['yaw_p',           'P · yaw     (per err_x)',     0, 50, 1, 2],
      ['skew_p',          'P · lateral (per skew)',      0, 50, 1, 2],
      ['alt_p',           'P · altitude (per err_y)',    0,100, 1, 2],
      ['d_yaw',           'D · yaw     (°/s)',           0,  2, 0.05, 3],
      ['d_lr',            'D · lateral (cm/s vgy)',      0,  2, 0.05, 3],
      ['d_ud',            'D · vertical(cm/s vgz)',      0,  2, 0.05, 3],
      ['d_fb',            'D · fwd/back(cm/s vgx)',      0,  2, 0.05, 3],
      ['yaw_max',         'Clamp · yaw max',             0, 80, 1, 4],
      ['lr_max',          'Clamp · lateral max',         0,100, 1, 4],
      ['ud_max',          'Clamp · vertical max',        0,100, 1, 4],
      ['rc_min',          'RC dead-floor',               0, 10, 1, 4],
      ['cam_hfov_deg',    'Cam HFOV (drawing only)',    30,110, 1, 5],
      ['marker_size_m',   'Marker physical size (m)',    0.05, 2.0, 0.01, 5],
    ];
    let arcParams = {};
    let arcAllowLive = false;
    let arcMode = 'observe';

    // Which drone id is the ArUco panel currently tracking? Mirrors the
    // main UI's active drone, picked up from /proxy/drones on each poll.
    let arcActiveId = null;

    async function arcLoadParams() {
      try {
        const r = await fetch('/proxy/aruco/params');
        arcParams = await r.json();
        arcRenderParams();
      } catch {}
    }
    function arcRenderParams() {
      const cont = document.getElementById('arc_params');
      cont.innerHTML = '';
      let curGroup = -1;
      SLIDERS.forEach(([k,label,mn,mx,st,grp]) => {
        if (grp !== curGroup) {
          curGroup = grp;
          const h = document.createElement('div');
          h.className = 'arc-pgroup';
          h.innerHTML = '<div class="arc-pgroup-label">' + PGROUPS[grp] + '</div>';
          cont.appendChild(h);
        }
        const v = arcParams[k] ?? 0;
        const r = document.createElement('div');
        r.className = 'arc-row';
        r.innerHTML =
          '<label title="'+k+'">'+label+' <span class="info-icon" data-info="'+k+'">i</span></label>' +
          '<input type="range" min="'+mn+'" max="'+mx+'" step="'+st+'" value="'+v+'" data-k="'+k+'" />' +
          '<input type="number" min="'+mn+'" max="'+mx+'" step="'+st+'" value="'+v+'" data-k="'+k+'" />';
        cont.appendChild(r);
      });
      cont.querySelectorAll('input').forEach(el => {
        el.addEventListener('input', () => {
          const k = el.dataset.k;
          const v = parseFloat(el.value);
          if (isNaN(v)) return;
          arcParams[k] = v;
          cont.querySelectorAll('input[data-k="'+k+'"]').forEach(s => { if (s !== el) s.value = v; });
        });
        el.addEventListener('change', () => {
          const k = el.dataset.k;
          const v = parseFloat(el.value);
          if (!isNaN(v)) fetch('/proxy/aruco/params', {method:'POST',
              headers:{'Content-Type':'application/json'},
              body: JSON.stringify({[k]: v})});
        });
      });
    }

    function fmt(x, n) { return (x === undefined || x === null || isNaN(x)) ? '—' : Number(x).toFixed(n); }

    function arcApplyModeUI(mode, allow) {
      arcMode = mode || 'observe';
      if (typeof allow === 'boolean') arcAllowLive = allow;
      const bObs  = document.getElementById('arc_mode_observe');
      const bLive = document.getElementById('arc_mode_live');
      bObs.classList.toggle('active', arcMode === 'observe');
      bLive.classList.toggle('active', arcMode === 'live');
      bLive.classList.toggle('live',   arcMode === 'live');
      // Keep the button clickable — the server is the source of truth for
      // allow_live. We just visually hint with the "gate" badge if the
      // server has REMOTE_NO_LIVE set.
      bLive.disabled = false;
      bLive.style.opacity = arcAllowLive ? '1' : '0.7';
      document.getElementById('arc_mode_gate').style.display = arcAllowLive ? 'none' : 'inline';
      const live = (arcMode === 'live');
      document.getElementById('arc_live_banner').classList.toggle('show', live);
      document.body.classList.toggle('arc-live-mode', live);
      document.getElementById('arc_manual').classList.toggle('show', live);
    }

    function arcRenderReadout(s) {
      if (!s.running) {
        document.getElementById('arc_readout').innerHTML = '<span style="color:#64748b;">stopped — press Start</span>';
        return;
      }
      const html =
        '<b>Marker</b><br>' +
        '<span class="k">ID:</span>'+(s.marker_id ?? '—')+
        '&nbsp;&nbsp;<span class="k">visible:</span>'+((s.visible_ids||[]).join(', ')||'—')+'<br>' +
        '<span class="k">distance:</span>'+fmt(s.distance_m,2)+' m '+
          '<span class="pd">(raw '+fmt(s.raw_distance_m,2)+', target '+fmt(arcParams.hover_distance_m,2)+')</span><br>' +
        '<span class="k">err_x:</span>'+fmt(s.err_x,3)+
          '&nbsp;&nbsp;<span class="k">err_y:</span>'+fmt(s.err_y,3)+
          '&nbsp;&nbsp;<span class="k">skew:</span>'+fmt(s.skew,3)+'<br>' +
        '<br><b>IMU</b>  '+
        '<span class="pd">vgx '+fmt(s.vx_cms,0)+', vgy '+fmt(s.vy_cms,0)+', vgz '+fmt(s.vz_cms,0)+' cm/s; '+
          'yaw '+fmt(s.yaw,1)+'° @ '+fmt(s.yaw_rate_dps,0)+'°/s; alt '+fmt(s.altitude_m,2)+' m</span><br>' +
        (s.mode === 'live'
          ? '<br><b class="arc-rc-sent">RC — SENT to drone</b>'
          : '<br><b>RC — would-send</b> <span class="pd">(observe — not sent)</span>') + '<br>' +
        '<span class="k">lr:</span>'+(s.rc_lr ?? '—')+' <span class="pd">(P='+fmt(s.rc_lr_p,1)+' D='+fmt(s.rc_lr_d,1)+')</span>' +
        '&nbsp; <span class="k">fb:</span>'+(s.rc_fb ?? '—')+' <span class="pd">(P='+fmt(s.rc_fb_p,1)+' D='+fmt(s.rc_fb_d,1)+')</span><br>' +
        '<span class="k">ud:</span>'+(s.rc_ud ?? '—')+' <span class="pd">(P='+fmt(s.rc_ud_p,1)+' D='+fmt(s.rc_ud_d,1)+')</span>' +
        '&nbsp; <span class="k">yaw:</span>'+(s.rc_yaw ?? '—')+' <span class="pd">(P='+fmt(s.rc_yaw_p,1)+' D='+fmt(s.rc_yaw_d,1)+')</span><br>' +
        (s.rc_sent_at ? '<span class="k">last sent:</span><span class="arc-rc-sent">'+((Date.now()/1000 - s.rc_sent_at).toFixed(1))+' s ago</span><br>' : '') +
        (s.rc_send_error ? '<span class="k">send err:</span><span style="color:#fca5a5;">'+s.rc_send_error+'</span><br>' : '') +
        // Arena safety guard banner — red when active, grey when idle.
        (s.guard && s.guard.active
          ? '<br><b style="color:#fca5a5;background:#7f1d1d;padding:2px 6px;border-radius:3px;">⛔ SAFETY GUARD</b> '
              + '<span class="pd">' + (s.guard.actions||[]).join(', ')
              + '  pos=('+ (s.guard.pos||[]).join(',') +')</span>'
          : '');
      document.getElementById('arc_readout').innerHTML = html;
    }

    function arcDrawTopDown(s) {
      const c = document.getElementById('arc_topdown');
      const ctx = c.getContext('2d');
      const W = c.width, H = c.height;
      ctx.fillStyle = '#0b1220'; ctx.fillRect(0,0,W,H);
      const cx = W/2;
      const marker_y = 36;
      const target = arcParams.hover_distance_m || 1.5;
      const maxDist = Math.max(target*2.2, 3.0);
      const ppm = (H-80)/maxDist;
      ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 1;
      for (let d=0.5; d<=maxDist; d+=0.5) { ctx.beginPath(); ctx.arc(cx, marker_y, d*ppm, 0, Math.PI, false); ctx.stroke(); }
      ctx.fillStyle = '#334155'; ctx.font = '9px monospace'; ctx.textAlign = 'left';
      for (let d=1; d<=maxDist; d+=1) ctx.fillText(d+'m', cx+4, marker_y + d*ppm - 2);
      ctx.strokeStyle = '#0ea5e9'; ctx.lineWidth = 1.5; ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.arc(cx, marker_y, target*ppm, 0, Math.PI, false); ctx.stroke();
      ctx.setLineDash([]);
      // marker
      const mw = 60;
      ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 6;
      ctx.beginPath(); ctx.moveTo(cx-mw, marker_y); ctx.lineTo(cx+mw, marker_y); ctx.stroke();
      ctx.fillStyle = '#22c55e'; ctx.font = '11px sans-serif'; ctx.textAlign = 'left';
      ctx.fillText('marker '+(s.marker_id ?? '?'), cx+mw+6, marker_y+4);
      if (!s.running || s.distance_m == null) {
        ctx.fillStyle = '#475569'; ctx.font = '13px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(s.running ? 'no marker visible' : 'stopped', W/2, H/2);
        return;
      }
      const dist = s.distance_m;
      const lateral_m = -(s.skew || 0) * dist;
      const drone_x = cx + lateral_m*ppm;
      const drone_y = marker_y + dist*ppm;
      const hfov = (arcParams.cam_hfov_deg || 69) * Math.PI / 180;
      const yaw_off_rad = Math.atan((s.err_x || 0) * Math.tan(hfov/2));
      const dxv = cx - drone_x, dyv = marker_y - drone_y;
      const aimAng = Math.atan2(dyv, dxv);
      const droneAng = aimAng - yaw_off_rad;
      // LoS
      ctx.strokeStyle = '#fbbf2470'; ctx.lineWidth = 1.5; ctx.setLineDash([3,3]);
      ctx.beginPath(); ctx.moveTo(drone_x, drone_y); ctx.lineTo(cx, marker_y); ctx.stroke();
      ctx.setLineDash([]);
      // FOV
      ctx.strokeStyle = '#38bdf833'; ctx.fillStyle = '#38bdf815';
      const fovLen = ppm * Math.max(2.5, dist+1.0);
      ctx.save(); ctx.translate(drone_x, drone_y); ctx.rotate(droneAng);
      ctx.beginPath(); ctx.moveTo(0,0);
      ctx.lineTo(fovLen*Math.cos(-hfov/2), fovLen*Math.sin(-hfov/2));
      ctx.lineTo(fovLen*Math.cos( hfov/2), fovLen*Math.sin( hfov/2));
      ctx.closePath(); ctx.fill(); ctx.stroke();
      ctx.restore();
      // drone
      ctx.save(); ctx.translate(drone_x, drone_y); ctx.rotate(droneAng);
      ctx.fillStyle = '#fbbf24'; ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(15,0); ctx.lineTo(-10,-10); ctx.lineTo(-10,10); ctx.closePath();
      ctx.fill(); ctx.stroke();
      const vx = s.vx_cms||0, vy = s.vy_cms||0;
      if (Math.hypot(vx,vy) > 2) {
        ctx.strokeStyle = '#ef4444'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(0,0); ctx.lineTo(vx*0.7, vy*0.7); ctx.stroke();
      }
      ctx.restore();
      ctx.fillStyle = '#fbbf24'; ctx.font = 'bold 13px monospace'; ctx.textAlign = 'center';
      ctx.fillText(dist.toFixed(2)+' m', (drone_x+cx)/2+14, (drone_y+marker_y)/2+4);
      ctx.fillStyle = '#cbd5e1'; ctx.font = '11px monospace'; ctx.textAlign = 'left';
      ctx.fillText('lateral '+lateral_m.toFixed(2)+' m', 8, H-22);
      ctx.fillText('dist err '+(dist-target).toFixed(2)+' m', 8, H-6);
      ctx.textAlign = 'right';
      ctx.fillText('target '+target.toFixed(2)+' m', W-8, H-22);
      ctx.fillText('err_x '+(s.err_x||0).toFixed(3), W-8, H-6);
    }

    async function arcPoll() {
      try {
        const r = await fetch('/proxy/aruco/state');
        const s = await r.json();
        const st = document.getElementById('arc_status');
        if (s.running) { st.textContent = '● ' + (s.status_msg || 'running'); st.style.color = '#22c55e'; }
        else           { st.textContent = '○ stopped';                          st.style.color = '#94a3b8'; }
        const dl = document.getElementById('arc_drone_label');
        if (s.drone_id && arcActiveId !== s.drone_id) {
          arcActiveId = s.drone_id;
        }
        if (arcActiveId) dl.textContent = '[drone '+arcActiveId+']';
        arcApplyModeUI(s.mode || 'observe', s.allow_live);
        arcRenderReadout(s);
        arcDrawTopDown(s);
      } catch(e) {}
    }

    // Buttons
    document.getElementById('arc_start').onclick = async () => {
      await fetch('/proxy/aruco/start', {method:'POST'});
      const av = document.getElementById('arc_video');
      av.src = '/proxy/aruco/video.mjpg?t=' + Date.now();
      av.setAttribute('data-active', '1');
      // Reload params for the now-active drone
      arcLoadParams();
    };
    document.getElementById('arc_stop').onclick = async () => {
      await fetch('/proxy/aruco/stop', {method:'POST'});
      const av = document.getElementById('arc_video');
      av.src = '';
      av.removeAttribute('data-active');
    };
    document.getElementById('arc_target_lock').onclick = async () => {
      const v = document.getElementById('arc_target_input').value;
      await fetch('/proxy/aruco/target', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({marker: v ? parseInt(v) : null})});
    };
    document.getElementById('arc_target_auto').onclick = async () => {
      document.getElementById('arc_target_input').value = '';
      await fetch('/proxy/aruco/target', {method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({marker: null})});
    };
    document.getElementById('arc_reload').onclick = arcLoadParams;

    function arcShowModeErr(msg, kind) {
      const el = document.getElementById('arc_mode_err');
      if (!el) return;
      clearTimeout(arcShowModeErr._t);
      if (!msg) { el.style.display = 'none'; return; }
      // kind: 'warn' = amber, default = red
      const warn = (kind === 'warn') || msg.startsWith('⚠');
      el.textContent = warn ? msg : ('✗ ' + msg);
      el.style.background = warn ? '#78350f' : '#7f1d1d';
      el.style.color      = warn ? '#fde68a' : '#fecaca';
      el.style.borderColor = warn ? '#f59e0b' : '#ef4444';
      el.style.display = 'inline';
      // Auto-hide warnings in 3s, errors in 8s
      arcShowModeErr._t = setTimeout(() => { el.style.display = 'none'; }, warn ? 3200 : 8000);
    }
    async function arcSetMode(mode) {
      console.log('[arc] set-mode request:', mode);
      // Clear previous error
      document.getElementById('arc_mode_err').style.display = 'none';
      try {
        const r = await fetch('/proxy/aruco/mode', {method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({mode})});
        let j;
        try { j = await r.json(); } catch { j = {}; }
        console.log('[arc] set-mode response:', r.status, j);
        if (!r.ok || !j.ok) {
          const msg = (j.error || j.message || r.statusText || 'unknown') + ' (HTTP ' + r.status + ')';
          arcShowModeErr(msg);
          return;
        }
        arcApplyModeUI(j.mode || mode, arcAllowLive);
      } catch (err) {
        console.error('[arc] set-mode failed:', err);
        arcShowModeErr('network error: ' + err);
      }
    }
    document.getElementById('arc_mode_observe').onclick = () => arcSetMode('observe');
    // LIVE button: always post to server, let it decide. Client-side arcAllowLive
    // is now just a UI hint (disabled state) — if somehow wrong, the server
    // returns 403 with a clear error message.
    // LIVE button: double-click / re-click confirmation (no browser confirm()).
    // First click arms — button pulses + warning shows.
    // Second click within 3 seconds → switches mode.
    // Keeps us off native confirm() dialogs which some browsers auto-dismiss.
    const liveBtn = document.getElementById('arc_mode_live');
    let _arcArmedUntil = 0;
    if (liveBtn) {
      liveBtn.onclick = () => {
        const now = Date.now();
        console.log('[arc] LIVE clicked, arcAllowLive=', arcAllowLive, 'armed=', (now < _arcArmedUntil));
        if (now < _arcArmedUntil) {
          // Armed → actually switch
          _arcArmedUntil = 0;
          liveBtn.style.animation = '';
          liveBtn.textContent = 'LIVE';
          arcShowModeErr(''); // hides
          arcSetMode('live');
          return;
        }
        // First click — arm for 3s
        _arcArmedUntil = now + 3000;
        liveBtn.style.animation = 'arcpulse 0.6s ease-in-out infinite';
        liveBtn.textContent = 'LIVE — click again';
        arcShowModeErr('⚠ Click LIVE again within 3 s to confirm');
        setTimeout(() => {
          if (Date.now() >= _arcArmedUntil) {
            _arcArmedUntil = 0;
            liveBtn.style.animation = '';
            liveBtn.textContent = 'LIVE';
            arcShowModeErr('');
          }
        }, 3100);
      };
      console.log('[arc] LIVE button click handler attached');
    } else {
      console.error('[arc] could not find arc_mode_live button — handler NOT attached');
    }

    async function arcPostCmd(path, confirmMsg) {
      if (confirmMsg && !confirm(confirmMsg)) return;
      const r = await fetch(path, {method:'POST'});
      const j = await r.json();
      if (!j.ok && j.error) alert(path + ' refused: ' + j.error);
    }
    document.getElementById('arc_takeoff').onclick   = () => arcPostCmd('/proxy/aruco/takeoff',   'Send TAKEOFF to the active drone?');
    document.getElementById('arc_land').onclick      = () => arcPostCmd('/proxy/aruco/land',      'Send LAND to the active drone?');
    document.getElementById('arc_rc_stop').onclick   = () => arcPostCmd('/proxy/aruco/rc_stop',   null);
    document.getElementById('arc_emergency').onclick = () => arcPostCmd('/proxy/aruco/emergency', '⛔ EMERGENCY STOP — cut motors immediately. Confirm?');

    arcLoadParams();
    setInterval(arcPoll, 500);    // was 250ms/4Hz → 2Hz; 500ms is plenty for the readout
    arcPoll();
    // Version marker — if this string doesn't appear in the DOM,
    // you're running stale JS (restart the Python server or hard-refresh).
    const BUILD = 'cz-start-btn-paused-visibility';
    console.log('[arc] init complete, build=' + BUILD);
    // FC/C2 version poll. Fires once at init + every 30 s. If the FC
    // reports a code_version that doesn't match the C2's, flash the
    // red banner at the top of the page with both versions so the
    // operator sees immediately they need to restart the FC.
    (function pollFcVersion() {
      async function check() {
        try {
          const r = await fetch('/proxy/fc_version');
          const d = await r.json();
          const box = document.getElementById('version_mismatch_banner');
          const detail = document.getElementById('version_mismatch_detail');
          if (!box || !detail) return;
          if (d.match) {
            box.style.display = 'none';
            return;
          }
          const fcVer = d.fc_code_version || (d.fc_error ? ('FC unreachable: ' + d.fc_error) : 'unknown');
          detail.innerHTML =
            '<span style="color:#fde68a;">C2:</span> ' + (d.c2_code_version || '?') +
            (d.c2_git_sha ? ' <span style="color:#86efac;">[' + d.c2_git_sha + ']</span>' : '') +
            '<br><span style="color:#fde68a;">FC:</span> ' + fcVer +
            (d.fc_git_sha ? ' <span style="color:#fca5a5;">[' + d.fc_git_sha + ']</span>' : '');
          box.style.display = '';
        } catch (e) {
          // Network hiccup — ignore; we'll try again on the next tick.
        }
      }
      check();
      setInterval(check, 30000);
    })();
    const ver = document.createElement('span');
    ver.id = 'arc_build_tag';
    ver.style.cssText = 'font-size:10px;color:#10b981;margin-left:8px;font-weight:700;';
    ver.textContent = 'build ' + BUILD;
    const hdr = document.querySelector('#aruco_panel h3');
    if (hdr) hdr.appendChild(ver);

    // Immediate visible click counter — proves the button event fires, even
    // if the subsequent fetch hangs or the server is wedged. Counts every
    // LIVE / OBSERVE / Takeoff / Land / RC-stop / Emergency press.
    let _arcClicks = 0;
    const clickTag = document.createElement('span');
    clickTag.id = 'arc_click_counter';
    clickTag.style.cssText = 'font-size:10px;color:#fbbf24;margin-left:8px;';
    clickTag.textContent = 'clicks: 0';
    if (hdr) hdr.appendChild(clickTag);
    function arcBumpClicks(src) {
      _arcClicks += 1;
      clickTag.textContent = 'clicks: ' + _arcClicks + ' (' + src + ')';
      console.log('[arc] click #' + _arcClicks + ' from ' + src);
    }
    // Wire onto the existing buttons defensively. We use addEventListener so
    // we don't overwrite the onclick handlers that actually do the work.
    ['arc_mode_observe','arc_mode_live','arc_takeoff','arc_land','arc_rc_stop','arc_emergency','arc_start','arc_stop'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('click', () => arcBumpClicks(id));
    });
  })();
  