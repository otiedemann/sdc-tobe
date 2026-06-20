
  // ── Live Telemetry Graphs (standalone, runs independently) ─────────
  (function(){
    const WINDOW_S = 10;
    const SAMPLE_HZ = 20;
    const CANVAS_W = 340, CANVAS_H = 130;
    const GROUPS = [
      {title:'Altitude (cm)',     keys:['height_cm','tof_cm','barometer_cm'], colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Attitude (deg)',    keys:['pitch','roll','yaw'],                colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Velocity (cm/s)',   keys:['vgx','vgy','vgz'],                   colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Acceleration',      keys:['agx','agy','agz'],                   colors:['#22d3ee','#f472b6','#a78bfa']},
      {title:'Speed',             keys:['speed'],                              colors:['#22d3ee']},
      {title:'Battery (%)',       keys:['battery'],                            colors:['#34d399']},
      {title:'Temperature (C)',   keys:['temperature'],                        colors:['#fb923c']},
      {title:'Position (m)',      keys:['pos_x','pos_y','pos_z'],              colors:['#22d3ee','#f472b6','#a78bfa']},
    ];
    const graphs = [];
    let visible = false, rafId = null, sampleTimer = null;

    function init() {
      const c = document.getElementById('graphs_container');
      if (!c || graphs.length) return;
      GROUPS.forEach(g => {
        const wrap = document.createElement('div');
        wrap.style.cssText = 'background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:6px;';
        const hdr = document.createElement('div');
        hdr.style.cssText = 'font-size:11px;color:#94a3b8;margin-bottom:4px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;';
        let html = '<b style="color:#e2e8f0;">' + g.title + '</b>';
        g.keys.forEach((k,i) => { html += '<span style="color:'+g.colors[i]+';">'+k+'</span>'; });
        hdr.innerHTML = html;
        wrap.appendChild(hdr);
        const cv = document.createElement('canvas');
        cv.width = CANVAS_W; cv.height = CANVAS_H;
        cv.style.cssText = 'width:100%;height:auto;display:block;background:#020617;border-radius:4px;';
        wrap.appendChild(cv);
        c.appendChild(wrap);
        graphs.push({keys: g.keys, colors: g.colors, samples: [], canvas: cv, ctx: cv.getContext('2d')});
      });
      console.log('[graphs] init done,', graphs.length, 'graphs');
    }

    function sample() {
      const ts = performance.now();
      const t = (typeof window.lastTelemetry === 'object' && window.lastTelemetry) ? Object.assign({}, window.lastTelemetry) : {};
      if (Array.isArray(window._lastPos)) { t.pos_x = window._lastPos[0]; t.pos_y = window._lastPos[1]; t.pos_z = window._lastPos[2]; }
      graphs.forEach(g => {
        const vals = {}; let any = false;
        g.keys.forEach(k => { const v = t[k]; if (v != null && !isNaN(v)) { vals[k] = Number(v); any = true; } else { vals[k] = null; } });
        if (any) g.samples.push({t: ts, vals});
        const cutoff = ts - WINDOW_S * 1000;
        while (g.samples.length > 0 && g.samples[0].t < cutoff) g.samples.shift();
      });
    }

    function draw() {
      if (!visible) { rafId = null; return; }
      const now = performance.now();
      graphs.forEach(g => {
        const ctx = g.ctx, W = g.canvas.width, H = g.canvas.height;
        ctx.fillStyle = '#020617'; ctx.fillRect(0,0,W,H);
        if (g.samples.length < 2) {
          ctx.fillStyle = '#475569'; ctx.font = '11px sans-serif';
          ctx.textAlign = 'center'; ctx.fillText('waiting for data...', W/2, H/2); ctx.textAlign = 'left';
          return;
        }
        const tMin = now - WINDOW_S*1000, tMax = now;
        let yMin = Infinity, yMax = -Infinity;
        g.samples.forEach(s => g.keys.forEach(k => { if (s.vals[k]!=null) { yMin=Math.min(yMin,s.vals[k]); yMax=Math.max(yMax,s.vals[k]); } }));
        if (!isFinite(yMin)) return;
        if (yMin === yMax) { yMin -= 1; yMax += 1; }
        const pad = (yMax-yMin)*0.1 || 1; yMin -= pad; yMax += pad;
        ctx.strokeStyle = '#1e293b'; ctx.lineWidth = 0.5;
        for (let i=0;i<=4;i++) { const y=(i/4)*H; ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }
        ctx.fillStyle = '#64748b'; ctx.font = '9px monospace';
        for (let i=0;i<=4;i++) { const v = yMin + ((4-i)/4)*(yMax-yMin); ctx.fillText(v.toFixed(1), 2, (i/4)*H + 9); }
        g.keys.forEach((k, ki) => {
          ctx.strokeStyle = g.colors[ki]; ctx.lineWidth = 1.5; ctx.beginPath();
          let started = false;
          g.samples.forEach(s => {
            if (s.vals[k] == null) { started = false; return; }
            const x = ((s.t - tMin) / (tMax - tMin)) * W;
            const y = H - ((s.vals[k] - yMin) / (yMax - yMin)) * H;
            if (!started) { ctx.moveTo(x,y); started = true; } else { ctx.lineTo(x,y); }
          });
          ctx.stroke();
          const last = g.samples[g.samples.length - 1];
          if (last && last.vals[k] != null) {
            ctx.fillStyle = g.colors[ki]; ctx.font = '10px monospace'; ctx.textAlign = 'right';
            ctx.fillText(last.vals[k].toFixed(1), W - 2, 10 + ki * 11); ctx.textAlign = 'left';
          }
        });
      });
      rafId = requestAnimationFrame(draw);
    }

    window.toggleGraphs = function() {
      visible = !visible;
      const btn = document.getElementById('graphs_toggle');
      const panel = document.getElementById('graphs_panel');
      if (btn) btn.textContent = visible ? 'Hide Graphs' : 'Show Graphs';
      if (panel) panel.style.display = visible ? 'block' : 'none';
      console.log('[graphs] toggle ->', visible);
      if (visible) {
        init();
        if (!sampleTimer) sampleTimer = setInterval(sample, 1000 / SAMPLE_HZ);
        if (!rafId) rafId = requestAnimationFrame(draw);
      } else {
        if (sampleTimer) { clearInterval(sampleTimer); sampleTimer = null; }
        if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
      }
    };
    console.log('[graphs] toggleGraphs registered');
  })();
  