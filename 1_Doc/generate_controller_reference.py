#!/usr/bin/env python3
"""Generate the Remote Controller Reference PDF.

Produces a self-contained operator document covering every editable
parameter and control surface of the remote web controller
(controller/remote_web_controller.py, port 8090). The logo is pulled
from 1_Doc/team_logo_transparent.png so it renders cleanly on white.

Parameter text is intentionally duplicated from PARAM_INFO in the JS —
we want the PDF to be readable without the UI open — but kept brief.

Run:
    python3 1_Doc/generate_controller_reference.py

Writes 1_Doc/Remote_Controller_Reference.pdf.
"""
from __future__ import annotations

from pathlib import Path
from datetime import date

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor, black, white, Color
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Image as RLImage, Table, TableStyle, KeepTogether,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Prefer the pre-downsampled variant (480 px, ~300 KB) for the PDF so
# the mini-logo in the page header doesn't bloat every page by 3 MB.
# Fall back to the full-size transparent PNG, then the original JPEG.
_LOGO_CANDIDATES = (
    HERE / "team_logo_pdf.png",
    HERE / "team_logo_transparent.png",
    HERE / "team_logo.png",
)
LOGO = next((p for p in _LOGO_CANDIDATES if p.exists()), _LOGO_CANDIDATES[0])
OUT  = HERE / "Remote_Controller_Reference.pdf"

# ── Palette ──────────────────────────────────────────────────────────
BRAND    = HexColor("#0369a1")   # deep sky-600
ACCENT   = HexColor("#0ea5e9")   # sky-500
MUTED    = HexColor("#475569")   # slate-600
HEADER_BG= HexColor("#e0f2fe")   # sky-100 (table header tint)
GRID     = HexColor("#cbd5e1")   # slate-300
SOFT     = HexColor("#f1f5f9")   # slate-100 (zebra row)
CODE_BG  = HexColor("#f8fafc")   # code inline
TEXT     = HexColor("#0f172a")


# ── Styles ───────────────────────────────────────────────────────────
def _styles():
    ss = getSampleStyleSheet()
    S = {}
    S["CoverTitle"] = ParagraphStyle(
        "CoverTitle", parent=ss["Title"], fontSize=28, leading=34,
        textColor=BRAND, alignment=TA_CENTER, spaceAfter=8,
        fontName="Helvetica-Bold")
    S["CoverSub"] = ParagraphStyle(
        "CoverSub", parent=ss["Title"], fontSize=14, leading=18,
        textColor=MUTED, alignment=TA_CENTER, spaceAfter=4,
        fontName="Helvetica")
    S["CoverTeam"] = ParagraphStyle(
        "CoverTeam", parent=ss["Title"], fontSize=11, leading=14,
        textColor=ACCENT, alignment=TA_CENTER, fontName="Helvetica-Oblique")
    S["H1"] = ParagraphStyle(
        "H1", parent=ss["Heading1"], fontSize=18, leading=22,
        textColor=BRAND, spaceBefore=14, spaceAfter=8,
        fontName="Helvetica-Bold")
    S["H2"] = ParagraphStyle(
        "H2", parent=ss["Heading2"], fontSize=13, leading=17,
        textColor=ACCENT, spaceBefore=10, spaceAfter=4,
        fontName="Helvetica-Bold")
    S["H3"] = ParagraphStyle(
        "H3", parent=ss["Heading3"], fontSize=11, leading=14,
        textColor=TEXT, spaceBefore=8, spaceAfter=2,
        fontName="Helvetica-Bold")
    S["Body"] = ParagraphStyle(
        "Body", parent=ss["BodyText"], fontSize=9.5, leading=13,
        textColor=TEXT, spaceAfter=4, fontName="Helvetica",
        alignment=TA_LEFT)
    S["Small"] = ParagraphStyle(
        "Small", parent=ss["BodyText"], fontSize=8, leading=10.5,
        textColor=MUTED, spaceAfter=3, fontName="Helvetica")
    S["Key"] = ParagraphStyle(
        "Key", parent=ss["BodyText"], fontSize=9, leading=11,
        textColor=TEXT, fontName="Courier-Bold")
    S["ParamName"] = ParagraphStyle(
        "ParamName", parent=ss["BodyText"], fontSize=8.4, leading=10.5,
        textColor=BRAND, fontName="Courier-Bold")
    S["ParamLabel"] = ParagraphStyle(
        "ParamLabel", parent=ss["BodyText"], fontSize=8.4, leading=10.5,
        textColor=TEXT, fontName="Helvetica-Bold")
    S["ParamDesc"] = ParagraphStyle(
        "ParamDesc", parent=ss["BodyText"], fontSize=8.2, leading=10.5,
        textColor=TEXT, fontName="Helvetica")
    S["TOC"] = ParagraphStyle(
        "TOC", parent=ss["BodyText"], fontSize=10.5, leading=15,
        textColor=TEXT, fontName="Helvetica")
    S["Note"] = ParagraphStyle(
        "Note", parent=ss["BodyText"], fontSize=9, leading=12,
        textColor=MUTED, backColor=SOFT, borderPadding=6,
        borderColor=GRID, borderWidth=0.5, spaceAfter=6,
        fontName="Helvetica-Oblique")
    return S


# ── Page decorations (header/footer) ─────────────────────────────────
def _on_later_pages(canv, doc):
    canv.saveState()
    w, h = A4
    # Header rule
    canv.setStrokeColor(GRID); canv.setLineWidth(0.4)
    canv.line(18*mm, h - 14*mm, w - 18*mm, h - 14*mm)
    canv.setFillColor(MUTED); canv.setFont("Helvetica", 8)
    canv.drawString(18*mm, h - 11*mm,
                    "Remote Controller Reference — Team To Be Defined · SDC26")
    # Mini-logo top-right
    try:
        canv.drawImage(str(LOGO), w - 36*mm, h - 19*mm,
                       width=18*mm, height=18*mm,
                       mask="auto", preserveAspectRatio=True)
    except Exception:
        pass
    # Footer
    canv.setStrokeColor(GRID); canv.line(18*mm, 14*mm, w - 18*mm, 14*mm)
    canv.setFillColor(MUTED); canv.setFont("Helvetica", 7.5)
    canv.drawString(18*mm, 10*mm, f"Generated {date.today().isoformat()}")
    canv.drawRightString(w - 18*mm, 10*mm, f"Page {doc.page}")
    canv.restoreState()


def _on_first_page(canv, doc):
    # Cover has no header/footer decorations — just the footer timestamp.
    canv.saveState()
    w, h = A4
    canv.setFillColor(MUTED); canv.setFont("Helvetica", 8)
    canv.drawString(18*mm, 12*mm,
                    "1_Doc/Remote_Controller_Reference.pdf — regenerate via generate_controller_reference.py")
    canv.drawRightString(w - 18*mm, 12*mm, f"Generated {date.today().isoformat()}")
    canv.restoreState()


# ── Content helpers ──────────────────────────────────────────────────
def _h1(text, S): return Paragraph(text, S["H1"])
def _h2(text, S): return Paragraph(text, S["H2"])
def _h3(text, S): return Paragraph(text, S["H3"])
def _p(text, S):  return Paragraph(text, S["Body"])
def _sm(text, S): return Paragraph(text, S["Small"])
def _note(text, S): return Paragraph(text, S["Note"])


def _kbd_table(items, S):
    """items = list of (keys, action) tuples."""
    data = [["Key(s)", "Action"]]
    for k, a in items:
        data.append([Paragraph(k, S["Key"]), Paragraph(a, S["ParamDesc"])])
    t = Table(data, colWidths=[45*mm, 110*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), HEADER_BG),
        ("TEXTCOLOR", (0,0), (-1,0), BRAND),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
        ("GRID", (0,0), (-1,-1), 0.3, GRID),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, SOFT]),
    ]))
    return t


def _button_table(items, S):
    """items = list of (label, description). Styled like buttons."""
    data = [["UI control", "What it does"]]
    for lbl, desc in items:
        data.append([Paragraph(lbl, S["ParamLabel"]),
                     Paragraph(desc, S["ParamDesc"])])
    t = Table(data, colWidths=[55*mm, 110*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), HEADER_BG),
        ("TEXTCOLOR", (0,0), (-1,0), BRAND),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
        ("GRID", (0,0), (-1,-1), 0.3, GRID),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, SOFT]),
    ]))
    return t


def _param_table(rows, S):
    """rows = list of dicts: key, label, default, range, units, body."""
    data = [["Key / Label", "Default", "Range", "Purpose"]]
    for r in rows:
        title = f"<b>{r['label']}</b><br/><font name='Courier-Bold' size='7.6' color='#0369a1'>{r['key']}</font>"
        dflt  = r.get("default", "—")
        rng   = r.get("range", "—")
        units = r.get("units", "")
        if units:
            rng = f"{rng}<br/><font size='7' color='#64748b'>[{units}]</font>"
        body  = r.get("body", "")
        data.append([
            Paragraph(title, S["ParamDesc"]),
            Paragraph(str(dflt), S["ParamDesc"]),
            Paragraph(rng, S["ParamDesc"]),
            Paragraph(body, S["ParamDesc"]),
        ])
    t = Table(data, colWidths=[42*mm, 18*mm, 22*mm, 90*mm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), HEADER_BG),
        ("TEXTCOLOR", (0,0), (-1,0), BRAND),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
        ("GRID", (0,0), (-1,-1), 0.3, GRID),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, SOFT]),
    ]))
    return t


# ── Parameter catalogs ───────────────────────────────────────────────
OBSERVER_PD = [
    {"key":"hover_distance_m","label":"Hover distance",
     "default":"2.0","range":"0.5–4.0","units":"metres","body":
     "Target stand-off distance from the marker during hover. The mission flies "
     "forward until the marker is this far away, then holds. Smaller = closer "
     "/ more detail but less safety margin against the net. 2&nbsp;m is the "
     "rules-safe default for SDC26."},
    {"key":"fb_max","label":"Approach speed (forward)",
     "default":"45","range":"0–100","units":"% RC","body":
     "Upper clamp for forward throttle when approaching a marker. Scales the P "
     "gain output — higher = faster approach but larger overshoot. Pair with "
     "dist_p: the effective command is <i>min(fb_max, dist_p · err_dist)</i>."},
    {"key":"fb_back_max","label":"Retreat speed (backward)",
     "default":"45","range":"0–100","units":"% RC","body":
     "Upper clamp for backward throttle when the drone is too close. Set lower "
     "than fb_max — retreats tend to happen near the net and should be "
     "cautious."},
    {"key":"dist_p","label":"Approach aggressiveness",
     "default":"20","range":"0–60","units":"P · err_dist(m)","body":
     "Proportional gain turning distance error (m) into forward RC%. Higher "
     "values react more sharply. Start low (10–20) and raise until hover "
     "distance is reached without overshoot."},
    {"key":"ema_alpha","label":"EMA smoothing (α)",
     "default":"0.3","range":"0.05–0.95","units":"0..1","body":
     "First-order low-pass on camera-derived errors. 1.0 = no smoothing "
     "(fastest, jitteriest); 0.05 = heavy smoothing (laggy). Typical 0.25–0.5."},
    {"key":"deadband_x","label":"Yaw/lateral dead-band (err_x)",
     "default":"0.05","range":"0–0.30","units":"normalised","body":
     "Below this threshold, yaw + strafe commands are zero. Stops the drone "
     "hunting around an already-centred marker. Typical 0.03–0.08."},
    {"key":"deadband_y","label":"Altitude dead-band (err_y)",
     "default":"0.06","range":"0–0.30","units":"normalised","body":
     "Below this threshold vertical command is zero. 0.05–0.1 typical."},
    {"key":"deadband_skew","label":"Skew dead-band",
     "default":"0.04","range":"0–0.30","units":"normalised","body":
     "Below this threshold the perpendicular-alignment (strafe) command is "
     "zero — small marker tilt doesn't need correcting."},
    {"key":"deadband_dist_m","label":"Distance dead-band",
     "default":"0.2","range":"0–1.0","units":"metres","body":
     "Below this distance error the forward/back command is zero. Keeps the "
     "drone parked once it's within range of hover_distance_m."},
    {"key":"yaw_p","label":"Yaw P-gain",
     "default":"10","range":"0–50","units":"per err_x","body":
     "Proportional gain from horizontal image-error to yaw RC%. Higher = "
     "snappier rotation but more overshoot."},
    {"key":"skew_p","label":"Lateral P-gain",
     "default":"25","range":"0–50","units":"per skew","body":
     "Proportional gain from marker-tilt to sideways RC%. Drives the strafe "
     "that produces a head-on approach."},
    {"key":"alt_p","label":"Altitude P-gain",
     "default":"25","range":"0–100","units":"per err_y","body":
     "Proportional gain from vertical image-error to vertical RC%."},
    {"key":"d_yaw","label":"Yaw D-damping",
     "default":"0.4","range":"0–2","units":"per °/s (gyro)","body":
     "Derivative damping on yaw using the gyro. Cancels oscillation — if the "
     "drone visibly orbits the marker, increase."},
    {"key":"d_lr","label":"Lateral D-damping",
     "default":"0.4","range":"0–2","units":"per cm/s (vgy)","body":
     "Derivative damping on sideways RC using the body-frame Y velocity. "
     "Prevents over-strafing."},
    {"key":"d_ud","label":"Vertical D-damping",
     "default":"0.5","range":"0–2","units":"per cm/s (vgz)","body":
     "Derivative damping on vertical RC using the body-frame Z velocity."},
    {"key":"d_fb","label":"Fwd/back D-damping",
     "default":"0.4","range":"0–2","units":"per cm/s (vgx)","body":
     "Derivative damping on forward/back RC using body-frame X velocity. Most "
     "important D term for not slamming into the net. Combined with the "
     "boundary guard this is the primary brake during fast approaches."},
    {"key":"yaw_max","label":"Clamp · max yaw",
     "default":"15","range":"0–80","units":"% RC","body":
     "Hard upper limit for yaw RC%. 20–30 typical — keeps the drone from "
     "spinning wildly on large errors."},
    {"key":"lr_max","label":"Clamp · max lateral",
     "default":"20","range":"0–100","units":"% RC","body":
     "Hard upper limit for sideways RC%. 20–40 typical."},
    {"key":"ud_max","label":"Clamp · max vertical",
     "default":"25","range":"0–100","units":"% RC","body":
     "Hard upper limit for vertical RC%. 20–40 typical."},
    {"key":"rc_min","label":"RC dead-floor",
     "default":"2","range":"0–10","units":"% RC","body":
     "Below this magnitude the RC output is forced to zero. Anafi ignores very "
     "small RC values anyway — this prevents buzzing while hovered."},
    {"key":"cam_hfov_deg","label":"Cam H-FOV (drawing only)",
     "default":"69","range":"30–110","units":"degrees","body":
     "Used ONLY to draw the camera cone in the top-down view — does NOT affect "
     "PnP or control. 69° is the Anafi nominal."},
    {"key":"marker_size_m","label":"Observer marker size",
     "default":"0.5","range":"0.05–2.0","units":"metres","body":
     "Physical marker side length for the observer's own PnP. Must match the "
     "printed markers. SDC26 markers are 0.5&nbsp;m. Keep in sync with the "
     "Position Tracker's marker size."},
]

POSITION_TRACKER = [
    {"key":"detect_profile","label":"Detection profile",
     "default":"Balanced","range":"Balanced / Sensitive / Strict","body":
     "Preset for the ArUco detector parameters. Sensitive catches distant / "
     "partially-lit markers at higher CPU cost; Strict rejects noisy "
     "detections; Balanced is the default."},
    {"key":"fov_deg","label":"Camera H-FOV",
     "default":"69","range":"40–120","units":"degrees","body":
     "Horizontal field-of-view used to synthesise the intrinsics matrix when "
     "no calibration file is loaded. Anafi 4K ≈ 69°. Uploading a .npz "
     "calibration overrides this."},
    {"key":"latency_ms","label":"Video-to-IMU latency",
     "default":"200","range":"0–800","units":"ms","body":
     "How old the camera frame is relative to the current IMU sample. The "
     "positioner rewinds the IMU buffer by this amount. Use the header Latency "
     "row with auto-set enabled to have the C2→FC + FC→drone + video decode "
     "total pushed here automatically."},
    {"key":"imu_weight","label":"IMU ↔ ArUco blend",
     "default":"0.30","range":"0–1","body":
     "Mix between pure ArUco pose (0) and pure IMU dead-reckoning (1). Higher "
     "= smoother but drifts more during marker outages. 30% is a good default."},
    {"key":"enable_kalman_filter","label":"Kalman filter",
     "default":"ON","range":"on / off","body":
     "Per-axis 1-D Kalman on x/y/z. Smoother and handles brief dropouts; OFF "
     "means pose jumps straight to the last ArUco solution — noisier but "
     "zero added latency."},
    {"key":"marker_size_m","label":"Marker size",
     "default":"0.5","range":"0.05–2.0","units":"metres","body":
     "Physical marker side length for PnP. Overrides arena_config.json when "
     "set. SDC26 markers are 0.5&nbsp;m."},
    {"key":"top_k_markers","label":"Top-K markers",
     "default":"0 (auto=4)","range":"0–10","body":
     "Use only the N closest markers in the weighted-mean fusion. 0 = auto "
     "(4). Smaller K = faster, less robust. Larger K = more samples but "
     "includes less-accurate distant detections."},
    {"key":"outlier_reject_m","label":"Outlier reject distance",
     "default":"2.5","range":"0.1–20","units":"metres","body":
     "Per-marker poses further than this from the weighted-mean position are "
     "rejected before re-averaging. Tighten for a tidy small arena; loosen if "
     "markers are far apart."},
    {"key":"pose_hold_sec","label":"Pose hold (dead-reckon)",
     "default":"0.8","range":"0–10","units":"seconds","body":
     "After the last valid ArUco fix, keep publishing pose via IMU dead-"
     "reckoning for this many seconds. Too short → pose vanishes every "
     "blink; too long → stale pose drifts metres. 0.5–1.0&nbsp;s typical."},
    {"key":"min_ref_count","label":"Minimum reference markers",
     "default":"1","range":"1–12","body":
     "Require at least this many markers visible before a fused pose is "
     "accepted. 2–3 gives a much more robust fix by cross-checking."},
    {"key":"min_ref_weight","label":"Minimum reference weight",
     "default":"0.0","range":"0–1","body":
     "Require the best-matching marker to have at least this fused weight. "
     "Rejects very low-confidence fits. 0.2–0.4 is restrictive but clean."},
    {"key":"meas_blend_min","label":"Measurement blend — low",
     "default":"0.35","range":"0–1","units":"α","body":
     "Minimum EMA α applied to fresh ArUco measurements. Used when fix "
     "quality is high (trust the filter). Lower = more Kalman smoothing."},
    {"key":"meas_blend_max","label":"Measurement blend — high",
     "default":"0.85","range":"0–1","units":"α","body":
     "Maximum EMA α used when fix quality is low (trust fresh measurements "
     "more). The positioner interpolates between min and max based on "
     "residual error and ref count."},
    {"key":"vel_blend","label":"Velocity blend",
     "default":"0.25","range":"0–1","body":
     "Blend between IMU velocity (0) and Kalman-state velocity derivative "
     "(1). 0.25 = 25% Kalman-derived, 75% raw IMU. Higher = smoother vz/vy "
     "plots but slower reaction."},
    {"key":"max_state_dt","label":"Max state Δt",
     "default":"1.0","range":"0.05–10","units":"seconds","body":
     "If more than this passes between updates, the Kalman state is reset "
     "instead of extrapolated. Prevents exploding covariance during outages."},
    {"key":"kalman_process_var","label":"Kalman Q (process var)",
     "default":"1e-3","range":"1e-6–10","body":
     "How much the state is expected to change between steps. Low Q = "
     "smooth/sluggish; high Q = reacts faster but noisier. Try 5e-4 for "
     "smooth hover, 5e-3 for dynamic missions."},
    {"key":"kalman_meas_var","label":"Kalman R (measurement var)",
     "default":"1e-1","range":"1e-6–10","body":
     "How noisy the ArUco measurements are. Low R = trust camera (snaps to "
     "detections); high R = trust model (smoother but lagging). Large "
     "markers at short range can use 1e-2; distant markers 3e-1."},
]


# ── Build document ───────────────────────────────────────────────────
def build():
    S = _styles()
    story = []

    # ─── Cover ────────────────────────────────────────────────────────
    story.append(Spacer(1, 36*mm))
    if LOGO.exists():
        img = RLImage(str(LOGO), width=60*mm, height=60*mm,
                      kind="proportional", mask="auto")
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Spacer(1, 14*mm))
    story.append(Paragraph("Remote Controller", S["CoverTitle"]))
    story.append(Paragraph("Operator Reference", S["CoverTitle"]))
    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(
        "All editable parameters &amp; the full control interface",
        S["CoverSub"]))
    story.append(Spacer(1, 30*mm))
    story.append(Paragraph("Team <b>To Be Defined</b> — SDC26", S["CoverTeam"]))
    story.append(Paragraph(
        f"Document generated {date.today().strftime('%B %d, %Y')}",
        S["CoverTeam"]))
    story.append(PageBreak())

    # ─── Table of Contents ────────────────────────────────────────────
    story.append(_h1("Contents", S))
    toc_items = [
        ("1", "Introduction &amp; architecture"),
        ("2", "Starting the controller"),
        ("3", "UI layout at a glance"),
        ("4", "Header row — drone bar, LAND ALL, Config, theme, logo"),
        ("5", "Latency widget"),
        ("6", "Keyboard shortcuts"),
        ("7", "WASD &amp; flight grid"),
        ("8", "Takeoff, land, recovery, logging"),
        ("9", "Advanced SDK controls"),
        ("10","Anafi / Olympe panel"),
        ("11","Video stream"),
        ("12","Telemetry panel"),
        ("13","Position Tracker"),
        ("14","ArUco Seek"),
        ("15","Mission Planner &amp; Special Missions"),
        ("16","Arena Configuration"),
        ("17","Tuning Parameters panel"),
        ("18","Parameter reference — Observer PD (22 knobs)"),
        ("19","Parameter reference — Position Tracker (16 knobs)"),
        ("20","Theme switcher"),
        ("21","Info icons (ⓘ)"),
        ("22","Troubleshooting quick-ref"),
    ]
    for num, title in toc_items:
        story.append(Paragraph(
            f"<font color='#0369a1'><b>{num}</b></font> &nbsp; {title}",
            S["TOC"]))
    story.append(PageBreak())

    # ─── 1. Intro ─────────────────────────────────────────────────────
    story.append(_h1("1 &nbsp; Introduction &amp; architecture", S))
    story.append(_p(
        "The Remote Web Controller (<font name='Courier-Bold'>controller/"
        "remote_web_controller.py</font>, default port <b>8090</b>) is the "
        "single operator cockpit for the Team To Be Defined drone swarm. It "
        "proxies commands to per-drone flight controllers "
        "(<font name='Courier-Bold'>controller_unified/unified_api_server.py</font>, "
        "port <b>8080</b> on each Raspberry Pi) and exposes every tunable "
        "knob, every pose-tracking parameter, and every mission workflow in "
        "one browser page.", S))
    story.append(_p(
        "The UI is a single self-contained HTML/JS page built from the "
        "Python file. All state is queried via HTTP (drone config, telemetry, "
        "position) and pushed via HTTP POST or a WebSocket-free SSE channel "
        "for positions. A single operator can switch between up to 5 Anafi "
        "drones by number-key or by clicking the drone bar.", S))

    # ─── 2. Starting ──────────────────────────────────────────────────
    story.append(_h1("2 &nbsp; Starting the controller", S))
    story.append(_p(
        "On the control laptop (the C2):", S))
    story.append(_sm(
        "<font name='Courier-Bold'>cd sdc-tobe/controller</font><br/>"
        "<font name='Courier-Bold'>python3 remote_web_controller.py</font>", S))
    story.append(_p(
        "Open <font name='Courier-Bold'>http://localhost:8090</font> in a "
        "modern browser. The page starts empty until it polls "
        "<font name='Courier-Bold'>/proxy/drones</font> and paints the drone "
        "bar. Each Raspberry Pi must already be running its per-drone "
        "<font name='Courier-Bold'>unified_api_server.py</font> on port 8080.", S))
    story.append(_note(
        "<b>Tip —</b> If the drone bar stays empty, verify the Pi's "
        "<font name='Courier-Bold'>flightctrlN</font> hostname resolves "
        "(from <font name='Courier-Bold'>controller/drones_config.json</font>) "
        "and that the Pi's API answers <font name='Courier-Bold'>GET /ping</font>.",
        S))

    # ─── 3. UI layout ─────────────────────────────────────────────────
    story.append(_h1("3 &nbsp; UI layout at a glance", S))
    story.append(_p(
        "From top to bottom the page is organised into collapsible sections. "
        "Each section remembers its collapsed state in <i>localStorage</i> so "
        "each operator gets the layout they left the last session with.", S))
    layout = [
        ("Header row", "Drone selector, LAND ALL, Config, theme toggle, logo."),
        ("Latency widget", "C2→FC + FC→drone RTT plus video offset; "
                            "auto-applies total to Position Tracker latency."),
        ("Tuning Parameters", "The single home for every live-tunable knob — "
                              "observer PD and position tracker side-by-side."),
        ("WASD Key controls", "The 3×3 key grid + takeoff/land/recover."),
        ("Anafi / Olympe controls", "Gimbal, magnetometer, environment, Wi-Fi, "
                                     "camera zoom, altitude/speed/tilt limits."),
        ("Video stream", "MJPEG or UDP-forward live feed, with zoom slider."),
        ("Telemetry", "Battery, GPS, altitude, attitude, velocity, state."),
        ("ArUco Seek", "Visual-servo hover panel (observer/live mode)."),
        ("Mission Planner", "Free-form step list (takeoff, move, wait, land)."),
        ("Special Missions", "Pre-built missions: Scan-all, Capture-targets."),
        ("Position Tracker", "Arena-frame 2D+3D views with live pose fusion."),
        ("Arena Configuration", "Marker layout editor &amp; physical dims."),
    ]
    story.append(_button_table(layout, S))
    story.append(PageBreak())

    # ─── 4. Header ────────────────────────────────────────────────────
    story.append(_h1("4 &nbsp; Header row", S))
    header_rows = [
        ("Drone bar",
         "Row of drone buttons (one per entry in "
         "<font name='Courier-Bold'>drones_config.json</font>). Click to "
         "switch active drone; hotkeys 1–5 do the same."),
        ("LAND ALL (0)",
         "Red emergency button. Lands every drone in the fleet. Keyboard "
         "shortcut is the zero key (not in the movement map, safe to hit one-handed)."),
        ("Config",
         "Opens the Drone Fleet configuration modal. Add / remove / rename "
         "drones, change their base URL. Saves to "
         "<font name='Courier-Bold'>drones_config.json</font>."),
        ("Theme toggle",
         "☀️ Light ↔ 🌙 Dark. Persisted in localStorage.key "
         "<font name='Courier-Bold'>sdc_theme</font>."),
        ("Team logo",
         "Served from <font name='Courier-Bold'>/logo.png</font> (alpha-"
         "masked variant if present, falls back to the original)."),
    ]
    story.append(_button_table(header_rows, S))

    # ─── 5. Latency widget ────────────────────────────────────────────
    story.append(_h1("5 &nbsp; Latency widget", S))
    story.append(_p(
        "Sits below the drone bar. Shows four numbers and one checkbox:", S))
    latency_rows = [
        ("Total",
         "Sum of the three components. Green when &lt; 100&nbsp;ms, amber "
         "100–250, red &gt; 250."),
        ("c2 → fc",
         "Round-trip time from the control laptop to the Pi's unified API, "
         "measured live by pinging <font name='Courier-Bold'>/ping</font>."),
        ("fc → drone",
         "Round-trip time from the Pi to the Anafi's Wi-Fi IP "
         "(<font name='Courier-Bold'>192.168.42.1</font>), measured once per "
         "second via ICMP ping and cached on the Pi."),
        ("video + N ms",
         "User-set constant accounting for local frame "
         "processing/decoding on the C2. Not measured — adjust to taste."),
        ("auto-set latency",
         "When checked, the total is pushed into the Position Tracker "
         "<b>Latency ms</b> slider on every poll so IMU rewinding always "
         "matches the current link conditions."),
    ]
    story.append(_button_table(latency_rows, S))

    # ─── 6. Keyboard ──────────────────────────────────────────────────
    story.append(_h1("6 &nbsp; Keyboard shortcuts", S))
    story.append(_p(
        "The keydown/keyup handler is active whenever no text field is "
        "focused. Keys repeat at ~20&nbsp;Hz while held.", S))
    kbd = [
        ("W / S", "Pitch forward / back (body-frame X)."),
        ("A / D", "Strafe left / right (body-frame Y)."),
        ("Q / E", "Rotate left / right (yaw)."),
        ("R / F", "Climb / descend (vertical thrust)."),
        ("X or Space", "Hard stop — zero all RC axes."),
        ("T",      "Takeoff on the active drone."),
        ("L",      "Land the active drone."),
        ("0",      "<b>Panic: LAND ALL</b> — lands every drone in the fleet."),
        ("1 – 5",  "Switch active drone to fleet slot N."),
    ]
    story.append(_kbd_table(kbd, S))
    story.append(_note(
        "<b>Safety —</b> the keydown handler does <i>not</i> re-fire while "
        "a key is held; the Pi sends a 200&nbsp;ms RC keep-alive on its own. "
        "If the browser tab loses focus, all keys are treated as released.",
        S))

    # ─── 7. WASD panel ────────────────────────────────────────────────
    story.append(_h1("7 &nbsp; WASD &amp; flight grid", S))
    story.append(_p(
        "A 3×3 button grid that mirrors the keyboard map — useful for "
        "touchscreens. The centre button is <b>STOP</b>. Each button "
        "press-and-holds the corresponding RC axis while the mouse is down.", S))

    # ─── 8. Takeoff / land / logging ──────────────────────────────────
    story.append(_h1("8 &nbsp; Takeoff, land, recovery, logging", S))
    story.append(_button_table([
        ("Takeoff (T)",       "Command Anafi takeoff. Pre-takeoff diagnostics "
                              "print on server stdout if state is unsafe."),
        ("Land (L)",          "Command soft landing with obstacle avoidance."),
        ("Recover",           "Attempts flat-trim + motor recovery after a crash."),
        ("Safe Takeoff: OFF", "Toggle. When ON, takeoff only proceeds if all "
                              "sensors (magneto, GPS, altimeter, alert state) "
                              "report healthy."),
        ("Enable Telemetry Log", "Start/stop recording telemetry to "
                                 "<font name='Courier-Bold'>remote_telemetry_log.jsonl</font>."),
        ("Download Telemetry Log","Serve the current JSONL file."),
        ("Clear Telemetry Log",   "Truncate the file."),
        ("Command Logging: OFF",  "Start/stop recording every outgoing command "
                                  "with timestamps. High-frequency events "
                                  "(key_down/key_up) throttled to 2&nbsp;Hz."),
        ("Download Command Log",  "Serve <font name='Courier-Bold'>remote_command_log.jsonl</font>."),
        ("Clear Command Log",     "Truncate it."),
    ], S))

    # ─── 9. Advanced SDK ──────────────────────────────────────────────
    story.append(_h1("9 &nbsp; Advanced SDK controls", S))
    story.append(_p(
        "Collapsed by default. Exposes raw Olympe/SDK operations for manual "
        "poking while debugging:", S))
    story.append(_button_table([
        ("Rotate CW 45° / CCW 45°", "Fixed yaw step."),
        ("Up 30cm / Down 30cm",     "Fixed vertical translate."),
        ("Forward / Back / Left / Right 30cm",
                                     "Fixed horizontal translate (body-frame)."),
        ("Stream ON / Stream OFF",  "Raw stream command (debug only — normal "
                                     "video is started from the Video panel)."),
        ("Set Speed",                "Applies the speed-input value (10–100) "
                                     "as the max horizontal speed cap."),
        ("Raw SDK input + Send",     "Send any SDK command string verbatim "
                                     "(e.g. <font name='Courier-Bold'>battery?</font>, "
                                     "<font name='Courier-Bold'>speed?</font>)."),
    ], S))

    # ─── 10. Anafi / Olympe ───────────────────────────────────────────
    story.append(_h1("10 &nbsp; Anafi / Olympe panel", S))
    story.append(_p(
        "Only shown for drones whose type is <i>anafi</i>. Groups Anafi-"
        "specific controls that have no Tello equivalent:", S))
    story.append(_button_table([
        ("Gimbal tilt slider",
         "-90° (straight down) to +30° (up). Apply with Set, or jump to "
         "Down / Forward."),
        ("Magnetometer",
         "Status line + <b>Recalibrate Magnetometer</b> button opens the "
         "figure-8 wizard (rotate the drone through three axes)."),
        ("Max altitude (m)",
         "Firmware-enforced ceiling. 0.5–150, default 5."),
        ("Max vert spd (m/s)",
         "Firmware-enforced vertical speed cap. 0.1–4, default 0.5."),
        ("Max tilt (°)",
         "Firmware-enforced attitude tilt cap. 1–35, default 15."),
        ("Apply Settings",
         "POST the three caps together (applied in one command)."),
        ("Environment",
         "Indoor (GPS-less, relaxed checks) vs Outdoor (GPS-required). "
         "Required before takeoff in the hall."),
        ("Wi-Fi band / channel",
         "Scan + switch between 2.4 GHz and 5 GHz. Use 5 GHz in busy "
         "venues — the Anafi is 5-GHz capable."),
        ("Camera zoom",
         "0% – 100% slider mapped onto the Anafi's digital+optical zoom. "
         "Live update; no arming."),
    ], S))

    # ─── 11. Video ────────────────────────────────────────────────────
    story.append(_h1("11 &nbsp; Video stream", S))
    story.append(_button_table([
        ("Mode: Off",          "No video."),
        ("Mode: MJPEG (Way 1)","The Pi decodes the drone's H.264 stream and "
                                "re-encodes as MJPEG for the browser. Single "
                                "hop, most compatible."),
        ("Mode: UDP forward (Way 2)",
                                "Pi forwards raw UDP to C2 for browser-side "
                                "decode. Lower CPU on Pi, more latency spread."),
        ("Start Video / Stop Video", "Toggle button."),
        ("Video URL",          "Shows the current MJPEG or forward URL — "
                                "useful for VLC debugging."),
    ], S))

    # ─── 12. Telemetry ────────────────────────────────────────────────
    story.append(_h1("12 &nbsp; Telemetry panel", S))
    story.append(_p(
        "Scrolling monospace block polled every ~250&nbsp;ms:", S))
    story.append(_sm(
        "battery %, board temperature, TOF/barometer altitude, flight-time, "
        "horizontal/vertical speed, Wi-Fi SNR, attitude pitch/roll/yaw, "
        "velocity vx/vy/vz (cm/s), acceleration ax/ay/az (cm/s²), SDK "
        "version, serial, GPS lat/lon/alt, gimbal pry, connected flag.", S))
    story.append(_note(
        "<b>Battery meter —</b> green ≥ 70%, amber 35–69%, red &lt; 35%. "
        "A drone flashes amber in the drone bar when &lt; 20%.", S))
    story.append(PageBreak())

    # ─── 13. Position Tracker ─────────────────────────────────────────
    story.append(_h1("13 &nbsp; Position Tracker", S))
    story.append(_p(
        "Arena-frame pose fusion — uses the drone's camera + ArUco markers + "
        "onboard IMU to estimate <b>x, y, z, heading</b> in arena "
        "coordinates. Feeds every other system: ArUco Seek, mission "
        "boundary guard, arena view, collision avoidance.", S))
    story.append(_button_table([
        ("Enable checkbox",      "Starts/stops the positioner on the Pi and "
                                  "opens/closes the SSE event stream."),
        ("3D view (Three.js)",    "<b>ON by default.</b> Shows orbit-able "
                                  "arena with the drone as a small axes gizmo. "
                                  "Uncheck to hide. 2D top-down canvas stays "
                                  "visible alongside the 3D view."),
        ("Show all drones",       "When checked, every fleet drone is plotted; "
                                  "unchecked shows only the active drone."),
        ("Coordinate readout",    "X (cyan) · Y (green) · Z (orange) · "
                                  "heading · velocity · ref count · FPS."),
        ("Vx/Vy/Vz, Ax/Ay/Az",    "Body-frame velocity and acceleration from "
                                  "the IMU telemetry."),
        ("Profile / FOV / Latency ms / ArUco ↔ IMU blend",
                                  "Top config row — most common knobs. See "
                                  "parameter reference §19."),
        ("Filters row",           "Kalman on/off, Marker size (m), Top-K, "
                                  "Outlier (m) — apply with <b>Apply filters</b>."),
        ("Precision row (advanced)",
                                  "pose-hold, min refs / min-ref-weight, "
                                  "meas_blend min/max, vel_blend, max Δt, "
                                  "Kalman Q/R. Apply with <b>Apply precision</b>."),
        ("Apply Config",          "Pushes the top row (Profile / FOV / Latency) "
                                  "to the server."),
        ("Upload Calibration (.npz)",
                                  "Load a pre-computed intrinsics file from "
                                  "OpenCV calibration. Overrides FOV."),
        ("Show ArUco Video",      "Annotated MJPEG feed with detected markers, "
                                  "rejection reasons, and fused-pose overlay."),
        ("Record",                "Writes an .mp4 of the annotated "
                                  "feed to <font name='Courier-Bold'>controller_unified/recordings/</font>."),
        ("Raw",                   "Record the raw frame instead of the "
                                  "annotated overlay."),
    ], S))

    # ─── 14. ArUco Seek ───────────────────────────────────────────────
    story.append(_h1("14 &nbsp; ArUco Seek", S))
    story.append(_p(
        "Visual-servo hover panel. The observer continuously runs a PD loop "
        "on whichever marker the mission has set as target (or whichever "
        "marker has the highest confidence in passive mode). Two modes:", S))
    story.append(_button_table([
        ("OBSERVE", "Render + tune — no RC output sent to the drone."),
        ("LIVE",    "Panel writes RC commands to the drone. Requires arming "
                    "(red banner pulses while armed)."),
        ("Takeoff / Land", "Per-drone shortcuts right in the panel."),
        ("STOP RC",  "Zero all RC axes immediately. Does not land."),
        ("Emergency", "Cut motors. Drone will fall — use only to avoid "
                      "imminent destruction."),
        ("Reload",    "Re-fetch observer params from the Pi (useful after "
                      "manual JSON edits)."),
    ], S))

    # ─── 15. Missions ─────────────────────────────────────────────────
    story.append(_h1("15 &nbsp; Mission Planner &amp; Special Missions", S))
    story.append(_p(
        "Two mission runners sit side-by-side. <b>Mission Planner</b> accepts "
        "a free-form step list (takeoff, 100 forward, 90 cw, wait 2, land). "
        "<b>Special Missions</b> are pre-built FSMs that operate in arena "
        "coordinates with collision avoidance between drones.", S))
    story.append(_h2("Scan all ArUco markers", S))
    story.append(_p(
        "Fleet-wide sequential scan. Each drone claims a marker (shared map "
        "prevents collisions), approaches head-on at <b>hover_distance_m</b>, "
        "hovers for <b>Hover s</b>, then chains to the next unclaimed marker. "
        "Skew and approach tolerances control when APPROACH transitions to "
        "HOVER (§18).", S))
    story.append(_button_table([
        ("Target markers",   "<b>1-12</b> default. Use dashes or comma lists. "
                              "0 excluded (origin)."),
        ("Hover s",          "Dwell time per marker (default 1.5&nbsp;s)."),
        ("Approach tol (m)", "Distance-error tolerance for APPROACH→HOVER."),
        ("Skew tol",         "Perpendicularity tolerance (~0.12 ≈ 9°)."),
    ], S))
    story.append(_h2("Capture enemy targets (SDC26)", S))
    story.append(_p(
        "Waypoint navigation to each opposing team's box, hovering with the "
        "camera aimed at the arena centre so many markers stay in view for "
        "localization. Six boxes default (3 red + 3 blue), configurable.", S))
    story.append(_button_table([
        ("Target boxes (JSON)", "List of <font name='Courier-Bold'>{id,x,y,home_team}</font>."),
        ("Home XY",            "World-frame return point (post-capture)."),
        ("Face XY",            "World-frame point the camera faces during transit."),
        ("Altitude (m)",       "Hover altitude above each box (default 1.5)."),
        ("Hover s",            "Must be ≥ 2&nbsp;s (SDC26 capture-hold rule); "
                                "default 4."),
    ], S))

    # ─── 16. Arena Config ─────────────────────────────────────────────
    story.append(_h1("16 &nbsp; Arena Configuration", S))
    story.append(_p(
        "Marker layout editor — the ground truth consumed by the Position "
        "Tracker and Missions. Collapsed by default; click <b>Show</b> to "
        "expand.", S))
    story.append(_button_table([
        ("Arena width / depth (m)",
         "Physical dimensions (20 × 10.8 for SDC26)."),
        ("Height min / max (m)",
         "Ceiling + floor clamps used by the boundary guard."),
        ("Marker size (m)",
         "Physical side length (default 0.5)."),
        ("Marker table",
         "Per-marker row: ID · X · Y · Z · wall (front/back/left/right). "
         "Click <b>+ Add marker</b>, <b>Delete</b> on a row, "
         "<b>Save Config</b>, or <b>Reset to Defaults</b>."),
    ], S))
    story.append(PageBreak())

    # ─── 17. Tuning panel ─────────────────────────────────────────────
    story.append(_h1("17 &nbsp; Tuning Parameters panel", S))
    story.append(_p(
        "The single home for <b>every</b> live-tunable knob. Two columns:", S))
    story.append(_button_table([
        ("Observer PD (left)", "22 sliders driving the visual-servo PD loop. "
                                "Used by every mission — these are the "
                                "\"mission approach\" parameters (flight "
                                "speed / rotation / aggressiveness)."),
        ("Position Tracker (right)",
                                "16 inputs driving arena-frame pose fusion. "
                                "Split into three rows: top (Profile / FOV / "
                                "Latency / IMU blend), Filters (Kalman / "
                                "marker size / Top-K / Outlier), and "
                                "Precision (pose-hold / blend / Kalman Q/R)."),
    ], S))
    story.append(_note(
        "Every knob has a small <b>ⓘ</b> info-icon next to its label. Click "
        "it for a pop-up with the parameter's full explanation, units, and "
        "tuning hints. This is the live equivalent of the tables on the "
        "next two pages.", S))

    # ─── 18. Observer PD table ────────────────────────────────────────
    story.append(PageBreak())
    story.append(_h1("18 &nbsp; Parameter reference — Observer PD", S))
    story.append(_sm(
        "These knobs drive the visual-servo PD loop that every mission "
        "uses to hover in front of markers. Applied live via "
        "<font name='Courier-Bold'>POST /proxy/aruco/params</font>.", S))
    story.append(_param_table(OBSERVER_PD, S))

    # ─── 19. Position Tracker table ───────────────────────────────────
    story.append(PageBreak())
    story.append(_h1("19 &nbsp; Parameter reference — Position Tracker", S))
    story.append(_sm(
        "These knobs drive the multi-marker ArUco + IMU fusion that produces "
        "the arena-frame pose. Applied live via "
        "<font name='Courier-Bold'>POST /proxy/position/config</font>. "
        "Values mirror the module constants in "
        "<font name='Courier-Bold'>controller_unified/ctrl_position.py</font>.",
        S))
    story.append(_param_table(POSITION_TRACKER, S))

    # ─── 20. Theme ────────────────────────────────────────────────────
    story.append(PageBreak())
    story.append(_h1("20 &nbsp; Theme switcher", S))
    story.append(_p(
        "The 🌙/☀️ button in the header toggles between dark (default) and "
        "light. Choice is saved in localStorage under "
        "<font name='Courier-Bold'>sdc_theme</font>. CSS overrides under "
        "<font name='Courier-Bold'>html[data-theme=\"light\"]</font> flip "
        "every major colour including panels, buttons, inputs, info-icons, "
        "and the parameter-info modal.", S))

    # ─── 21. Info icons ───────────────────────────────────────────────
    story.append(_h1("21 &nbsp; Info icons (ⓘ)", S))
    story.append(_p(
        "A 13&nbsp;px circled <i>i</i> appears next to every tuning label. "
        "Click to open a modal with the parameter's full description. Same "
        "text as in this PDF's reference tables — kept in sync by sourcing "
        "from the JavaScript <font name='Courier-Bold'>PARAM_INFO</font> map. "
        "Click outside the box or the <b>Close</b> button to dismiss.", S))

    # ─── 22. Troubleshooting ──────────────────────────────────────────
    story.append(_h1("22 &nbsp; Troubleshooting quick-ref", S))
    story.append(_button_table([
        ("Drone bar empty",
         "<font name='Courier-Bold'>flightctrlN</font> hostname not "
         "resolving, or the Pi's API isn't running on port 8080."),
        ("Takeoff refused",
         "Check the pre-takeoff diagnostic block printed to the Pi server's "
         "stdout (alert / motor / sensors / magneto)."),
        ("Pose jitters &gt; 10 cm",
         "Verify <b>marker_size_m = 0.5</b> in both the observer and the "
         "position tracker; upload a real calibration .npz; raise <b>imu_weight</b>; "
         "turn on Kalman."),
        ("Drone stutters during approach",
         "Increase <b>fb_max</b> slowly, not in one jump. Keep RC tick at "
         "400 ms default."),
        ("Drone flies too fast into net",
         "Lower <b>fb_max</b> and/or <b>dist_p</b>; the boundary guard "
         "uses velocity to predict the next 0.35 s — ensure telemetry is "
         "live (\"Vel\" in coord readout is not dashes)."),
        ("Mission won't rotate after hover",
         "Confirm <b>search_rc</b> is non-zero during SEARCH by inspecting "
         "the JSONL in <font name='Courier-Bold'>controller/logs/</font>."),
        ("Latency widget stays red",
         "<b>fc → drone</b> ping timing out — switch Wi-Fi channel or band "
         "in the Anafi panel."),
    ], S))

    # ─── Build ────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=22*mm, bottomMargin=20*mm,
        title="Remote Controller Reference — Team To Be Defined",
        author="Team To Be Defined", subject="SDC26 operator documentation")
    doc.build(story, onFirstPage=_on_first_page, onLaterPages=_on_later_pages)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.1f} KB, "
          f"logo={'yes' if LOGO.exists() else 'MISSING'})")


if __name__ == "__main__":
    build()
