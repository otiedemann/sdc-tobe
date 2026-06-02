#!/usr/bin/env python3
"""Build the SDC26 ToBeDefined Technical Report as a branded A4 PDF.

Pure ``reportlab`` (no LaTeX, no external HTML→PDF tools). Embeds the
team logo, follows the three required sections from §1.6.3 of the
regulations (localisation; flight control & trajectory planning; swarm
strategy), and draws all figures natively as vector graphics in the
team's corporate-identity palette. Output:

    1_Doc/SDC26_ToBeDefined_Technical_Report.pdf

The prose is kept faithful to the open-sourced code base
(github.com/otiedemann/sdc-tobe) — the strategy layer is a transparent
rule-based planner, not a learned policy, and the figures depict the
actual mission scripts and decision ladder the code emits.
"""
from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib.colors import HexColor, Color
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfgen import canvas as canvas_mod
from reportlab.graphics.shapes import (
    Drawing, Rect, Line, Circle, String, Polygon, PolyLine, Group,
)
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, KeepTogether,
)

ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "1_Doc" / "team_logo_pdf.png"
OUT = ROOT / "1_Doc" / "SDC26_ToBeDefined_Technical_Report.pdf"

# ---------------------------------------------------------------------------
# Corporate-identity palette (matches the team mission-patch logo)
# ---------------------------------------------------------------------------
NAVY = HexColor("#10243d")     # primary
ACCENT = HexColor("#3b82c4")   # blue accent
RED = HexColor("#d6443a")      # the patch "vector" swoosh — sparing accent
MUTED = HexColor("#6b7280")
RULE = HexColor("#cbd5e1")
CODE_BG = HexColor("#f3f4f6")
SKY = HexColor("#eaf1f8")      # light blue fill
MIST = HexColor("#f5f7fa")     # very light fill
REDZONE = HexColor("#fbe9e7")  # red home-zone tint
BLUEZONE = HexColor("#e7f0fb")  # blue home-zone tint
WHITE = HexColor("#ffffff")
INK = HexColor("#1f2933")


def make_styles():
    s = {}
    s["TitleBig"] = ParagraphStyle(
        "TitleBig", fontName="Times-Bold", fontSize=28, leading=34,
        alignment=TA_CENTER, textColor=NAVY, spaceAfter=8)
    s["Sub"] = ParagraphStyle(
        "Sub", fontName="Times-Roman", fontSize=13, leading=17,
        alignment=TA_CENTER, textColor=MUTED, spaceAfter=4)
    s["Tag"] = ParagraphStyle(
        "Tag", fontName="Helvetica-Bold", fontSize=10, leading=12,
        alignment=TA_CENTER, textColor=ACCENT, spaceAfter=2)
    s["AbstractTitle"] = ParagraphStyle(
        "AbstractTitle", fontName="Helvetica-Bold", fontSize=10,
        leading=12, alignment=TA_LEFT, textColor=NAVY, spaceAfter=4)
    s["H1"] = ParagraphStyle(
        "H1", fontName="Times-Bold", fontSize=15, leading=19,
        textColor=NAVY, spaceBefore=2, spaceAfter=6)
    s["H2"] = ParagraphStyle(
        "H2", fontName="Times-Bold", fontSize=11.5, leading=14,
        textColor=NAVY, spaceBefore=6, spaceAfter=3)
    s["Body"] = ParagraphStyle(
        "Body", fontName="Times-Roman", fontSize=10, leading=13,
        alignment=TA_JUSTIFY, spaceAfter=4)
    s["Caption"] = ParagraphStyle(
        "Caption", fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
        alignment=TA_CENTER, textColor=MUTED, spaceBefore=3, spaceAfter=8)
    s["Code"] = ParagraphStyle(
        "Code", fontName="Courier", fontSize=8.5, leading=11,
        backColor=CODE_BG, borderColor=RULE, borderWidth=0.4,
        borderPadding=6, spaceBefore=2, spaceAfter=6)
    return s


S = make_styles()


def P(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, S[style])


def CAP(text: str) -> Paragraph:
    return Paragraph(text, S["Caption"])


# ---------------------------------------------------------------------------
# Vector-graphics primitives (all figures drawn natively in CI colours)
# ---------------------------------------------------------------------------

def _arrow(g, x1, y1, x2, y2, color=NAVY, w=1.1, head=6.0, dash=None):
    ln = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=w)
    if dash:
        ln.strokeDashArray = dash
    g.add(ln)
    ang = math.atan2(y2 - y1, x2 - x1)
    a1 = ang + math.radians(150)
    a2 = ang - math.radians(150)
    g.add(Polygon(
        [x2, y2,
         x2 + head * math.cos(a1), y2 + head * math.sin(a1),
         x2 + head * math.cos(a2), y2 + head * math.sin(a2)],
        fillColor=color, strokeColor=color))


def _text(g, x, y, t, size=8, color=INK, font="Helvetica", anchor="middle"):
    g.add(String(x, y, t, fontName=font, fontSize=size, fillColor=color,
                 textAnchor=anchor))


def _node(g, x, y, w, h, title, sub=None, fill=SKY, stroke=ACCENT,
          tcolor=NAVY, tsize=8.5, ssize=7, sw=1.1, font="Helvetica-Bold"):
    g.add(Rect(x, y, w, h, fillColor=fill, strokeColor=stroke, strokeWidth=sw,
               rx=4, ry=4))
    cx = x + w / 2.0
    if sub:
        _text(g, cx, y + h / 2.0 + 1.5, title, tsize, tcolor, font)
        _text(g, cx, y + h / 2.0 - 8.0, sub, ssize, MUTED, "Helvetica")
    else:
        _text(g, cx, y + h / 2.0 - 3, title, tsize, tcolor, font)


def _drone(g, cx, cy, r, color=NAVY, body=ACCENT):
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        g.add(Line(cx, cy, cx + dx * r, cy + dy * r,
                   strokeColor=color, strokeWidth=1.3))
        g.add(Circle(cx + dx * r, cy + dy * r, r * 0.42,
                     fillColor=None, strokeColor=color, strokeWidth=1.1))
    g.add(Circle(cx, cy, r * 0.5, fillColor=body, strokeColor=color,
                 strokeWidth=1.0))


def _star(g, cx, cy, r, color=RED):
    pts = []
    for i in range(10):
        rad = r if i % 2 == 0 else r * 0.45
        a = math.radians(-90 + i * 36)
        pts += [cx + rad * math.cos(a), cy + rad * math.sin(a)]
    g.add(Polygon(pts, fillColor=color, strokeColor=color))


def _band(d, h=4, color=NAVY):
    """A thin CI accent band along the top of a figure."""
    d.add(Rect(0, d.height - h, d.width, h, fillColor=color, strokeColor=None))


# ---------------------------------------------------------------------------
# Figure 1 — System architecture
# ---------------------------------------------------------------------------

def fig_architecture():
    W, H = 16.4 * cm, 7.4 * cm
    d = Drawing(W, H)
    d.hAlign = "CENTER"
    d.add(Rect(0, 0, W, H, fillColor=MIST, strokeColor=RULE, strokeWidth=0.6))
    _band(d)

    # Tailnet wrapper
    d.add(Rect(0.3 * cm, 0.3 * cm, W - 0.6 * cm, H - 0.9 * cm,
               fillColor=None, strokeColor=ACCENT, strokeWidth=0.8,
               rx=6, ry=6, strokeDashArray=[3, 3]))
    _text(d, 2.45 * cm, H - 0.62 * cm, "Tailscale tailnet  (MagicDNS, air-gapped)",
          7.5, ACCENT, "Helvetica-Oblique", "middle")

    g = Group()
    colw, rowh = 3.5 * cm, 1.5 * cm
    # Row 1 — drones
    dy = 5.3 * cm
    for i, lab in enumerate(("Anafi 1", "Anafi 2", "…", "Anafi 5")):
        x = (1.0 + i * 3.6) * cm
        if lab == "…":
            _text(g, x + colw / 2, dy + rowh / 2 - 3, "· · ·", 12, MUTED)
            continue
        _node(g, x, dy, colw, rowh, "", fill=WHITE, stroke=NAVY)
        _drone(g, x + 0.7 * cm, dy + rowh / 2, 7, NAVY, ACCENT)
        _text(g, x + colw / 2 + 0.45 * cm, dy + rowh / 2 + 2, lab, 8, NAVY,
              "Helvetica-Bold")
        _text(g, x + colw / 2 + 0.45 * cm, dy + rowh / 2 - 8, "caged",
              6.5, MUTED, "Helvetica")
    # Row 2 — per-drone FC
    fy = 3.2 * cm
    for i in range(4):
        x = (1.0 + i * 3.6) * cm
        if i == 2:
            _text(g, x + colw / 2, fy + rowh / 2 - 3, "· · ·", 12, MUTED)
            continue
        _node(g, x, fy, colw, rowh, "marker_mission  :8080",
              "vision · mission DSL · Olympe", fill=SKY, stroke=ACCENT)
        _arrow(g, x + colw / 2, dy, x + colw / 2, fy + rowh, NAVY, 1.0, 5)
    # Row 3 — C2
    cy = 1.5 * cm
    cx0, cw = 1.0 * cm, 6.7 * cm
    _node(g, cx0, cy, cw, rowh, "marker_mission_c2  :8090",
          "fleet poll · proxy · live dashboard", fill=NAVY, stroke=NAVY,
          tcolor=WHITE)
    # Row 3 — strategy
    sx0 = 8.6 * cm
    _node(g, sx0, cy, cw, rowh, "strategy  :8091 / :8092",
          "roles · rule-based planner · DSL", fill=NAVY, stroke=NAVY,
          tcolor=WHITE)
    # FC -> C2 arrows
    _arrow(g, 2.7 * cm, fy, 3.0 * cm, cy + rowh, NAVY, 1.0, 5)
    _arrow(g, 9.0 * cm, fy, 8.0 * cm, cy + rowh, NAVY, 1.0, 5)
    _arrow(g, 7.7 * cm, cy + rowh / 2, 8.6 * cm, cy + rowh / 2, ACCENT, 1.2, 6)
    _text(g, 8.15 * cm, cy + rowh / 2 + 4, "HTTP/JSON", 6, MUTED,
          "Helvetica-Oblique")
    # Simulator note (right)
    _node(g, 12.5 * cm, fy, 2.9 * cm, rowh, "Parrot Sphinx",
          "marker_mission_sim", fill=REDZONE, stroke=RED, tcolor=RED)
    _arrow(g, 12.5 * cm, fy + rowh / 2, 11.6 * cm, fy + rowh / 2, RED, 1.0, 5,
           dash=[2, 2])
    _text(g, 13.95 * cm, fy - 0.35 * cm, "drop-in FCs", 6.3, RED,
          "Helvetica-Oblique")
    d.add(g)
    return d


# ---------------------------------------------------------------------------
# Figure 2 — Localisation pipeline
# ---------------------------------------------------------------------------

def fig_localisation():
    W, H = 16.4 * cm, 6.2 * cm
    d = Drawing(W, H)
    d.hAlign = "CENTER"
    d.add(Rect(0, 0, W, H, fillColor=MIST, strokeColor=RULE, strokeWidth=0.6))
    _band(d)
    g = Group()
    midy = 3.7 * cm
    bw, bh = 2.45 * cm, 1.25 * cm
    xs = [0.35, 2.95, 5.55]
    labels = [("Camera", "~25 fps frame"),
              ("ArUco detect", "DICT_4X4_50"),
              ("IPPE_SQUARE", "planar PnP")]
    for (lab, sub), xc in zip(labels, xs):
        _node(g, xc * cm, midy, bw, bh, lab, sub, fill=SKY, stroke=ACCENT)
    _arrow(g, 2.80 * cm, midy + bh / 2, 2.95 * cm, midy + bh / 2, NAVY, 1.0, 5)
    _arrow(g, 5.40 * cm, midy + bh / 2, 5.55 * cm, midy + bh / 2, NAVY, 1.0, 5)

    # Two-branch fork
    forkx = 8.0 * cm
    _arrow(g, 8.0 * cm, midy + bh / 2, 8.55 * cm, midy + bh + 0.35 * cm,
           NAVY, 1.0, 5)
    _arrow(g, 8.0 * cm, midy + bh / 2, 8.55 * cm, midy - 0.35 * cm, NAVY, 1.0, 5)
    _node(g, 8.6 * cm, midy + bh - 0.05 * cm, 2.0 * cm, 0.7 * cm,
          "branch A", fill=WHITE, stroke=MUTED, tcolor=INK, tsize=7.5)
    _node(g, 8.6 * cm, midy - 0.75 * cm, 2.0 * cm, 0.7 * cm,
          "branch B (mirror)", fill=WHITE, stroke=MUTED, tcolor=INK, tsize=6.6)

    # Disambiguation priors
    _node(g, 11.0 * cm, midy + 0.55 * cm, 2.55 * cm, 1.4 * cm,
          "Disambiguate", "magnetometer + altimeter priors",
          fill=BLUEZONE, stroke=ACCENT, ssize=6.3)
    _arrow(g, 10.6 * cm, midy + bh, 11.0 * cm, midy + 1.1 * cm, ACCENT, 1.0, 5)
    _arrow(g, 10.6 * cm, midy - 0.4 * cm, 11.0 * cm, midy + 0.9 * cm, ACCENT,
           1.0, 5)

    # Fuse + Kalman (lower row)
    lowy = 1.0 * cm
    _node(g, 11.0 * cm, lowy, 2.55 * cm, 1.25 * cm, "Inverse-dist. fuse",
          "+ anchor (multi-marker)", fill=SKY, stroke=ACCENT, ssize=6.3)
    _arrow(g, 12.27 * cm, midy + 0.55 * cm, 12.27 * cm, lowy + 1.25 * cm,
           NAVY, 1.0, 5)
    _node(g, 13.8 * cm, lowy, 2.3 * cm, 1.25 * cm, "Kalman filter",
          "6-state CV + IMU vel.", fill=NAVY, stroke=NAVY, tcolor=WHITE,
          ssize=6.3)
    _arrow(g, 13.55 * cm, lowy + bh / 2 - 1, 13.8 * cm, lowy + bh / 2 - 1,
           NAVY, 1.0, 5)
    _text(g, 14.95 * cm, lowy - 0.32 * cm, "arena fix  (px, py, pz, v)",
          6.6, NAVY, "Helvetica-Bold")

    # IMU velocity feed-in
    _text(g, 13.0 * cm, midy + 2.25 * cm,
          "Anafi vgx/vgy/vgz  →  arena frame (magnetic-north rotation)",
          6.6, MUTED, "Helvetica-Oblique")
    _arrow(g, 14.9 * cm, midy + 2.05 * cm, 14.9 * cm, lowy + 1.25 * cm,
           MUTED, 0.9, 5, dash=[2, 2])
    d.add(g)
    return d


# ---------------------------------------------------------------------------
# Arena minimap helper (top-down, Y = length vertical, X = width horizontal)
# ---------------------------------------------------------------------------

def _arena(g, ox, oy, w, h):
    """Draw the 20 m × 10 m arena top-down into [ox,oy,w,h]; return mappers."""
    def mx(X):  # arena X in [-5,5] -> drawing x
        return ox + (X + 5.0) / 10.0 * w
    def my(Y):  # arena Y in [-10,10] -> drawing y
        return oy + (Y + 10.0) / 20.0 * h
    # zones
    g.add(Rect(ox, my(-10), w, my(-5) - my(-10), fillColor=REDZONE,
               strokeColor=None))
    g.add(Rect(ox, my(5), w, my(10) - my(5), fillColor=BLUEZONE,
               strokeColor=None))
    g.add(Rect(ox, oy, w, h, fillColor=None, strokeColor=NAVY, strokeWidth=1.0))
    for Y in (-5, 5):
        ln = Line(ox, my(Y), ox + w, my(Y), strokeColor=MUTED, strokeWidth=0.6)
        ln.strokeDashArray = [3, 2]
        g.add(ln)
    # back-wall home markers (13 red / 9 blue)
    g.add(Rect(mx(0) - 4, my(-10) - 0, 8, 4, fillColor=RED, strokeColor=RED))
    _text(g, mx(0) + 14, my(-10) + 1, "13", 6, RED, "Helvetica-Bold")
    g.add(Rect(mx(0) - 4, my(10) - 4, 8, 4, fillColor=ACCENT, strokeColor=ACCENT))
    _text(g, mx(0) + 14, my(10) - 6, "9", 6, ACCENT, "Helvetica-Bold")
    return mx, my


def _boxglyph(g, x, y, color, label):
    g.add(Rect(x - 5, y - 5, 10, 10, fillColor=WHITE, strokeColor=color,
               strokeWidth=1.3))
    _text(g, x, y - 2.5, label, 6.5, color, "Helvetica-Bold")


# Slot layout (Figure-1 triangle), matching default_target_layout.json
SLOTS = {1: (-3, -6.5), 2: (0, -9), 3: (3, -6.5),
         4: (-3, 6.5), 5: (0, 9), 6: (3, 6.5)}


# ---------------------------------------------------------------------------
# Figure 3 — Attack trajectory (vision-relative)
# ---------------------------------------------------------------------------

def fig_attack():
    W, H = 16.4 * cm, 8.0 * cm
    d = Drawing(W, H)
    d.hAlign = "CENTER"
    d.add(Rect(0, 0, W, H, fillColor=MIST, strokeColor=RULE, strokeWidth=0.6))
    _band(d)
    g = Group()
    ax, ay, aw, ah = 0.7 * cm, 0.7 * cm, 5.6 * cm, 6.4 * cm
    mx, my = _arena(g, ax, ay, aw, ah)
    _text(g, ax + aw / 2, ay + ah + 0.18 * cm, "RED home", 6.5, RED,
          "Helvetica-Bold")
    _text(g, ax + aw / 2, ay - 0.30 * cm, "(20 m × 10 m, top-down)", 6, MUTED,
          "Helvetica-Oblique")
    for sl, (X, Y) in SLOTS.items():
        col = RED if sl <= 3 else ACCENT
        _boxglyph(g, mx(X), my(Y), col, str(sl))
    # path: home (0,-7) -> approach box5 (0,9) -> over -> back to wall 13
    start = (mx(-1.5), my(-6.0))
    appr = (mx(0), my(7.4))      # APPROACH standoff ~1 m below box 5
    over = (mx(0), my(9))        # over the box (capture)
    g.add(Circle(start[0], start[1], 5, fillColor=RED, strokeColor=NAVY))
    _arrow(g, start[0], start[1], appr[0], appr[1], NAVY, 1.4, 7)
    _arrow(g, appr[0], appr[1], over[0], over[1], ACCENT, 1.4, 6)
    _star(g, over[0], over[1], 7, RED)
    back = (mx(0), my(-6.5))
    rp = PolyLine([over[0], over[1], mx(2.2), my(2), back[0], back[1]],
                  strokeColor=MUTED, strokeWidth=1.1)
    rp.strokeDashArray = [4, 3]
    g.add(rp)
    _arrow(g, mx(1.6), my(-3), back[0], back[1], MUTED, 1.1, 6, dash=[4, 3])
    _drone(g, appr[0] - 0.0, appr[1], 6, NAVY, ACCENT)

    # Right: the actual emitted script + legend
    sx = 7.0 * cm
    _text(g, sx, ay + ah + 0.18 * cm,
          "Emitted mission script  (red attacks slot 5, face 35)",
          7.5, NAVY, "Helvetica-Bold", "start")
    script = [
        ("TAKEOFF", ""),
        ("YAW 0", "face the enemy half (absolute)"),
        ("HEIGHT 2.20", "climb above the 0.73 m boxes"),
        ("APPROACH 35 1.00", "vision-home to 1 m off the box"),
        ("HEIGHT 1.50", "rise into the 1–2 m capture band"),
        ("FB_IMU 0.90", "bounded step over the box centre"),
        ("HOOVER 3.5", "dwell ≥2 s → capture flips 35→45"),
        ("HEIGHT 2.20", "climb to clear the boxes"),
        ("YAW 180", "turn toward home"),
        ("APPROACH 13 3.50", "vision-home to the home wall"),
        ("YAW 0", "re-face enemy, ready to re-arm"),
    ]
    ly = ay + ah - 0.25 * cm
    for code, note in script:
        g.add(String(sx, ly, code, fontName="Courier-Bold", fontSize=8,
                     fillColor=NAVY))
        if note:
            g.add(String(sx + 3.05 * cm, ly, "# " + note, fontName="Courier",
                         fontSize=7, fillColor=MUTED))
        ly -= 0.46 * cm
    # legend
    ly -= 0.15 * cm
    g.add(Line(sx, ly, sx + 0.5 * cm, ly, strokeColor=NAVY, strokeWidth=1.4))
    _text(g, sx + 0.65 * cm, ly - 2.5, "bounded IMU / cruise", 7, INK,
          "Helvetica", "start")
    g.add(Line(sx + 4.3 * cm, ly, sx + 4.8 * cm, ly, strokeColor=ACCENT,
               strokeWidth=1.4))
    _text(g, sx + 4.95 * cm, ly - 2.5, "vision APPROACH", 7, INK, "Helvetica",
          "start")
    ly -= 0.40 * cm
    lr = Line(sx, ly, sx + 0.5 * cm, ly, strokeColor=MUTED, strokeWidth=1.1)
    lr.strokeDashArray = [4, 3]
    g.add(lr)
    _text(g, sx + 0.65 * cm, ly - 2.5, "return home (vision)", 7, INK,
          "Helvetica", "start")
    _star(g, sx + 4.55 * cm, ly + 1, 5, RED)
    _text(g, sx + 4.95 * cm, ly - 2.5, "capture (2 s hover)", 7, INK,
          "Helvetica", "start")
    d.add(g)
    return d


# ---------------------------------------------------------------------------
# Figure 4 — Swarm roles + planner decision ladder
# ---------------------------------------------------------------------------

def fig_swarm():
    W, H = 16.4 * cm, 8.2 * cm
    d = Drawing(W, H)
    d.hAlign = "CENTER"
    d.add(Rect(0, 0, W, H, fillColor=MIST, strokeColor=RULE, strokeWidth=0.6))
    _band(d)
    g = Group()
    # Left: arena with roles
    ax, ay, aw, ah = 0.7 * cm, 0.7 * cm, 5.6 * cm, 6.6 * cm
    mx, my = _arena(g, ax, ay, aw, ah)
    _text(g, ax + aw / 2, ay + ah + 0.18 * cm, "Roles in play  (red team)",
          7.5, NAVY, "Helvetica-Bold")
    for sl, (X, Y) in SLOTS.items():
        col = RED if sl <= 3 else ACCENT
        _boxglyph(g, mx(X), my(Y), col, str(sl))
    # scout spinning at centre
    _drone(g, mx(0), my(0), 6, NAVY, ACCENT)
    g.add(Circle(mx(0), my(0), 12, fillColor=None, strokeColor=ACCENT,
                 strokeWidth=0.8, strokeDashArray=[2, 2]))
    _text(g, mx(0), my(0) - 0.62 * cm, "scout", 6.3, NAVY, "Helvetica-Bold")
    # defender hovering in neutral facing back wall
    _drone(g, mx(-2.0), my(-4.2), 6, RED, WHITE)
    _text(g, mx(-2.0), my(-4.2) - 0.6 * cm, "defender", 6.3, RED,
          "Helvetica-Bold")
    # attackers heading to blue boxes
    for (sx_, sy_), (tx, ty) in (((1.8, -4.0), (3, 6.5)), ((3.2, -2.0), (0, 9))):
        _drone(g, mx(sx_), my(sy_), 5, NAVY, ACCENT)
        _arrow(g, mx(sx_), my(sy_) + 0.2 * cm, mx(tx), my(ty) - 0.25 * cm,
               NAVY, 1.0, 5, dash=[3, 2])
    _text(g, mx(2.6), my(-3.0) - 0.2 * cm, "attackers", 6.3, NAVY,
          "Helvetica-Bold")

    # Right: decision ladder
    lx, lw = 7.0 * cm, 8.9 * cm
    _text(g, lx, ay + ah + 0.18 * cm,
          "Per-tick planner — first applicable rule wins",
          7.5, NAVY, "Helvetica-Bold", "start")
    rungs = [
        ("0  ·  Guard", "drop disabled / low-battery / lost-link to RETURN-IDLE",
         MIST, MUTED),
        ("1  ·  Defend", "own box enemy-held & unlocked → recapture NOW "
         "(nearest free drone)", REDZONE, RED),
        ("2  ·  5-pt full sortie", "≥2 drones can each cover an enemy box → "
         "coordinated all-out (every drone outside home at trigger)",
         BLUEZONE, ACCENT),
        ("3  ·  Attack", "greedy nearest-slot over the attackable enemy set "
         "(red→4-6, blue→1-3)", SKY, ACCENT),
        ("4  ·  Bank", "secure a 5-pt attempt: all drones home before the "
         "enemy can re-flip", MIST, NAVY),
    ]
    ry = ay + ah - 0.35 * cm
    rh = 1.06 * cm
    for title, body, fill, stroke in rungs:
        g.add(Rect(lx, ry - rh, lw, rh, fillColor=fill, strokeColor=stroke,
                   strokeWidth=1.0, rx=3, ry=3))
        g.add(String(lx + 0.18 * cm, ry - 0.34 * cm, title,
                     fontName="Helvetica-Bold", fontSize=8.5, fillColor=NAVY))
        # wrap body to width
        words, line, lines = body.split(), "", []
        for wd in words:
            if len(line) + len(wd) + 1 > 64:
                lines.append(line); line = wd
            else:
                line = (line + " " + wd).strip()
        if line:
            lines.append(line)
        yy = ry - 0.60 * cm
        for li in lines[:2]:
            g.add(String(lx + 0.18 * cm, yy, li, fontName="Helvetica",
                         fontSize=7, fillColor=INK))
            yy -= 0.30 * cm
        if ry - rh > ay:
            _arrow(g, lx + lw / 2, ry - rh, lx + lw / 2, ry - rh - 0.16 * cm,
                   MUTED, 0.8, 4)
        ry -= rh + 0.22 * cm
    d.add(g)
    return d


# ---------------------------------------------------------------------------
# Footer (Page X of N), cover clean
# ---------------------------------------------------------------------------

class FooterCanvas(canvas_mod.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved = []

    def showPage(self):
        self._saved.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        n = len(self._saved)
        for i, state in enumerate(self._saved, start=1):
            self.__dict__.update(state)
            self._stamp(i, n)
            super().showPage()
        super().save()

    def _stamp(self, i: int, n: int) -> None:
        if i == 1:
            return
        self.setStrokeColor(RULE)
        self.setLineWidth(0.4)
        self.line(2 * cm, 1.55 * cm, A4[0] - 2 * cm, 1.55 * cm)
        self.setFillColor(MUTED)
        self.setFont("Times-Italic", 8.5)
        self.drawString(2 * cm, 1.05 * cm,
                        "SDC26 — Team ToBeDefined — Technical Report")
        self.drawRightString(A4[0] - 2 * cm, 1.05 * cm, f"Page {i} of {n}")


# ---------------------------------------------------------------------------
# Document content
# ---------------------------------------------------------------------------

def cover() -> list:
    story = [Spacer(1, 1.5 * cm)]
    img = Image(str(LOGO), width=6.0 * cm, height=6.0 * cm)
    img.hAlign = "CENTER"
    story += [img, Spacer(1, 0.7 * cm)]
    story += [
        P("SDC26", "Tag"),
        P("Technical Report", "TitleBig"),
        P("Team ToBeDefined", "Sub"),
        Spacer(1, 0.25 * cm),
        P("Swarm Drone Challenge 2026 — ILA Berlin, 11 June 2026", "Sub"),
        Spacer(1, 1.3 * cm),
        P("Abstract", "AbstractTitle"),
        P(
            "This report describes the technical solution developed by Team "
            "ToBeDefined for the Swarm Drone Challenge 2026. Each Parrot Anafi "
            "flies under an on-board mission engine driven by ArUco-only "
            "positioning (no GPS), a domain-specific mission-scripting language, "
            "and a shared command-and-control (C2) overlay that coordinates the "
            "swarm across the 20 m × 10 m arena. The design prioritises "
            "drift-free, vision-relative homing over absolute world-frame "
            "steering, uses bounded closed-loop primitives for repeatable "
            "manoeuvres, and exposes a transparent rule-based strategy layer "
            "that translates the game's scoring tiers into per-drone role and "
            "target assignments. The same code runs the Parrot Sphinx simulator "
            "and the real flight controllers, so every change is validated "
            "against identical mission scripts and telemetry contracts before "
            "flying hardware. The stack is roughly 90,000 lines of Python and is "
            "open-sourced in line with the competition's encouragement of "
            "shared code."),
    ]
    return story


def section_intro_and_architecture() -> list:
    return [
        PageBreak(),
        P("1.  Introduction", "H1"),
        P(
            "The Swarm Drone Challenge 2026 (SDC26) pits two teams of up to "
            "five Parrot Anafi drones against each other in a 20 m × 10 m "
            "indoor arena. Each side defends three target boxes in its home "
            "zone and attempts to flip the three enemy boxes by hovering over "
            "them, within a 1–2 m capture band, for at least two seconds. The "
            "boxes carry DICT_4X4_50 ArUco markers whose encoding — leading "
            "digit 4 for red, 3 for blue, indicates the owner; trailing digit "
            "is the box ID 1–6 — flips on capture. Points accrue per attempt at "
            "three tiers (1, 5, or 10), with an instant-win Special Manoeuvre "
            "for controlling all six boxes for five continuous seconds. A match "
            "lasts ten minutes; GPS is unavailable in the indoor venue, so all "
            "positioning is derived from on-board cameras and inertial sensors."),
        P(
            "Team ToBeDefined attacks the problem with a single, uniform "
            "software stack that runs both on each drone's flight controller "
            "(an x86 Linux companion computer flying the Anafi via Parrot "
            "Olympe) and on a lightweight central C2 overlay. The same code "
            "base powers the Parrot Sphinx simulator, which stands in for the "
            "real flight controllers during development; a single operator "
            "switch (by IP) re-points any drone from the simulator to a real "
            "controller, and a second switch hot-swaps the whole stack between "
            "the competition arena and a smaller testing arena — live, with no "
            "restart."),
        P("2.  System Architecture", "H1"),
        P(
            "Three process types form the runtime. <b>marker_mission</b> "
            "(port 8080) is the per-drone flight controller: it owns the "
            "Olympe connection, runs the ArUco vision loop, executes the "
            "mission-scripting language, and exposes a small HTTP API for "
            "telemetry, RC, and mission control. <b>marker_mission_c2</b> "
            "(port 8090) is the fleet layer: it polls every flight controller, "
            "proxies operator commands, and serves the live state UI. The "
            "<b>strategy</b> servers (ports 8091 red / 8092 blue) sit on top of "
            "the C2 and own slot tracking, the per-drone role assignment, and "
            "the rule-based planner of Section 5. All inter-process traffic is "
            "HTTP/JSON, and the whole fleet is joined into a single Tailscale "
            "tailnet so the operator UIs are reachable by MagicDNS name while "
            "remaining cleanly air-gapped from the public internet."),
        fig_architecture(),
        CAP("Figure 1 — Runtime architecture. Each caged Anafi is flown by its "
            "own marker_mission flight controller; the C2 aggregates the fleet "
            "and the per-team strategy servers issue role and target "
            "assignments. The Sphinx simulator is a drop-in replacement for the "
            "real flight controllers."),
        P(
            "Mission behaviour is expressed in a small text-based scripting "
            "language — the <i>mission DSL</i> — parsed and executed by the "
            "flight controller. Its primitives cover discrete IMU closed-loop "
            "moves (<font face='Courier'>FB_IMU</font>, "
            "<font face='Courier'>LR_IMU</font>, "
            "<font face='Courier'>UD_IMU</font>, "
            "<font face='Courier'>YAW_IMU</font>), absolute heading holds "
            "(<font face='Courier'>YAW</font>), open-loop stick windows "
            "(<font face='Courier'>FB_RC</font>, "
            "<font face='Courier'>UD_RC</font>), vision homing "
            "(<font face='Courier'>APPROACH</font>, "
            "<font face='Courier'>FB_BRAKE</font>), holds "
            "(<font face='Courier'>HOOVER</font>), and coarse world-frame "
            "pre-positioning (<font face='Courier'>TO</font>). Each step "
            "compiles to a state-machine phase that drives a closed-loop RC "
            "stream at the configured control rate (10 Hz), gives vision its "
            "own thread, and writes a structured per-tick log "
            "consumed by an offline replay viewer for tuning."),
    ]


def section_localisation() -> list:
    return [
        PageBreak(),
        P("3.  Localisation", "H1"),
        P(
            "The arena provides eight pillars carrying sixteen ArUco markers — "
            "an upper marker at 4 m and a lower marker at 2 m per pillar; IDs "
            "1–8 upper, 9–16 lower, 0.5 × 0.5 m, DICT_4X4_50. Each flight "
            "controller's vision loop runs OpenCV ArUco detection on the "
            "on-board camera and computes a per-marker pose with the "
            "IPPE_SQUARE planar-pose algorithm. Planar PnP returns two "
            "equally-valid solutions related by a reflection about the marker "
            "plane; the wrong branch places the drone on the far side of the "
            "marker, so resolving it correctly is the crux of the pipeline "
            "(Figure 2). We disambiguate with two independent priors."),
        fig_localisation(),
        CAP("Figure 2 — Localisation pipeline. The IPPE planar-pose ambiguity "
            "is resolved by a magnetometer prior (works on the ground too) and "
            "an altimeter prior; multi-marker fixes are inverse-distance fused "
            "around a remembered anchor, then smoothed by a six-state "
            "constant-velocity Kalman filter that also ingests Anafi body "
            "velocity."),
        P("3.1  Magnetometer and altimeter priors", "H2"),
        P(
            "The arena's magnetic-north heading is calibrated once and stored "
            "on the active arena configuration. The two IPPE branches imply two "
            "candidate drone headings; the branch whose heading agrees with the "
            "live magnetometer reading within a configurable slack (default "
            "12°) is accepted. Because this test needs no altitude, it gives a "
            "clean fix even on the ground at take-off. When the drone is "
            "airborne, the Anafi's downward ultrasound altimeter provides a "
            "second, independent check on the vertical component of each "
            "branch. We require agreement from at least one prior; if only one "
            "marker is visible and the magnetometer is uncalibrated, the fix is "
            "rejected and the previous estimate is allowed to decay rather than "
            "jump to a mirrored pose."),
        P("3.2  Kalman filter and multi-marker anchor", "H2"),
        P(
            "Raw per-tick fixes are fused with Anafi-reported body velocities "
            "in a six-state constant-velocity Kalman filter whose state is "
            "<font face='Courier'>[px, py, pz, vx, vy, vz]</font> in metres and "
            "metres per second. Two heterogeneous updates feed it: an "
            "intermittent, accurate position update from the aggregated ArUco "
            "fix, and a fast, every-tick velocity update from the Anafi "
            "<font face='Courier'>vgx/vgy/vgz</font> telemetry rotated into the "
            "arena frame using the calibrated magnetic-north offset. The filter "
            "both smooths measurement jitter and supplies the velocity estimate "
            "the controller uses to damp lateral oscillation during APPROACH. "
            "When several reference markers are visible at once, their world "
            "positions are weighted by inverse distance and fused into one "
            "arena fix; the dominant contributor is remembered as the "
            "<i>anchor</i> across frames so a noisy single-marker reading "
            "cannot drag the estimate to the opposite side of the field."),
        P("3.3  No GPS, and a hard-won failure mode", "H2"),
        P(
            "No GPS information is used anywhere in the live positioning or "
            "control path, as required by §4.4 of the regulations; GPS appears "
            "only as an off-line ground-truth reference inside the simulator "
            "during tuning. In metal-rich indoor spaces the magnetometer can "
            "still be unreliable, and we learned the hard way that an active "
            "reverse-brake wall guard acting on a mirrored single-marker fix "
            "will drive the drone <i>into</i> the wall it appears to be near, at "
            "saturated stick. The mitigation, now the standing design rule "
            "across the stack, is to never steer open-loop on an absolute fix: "
            "precise motion is always either a bounded IMU step or a "
            "vision-relative APPROACH onto a marker the camera can actually see "
            "(Section 4), which is correct regardless of which IPPE branch the "
            "world-frame estimator chose."),
    ]


def section_flight_control() -> list:
    return [
        PageBreak(),
        P("4.  Flight Control and Trajectory Planning", "H1"),
        P(
            "The Anafi exposes both a piloting (PCMD) interface for continuous "
            "roll/pitch/yaw/throttle commands and a discrete move-by interface "
            "that closes a position loop on the drone's own IMU and optical "
            "flow. Our mission engine treats these as complementary primitive "
            "families. <b>IMU primitives</b> "
            "(<font face='Courier'>FB_IMU</font>, "
            "<font face='Courier'>LR_IMU</font>, "
            "<font face='Courier'>UD_IMU</font>, "
            "<font face='Courier'>YAW_IMU</font>) execute an exact bounded "
            "displacement and stop. <b>RC primitives</b> "
            "(<font face='Courier'>FB_RC</font>, "
            "<font face='Courier'>UD_RC</font>) pin a stick value for a "
            "duration and then auto-brake. <b>Vision primitives</b> "
            "(<font face='Courier'>APPROACH</font>, "
            "<font face='Courier'>FB_BRAKE</font>) home on a named ArUco "
            "marker's measured bearing and distance and are drift-free "
            "regardless of the world-frame fix. An absolute "
            "<font face='Courier'>YAW</font> primitive holds a fixed arena "
            "heading so a manoeuvre never inherits a stale heading from the "
            "previous leg."),
        P("4.1  The canonical attack manoeuvre", "H2"),
        P(
            "Every attack run is generated programmatically by the strategy "
            "server for the selected target slot and composes to the pattern in "
            "Figure 3. The attacker climbs above the 0.73 m boxes, "
            "vision-homes onto the enemy face to a 1 m stand-off, rises into "
            "the 1–2 m capture band, takes one bounded "
            "<font face='Courier'>FB_IMU</font> step sized to the stand-off so "
            "it stops over the box centre, and hovers for at least two seconds "
            "to flip the marker. It then climbs, turns, and vision-homes onto "
            "its own back-wall marker to return — never an open-loop cruise "
            "back across the field."),
        fig_attack(),
        CAP("Figure 3 — The vision-relative attack manoeuvre and the exact "
            "mission script the strategy server emits. Solid navy = bounded "
            "IMU / cruise; blue = vision APPROACH; dashed = vision-homed "
            "return; star = the 2-second capture hover."),
        P(
            "This composition reflects three hard-won lessons. First, every "
            "horizontal displacement is either a bounded IMU step or a "
            "vision-homed APPROACH — never an open-loop RC cruise of "
            "non-trivial distance — because forward pitch tilts the camera "
            "down and hides the destination marker, so a vision-tripped brake "
            "never fires and the drone reaches its timeout at full stick. "
            "Second, the over-the-box translation is a bounded "
            "<font face='Courier'>FB_IMU</font> sized to the APPROACH "
            "stand-off, so even with a noisy fix the drone stops above the "
            "target rather than continuing into the wall behind it. Third, the "
            "turns use the IMU-closed-loop heading holds "
            "(<font face='Courier'>YAW</font> / "
            "<font face='Courier'>YAW_IMU</font>) rather than a yaw-rate stick, "
            "which rolled off in timing tests and sent the next leg sideways. "
            "In the home zone the drone always flies higher than the 0.73 m "
            "open boxes and only descends into the capture band directly over a "
            "target."),
        P("4.2  Safety layers", "H2"),
        P(
            "Several overlays guard the runtime regardless of which script is "
            "loaded. A hard altitude ceiling clamps the throttle channel above "
            "a configurable height. A C2 watchdog forces an automatic land if "
            "the operator heartbeat goes silent for longer than the "
            "regulation's two-second threshold (§3.3). A master kill-all lands "
            "the entire fleet from a single key-press and is the responsibility "
            "of the team's Safety Officer (§2.2). Each drone in flight is "
            "shrouded in a protective cage as recommended in §3.2. The "
            "simulator enforces the same arena bounds, so a script that would "
            "drive into a wall is caught in simulation before it ever reaches "
            "hardware."),
    ]


def section_swarm() -> list:
    return [
        PageBreak(),
        P("5.  Swarm Strategy", "H1"),
        P(
            "Coordination is handled by a small, deterministic <b>planner</b> "
            "that runs each tick on the strategy server. It is deliberately "
            "rule-based rather than learned: every tick it reads each drone's "
            "status (connection, battery, magnetometer calibration, mission "
            "phase, position) and the live slot map (the holder of each of the "
            "six boxes and how recently it was observed), then emits a "
            "(role, target) assignment per drone. The planner is a strict "
            "priority ladder — the first applicable rule wins — which makes its "
            "behaviour transparent and reproducible, a property we value "
            "precisely because the code is open for inspection (Figure 4)."),
        fig_swarm(),
        CAP("Figure 4 — Left: a typical role mix — a scout refreshing the slot "
            "map from the centre, two attackers vectored at enemy boxes, and a "
            "defender holding in the neutral zone. Right: the per-tick planner "
            "ladder; the first applicable rule wins."),
        P(
            "The ladder encodes the regulations' scoring tiers directly. "
            "Recapturing an enemy-held box inside our own home zone scores zero "
            "under the anti-farming rule of §1.4.3, so the planner treats a "
            "home recapture as defence, not points. Rule 1 (defend) is "
            "top-priority and immediate: the moment one of our boxes shows the "
            "enemy colour, the nearest free drone breaks off to re-flip it. "
            "Rule 2 is the five-point full sortie — when at least two drones "
            "can each cover a different enemy box, the planner schedules a "
            "coordinated all-out so that every team drone is outside the home "
            "zone at the moment of capture, which the regulations reward with "
            "the coordination bonus; the team then BANKs (Rule 4), returning "
            "home together before the enemy can re-flip, to secure the attempt. "
            "Rule 3 is the fallback: a greedy nearest-slot assignment over the "
            "attackable enemy set. The Special Manoeuvre — all six boxes for "
            "five seconds — is treated as a terminal objective and "
            "short-circuits selection when within reach."),
        P("5.1  Roles, dedup, and asymmetric attack", "H2"),
        P(
            "The planner assigns one of four roles per drone: <b>attacker</b> "
            "(drive a capture or recapture), <b>scout</b> (rotate in place at "
            "the arena centre to refresh stale slots so the map stays current), "
            "<b>defender</b> (protect our home boxes), and <b>idle/return</b> "
            "(low battery, lost link, or no useful work). Assignments are "
            "de-duplicated through three sets maintained each tick — slots "
            "already <i>taken</i> by a plan this tick, slots an in-flight drone "
            "is already <i>targeting</i>, and slots inside the five-second "
            "post-capture <i>lock</i> — so two drones never converge on the "
            "same box. The attack set is asymmetric per team: red drones may "
            "only target boxes 4–6 (the blue zone) and blue drones boxes 1–3, "
            "and the planner rejects any operator override that violates the "
            "constraint."),
        P("5.2  The defender", "H2"),
        P(
            "The defender does not camp over its boxes (which would risk the "
            "§1.3 dead-drone-over-target rule and forfeit the coordination "
            "bonus). Instead it hovers at a fixed station just inside the "
            "neutral zone, facing its own back wall and holding position by a "
            "vision APPROACH on the back-wall marker — no absolute world "
            "steering. Because the slot map is shared, it detects a lost box "
            "itself the instant the marker flips and dashes in to recapture, "
            "then returns to its station. This keeps it both outside the home "
            "zone (so the team's five-point attempts stay valid) and as close "
            "as legally possible to the boxes it protects."),
        P("5.3  Open source and reproducibility", "H2"),
        P(
            "The entire stack is open-sourced at "
            "<font face='Courier'>github.com/otiedemann/sdc-tobe</font> — "
            "roughly 90,000 lines of Python spanning the flight controller, the "
            "C2, the strategy layer, and the simulator. The flight controller, "
            "C2, and strategy share a single configuration and virtual "
            "environment; the mission DSL ships with a test harness; and every "
            "flight is recorded into a per-tick log plus an annotated video "
            "that the in-browser replay viewer plays back synchronously, so "
            "any result in this report can be reproduced from the repository."),
        P("6.  Conclusion", "H1"),
        P(
            "One design philosophy runs through every layer of the stack: "
            "prefer drift-free, vision-relative homing and bounded closed-loop "
            "motion over absolute world-frame steering and open-loop cruises. "
            "That choice is forced by the realities of ArUco-only indoor "
            "positioning, and it is what lets the swarm fly safely and "
            "repeatably. The remaining work — a position-sanity gate that lets "
            "the active wall guard be re-enabled once a fix is trusted, and "
            "further tuning of the coordinated five-point sortie — is scoped "
            "and queued, and will be delivered for the final at ILA Berlin on "
            "11 June 2026."),
    ]


def main():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2.0 * cm,
        title="SDC26 Technical Report — Team ToBeDefined",
        author="Team ToBeDefined",
        subject="Swarm Drone Challenge 2026 — Technical Report")
    story = []
    story += cover()
    story += section_intro_and_architecture()
    story += section_localisation()
    story += section_flight_control()
    story += section_swarm()
    doc.build(story, canvasmaker=FooterCanvas)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
