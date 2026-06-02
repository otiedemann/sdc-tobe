#!/usr/bin/env python3
"""Build the SDC26 ToBeDefined Technical Report as a branded A4 PDF.

Pure ``reportlab`` (no LaTeX, no external HTML→PDF tools). Embeds the team
logo, follows the three required sections from §1.6.3 of the regulations
(localisation; flight control & trajectory planning; swarm strategy), and
draws all figures natively as vector graphics in the team's corporate-identity
palette. Output:

    1_Doc/SDC26_ToBeDefined_Technical_Report.pdf

The writing is deliberately plain and scannable — short sentences, bullet
lists, and a one-line "Key idea" per section — while staying faithful to the
open-sourced code base (github.com/otiedemann/sdc-tobe). Rebuild with:

    .venv/bin/python tools/build_tech_report.py
"""
from __future__ import annotations

import math
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as canvas_mod
from reportlab.graphics.shapes import (
    Drawing, Rect, Line, Circle, Ellipse, String, Polygon, PolyLine, Group,
)
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
)


def FIG(drawing, caption):
    """Keep a figure and its caption on the same page."""
    return KeepTogether([drawing, caption])


def HEAD(*flowables):
    """Keep a heading with the block that follows it (no orphan headings)."""
    return KeepTogether(list(flowables))

ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "1_Doc" / "team_logo_pdf.png"
OUT = ROOT / "1_Doc" / "SDC26_ToBeDefined_Technical_Report.pdf"

# ---------------------------------------------------------------------------
# Corporate-identity palette (matches the team mission-patch logo)
# ---------------------------------------------------------------------------
NAVY = HexColor("#10243d")
ACCENT = HexColor("#3b82c4")
RED = HexColor("#d6443a")
MUTED = HexColor("#6b7280")
RULE = HexColor("#cbd5e1")
SHADOW = HexColor("#c4cdd8")
CODE_BG = HexColor("#f3f4f6")
SKY = HexColor("#eaf1f8")
MIST = HexColor("#f6f8fb")
GRIDC = HexColor("#e4eaf1")
REDZONE = HexColor("#fbe9e7")
BLUEZONE = HexColor("#e7f0fb")
WHITE = HexColor("#ffffff")
INK = HexColor("#243140")


def make_styles():
    s = {}
    s["TitleBig"] = ParagraphStyle(
        "TitleBig", fontName="Times-Bold", fontSize=30, leading=34,
        alignment=TA_CENTER, textColor=NAVY, spaceAfter=8)
    s["Sub"] = ParagraphStyle(
        "Sub", fontName="Times-Roman", fontSize=13, leading=17,
        alignment=TA_CENTER, textColor=MUTED, spaceAfter=4)
    s["Tag"] = ParagraphStyle(
        "Tag", fontName="Helvetica-Bold", fontSize=10, leading=12,
        alignment=TA_CENTER, textColor=ACCENT, spaceAfter=2)
    s["AbstractTitle"] = ParagraphStyle(
        "AbstractTitle", fontName="Helvetica-Bold", fontSize=10, leading=12,
        alignment=TA_LEFT, textColor=NAVY, spaceAfter=4)
    s["H1"] = ParagraphStyle(
        "H1", fontName="Helvetica-Bold", fontSize=15, leading=19,
        textColor=NAVY, spaceBefore=10, spaceAfter=6)
    s["H2"] = ParagraphStyle(
        "H2", fontName="Helvetica-Bold", fontSize=11, leading=14,
        textColor=ACCENT, spaceBefore=8, spaceAfter=2)
    s["Body"] = ParagraphStyle(
        "Body", fontName="Times-Roman", fontSize=10.5, leading=15,
        alignment=TA_LEFT, textColor=INK, spaceAfter=6)
    s["Bullet"] = ParagraphStyle(
        "Bullet", fontName="Times-Roman", fontSize=10.5, leading=14.5,
        alignment=TA_LEFT, textColor=INK, leftIndent=16, bulletIndent=3,
        bulletFontName="Helvetica-Bold", bulletFontSize=9, bulletColor=ACCENT,
        spaceAfter=3)
    s["Callout"] = ParagraphStyle(
        "Callout", fontName="Helvetica", fontSize=9.8, leading=13.5,
        textColor=NAVY, backColor=SKY, borderColor=ACCENT, borderWidth=0.8,
        borderRadius=4, borderPadding=8, spaceBefore=2, spaceAfter=10)
    s["Caption"] = ParagraphStyle(
        "Caption", fontName="Helvetica-Oblique", fontSize=8.5, leading=11,
        alignment=TA_CENTER, textColor=MUTED, spaceBefore=3, spaceAfter=9)
    s["Code"] = ParagraphStyle(
        "Code", fontName="Courier", fontSize=8.5, leading=11, backColor=CODE_BG,
        borderColor=RULE, borderWidth=0.4, borderPadding=6, spaceBefore=2,
        spaceAfter=6)
    return s


S = make_styles()


def P(text, style="Body"):
    return Paragraph(text, S[style])


def BUL(text):
    return Paragraph(text, S["Bullet"], bulletText="•")


def KEY(text):
    return Paragraph(
        "<b><font color='#3b82c4'>Key idea</font></b>&nbsp;&nbsp;" + text,
        S["Callout"])


def CAP(text):
    return Paragraph(text, S["Caption"])


# ---------------------------------------------------------------------------
# Vector-graphics primitives (all figures drawn natively, CI colours)
# ---------------------------------------------------------------------------

def _arrow(g, x1, y1, x2, y2, color=NAVY, w=1.1, head=6.0, dash=None):
    ln = Line(x1, y1, x2, y2, strokeColor=color, strokeWidth=w)
    if dash:
        ln.strokeDashArray = dash
    g.add(ln)
    ang = math.atan2(y2 - y1, x2 - x1)
    a1, a2 = ang + math.radians(150), ang - math.radians(150)
    g.add(Polygon(
        [x2, y2, x2 + head * math.cos(a1), y2 + head * math.sin(a1),
         x2 + head * math.cos(a2), y2 + head * math.sin(a2)],
        fillColor=color, strokeColor=color))


def _text(g, x, y, t, size=8, color=INK, font="Helvetica", anchor="middle"):
    g.add(String(x, y, t, fontName=font, fontSize=size, fillColor=color,
                 textAnchor=anchor))


def _card(g, x, y, w, h, fill, stroke, sw=1.1, stack=0):
    """Rounded card with a soft drop shadow; ``stack`` draws offset ghost
    cards behind it to imply multiplicity (e.g. 'one per drone')."""
    for k in range(stack, 0, -1):
        o = 2.6 * k
        g.add(Rect(x + o, y + o, w, h, fillColor=WHITE, strokeColor=RULE,
                   strokeWidth=0.8, rx=5, ry=5))
    g.add(Rect(x + 1.8, y - 2.0, w, h, fillColor=SHADOW, strokeColor=None,
               rx=5, ry=5))
    g.add(Rect(x, y, w, h, fillColor=fill, strokeColor=stroke, strokeWidth=sw,
               rx=5, ry=5))


def _node(g, x, y, w, h, title, sub=None, fill=SKY, stroke=ACCENT,
          tcolor=NAVY, tsize=8.5, ssize=7, stack=0, sw=1.1):
    _card(g, x, y, w, h, fill, stroke, sw, stack)
    cx = x + w / 2.0
    if sub:
        _text(g, cx, y + h / 2.0 + 2.0, title, tsize, tcolor, "Helvetica-Bold")
        _text(g, cx, y + h / 2.0 - 8.0, sub, ssize, MUTED, "Helvetica")
    else:
        _text(g, cx, y + h / 2.0 - 3, title, tsize, tcolor, "Helvetica-Bold")


def _drone(g, cx, cy, r, color=NAVY, body=ACCENT, ground=False):
    if ground:
        g.add(Ellipse(cx + 1, cy - r - 2, r * 0.85, r * 0.22, fillColor=SHADOW,
                      strokeColor=None))
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        g.add(Line(cx, cy, cx + dx * r, cy + dy * r, strokeColor=color,
                   strokeWidth=1.5))
        g.add(Ellipse(cx + dx * r, cy + dy * r, r * 0.5, r * 0.33,
                      fillColor=WHITE, strokeColor=color, strokeWidth=1.1))
    g.add(Circle(cx, cy, r * 0.56, fillColor=body, strokeColor=color,
                 strokeWidth=1.1))
    g.add(Circle(cx, cy + r * 0.56, r * 0.2, fillColor=NAVY, strokeColor=color,
                 strokeWidth=0.8))  # forward camera nub


def _marker(g, x, y, s, color=NAVY):
    g.add(Rect(x, y, s, s, fillColor=WHITE, strokeColor=color, strokeWidth=0.8))
    c = s / 4.0
    for i, j in ((1, 1), (2, 2), (1, 3), (3, 1), (2, 0)):
        g.add(Rect(x + i * c, y + j * c, c, c, fillColor=color, strokeColor=None))


def _cam(g, cx, cy, color=ACCENT):
    g.add(Rect(cx - 5, cy - 4, 10, 8, fillColor=NAVY, strokeColor=NAVY, rx=1,
               ry=1))
    g.add(Polygon([cx + 5, cy + 3, cx + 13, cy + 7, cx + 13, cy - 7,
                   cx + 5, cy - 3], fillColor=None, strokeColor=color,
                  strokeWidth=0.9))


def _star(g, cx, cy, r, color=RED):
    pts = []
    for i in range(10):
        rad = r if i % 2 == 0 else r * 0.45
        a = math.radians(-90 + i * 36)
        pts += [cx + rad * math.cos(a), cy + rad * math.sin(a)]
    g.add(Polygon(pts, fillColor=color, strokeColor=color))


def _band(d, h=4, color=NAVY):
    d.add(Rect(0, d.height - h, d.width, h, fillColor=color, strokeColor=None))


def _frame(d):
    d.add(Rect(0, 0, d.width, d.height, fillColor=MIST, strokeColor=RULE,
               strokeWidth=0.6, rx=3, ry=3))
    _band(d)


SLOTS = {1: (-3, -6.5), 2: (0, -9), 3: (3, -6.5),
         4: (-3, 6.5), 5: (0, 9), 6: (3, 6.5)}


def _arena(g, ox, oy, w, h, grid=True):
    def mx(X):
        return ox + (X + 5.0) / 10.0 * w

    def my(Y):
        return oy + (Y + 10.0) / 20.0 * h
    g.add(Rect(ox, oy, w, h, fillColor=WHITE, strokeColor=None))
    g.add(Rect(ox, my(-10), w, my(-5) - my(-10), fillColor=REDZONE,
               strokeColor=None))
    g.add(Rect(ox, my(5), w, my(10) - my(5), fillColor=BLUEZONE,
               strokeColor=None))
    if grid:
        step = w / 10.0
        x = ox
        while x <= ox + w + 0.1:
            g.add(Line(x, oy, x, oy + h, strokeColor=GRIDC, strokeWidth=0.4))
            x += step
        y = oy
        while y <= oy + h + 0.1:
            g.add(Line(ox, y, ox + w, y, strokeColor=GRIDC, strokeWidth=0.4))
            y += step
    g.add(Rect(ox, oy, w, h, fillColor=None, strokeColor=NAVY, strokeWidth=1.1))
    for Y in (-5, 5):
        ln = Line(ox, my(Y), ox + w, my(Y), strokeColor=MUTED, strokeWidth=0.6)
        ln.strokeDashArray = [3, 2]
        g.add(ln)
    g.add(Rect(mx(0) - 5, my(-10), 10, 4, fillColor=RED, strokeColor=RED))
    _text(g, mx(0) + 15, my(-10) + 1, "13", 6, RED, "Helvetica-Bold")
    g.add(Rect(mx(0) - 5, my(10) - 4, 10, 4, fillColor=ACCENT, strokeColor=ACCENT))
    _text(g, mx(0) + 15, my(10) - 6, "9", 6, ACCENT, "Helvetica-Bold")
    return mx, my


def _boxglyph(g, x, y, color, label):
    g.add(Rect(x - 6, y - 6, 12, 12, fillColor=WHITE, strokeColor=color,
               strokeWidth=1.4, rx=1.5, ry=1.5))
    g.add(Rect(x - 3, y - 3, 6, 6, fillColor=color, strokeColor=None))
    _text(g, x, y + 9, label, 6, color, "Helvetica-Bold")


# ---------------------------------------------------------------------------
# Figure 1 — System architecture (clean vertical stack)
# ---------------------------------------------------------------------------

def fig_architecture():
    W, H = 16.4 * cm, 7.5 * cm
    d = Drawing(W, H)
    d.hAlign = "CENTER"
    _frame(d)
    d.add(Rect(0.3 * cm, 0.3 * cm, W - 0.6 * cm, H - 0.95 * cm, fillColor=None,
               strokeColor=ACCENT, strokeWidth=0.8, rx=6, ry=6,
               strokeDashArray=[3, 3]))
    _text(d, W / 2, H - 0.62 * cm, "Tailscale tailnet — air-gapped, MagicDNS",
          7.5, ACCENT, "Helvetica-Oblique")
    g = Group()
    cx = 6.2 * cm
    bw = 7.4 * cm
    bx = cx - bw / 2

    # Drone (x5)
    dy = 5.55 * cm
    _drone(g, cx, dy + 0.45 * cm, 9, NAVY, ACCENT)
    _text(g, cx, dy - 0.05 * cm, "Parrot Anafi  ×5   (caged)", 8.5, NAVY,
          "Helvetica-Bold")
    # FC (x5)
    fy = 3.75 * cm
    _node(g, bx, fy, bw, 1.25 * cm, "marker_mission   ·   flight controller  :8080",
          "ArUco vision  ·  mission language  ·  Olympe link", fill=SKY,
          stroke=ACCENT, stack=2)
    _arrow(g, cx, dy - 0.35 * cm, cx, fy + 1.25 * cm + 0.18 * cm, NAVY, 1.1, 6)
    # C2
    c2y = 2.05 * cm
    _node(g, bx, c2y, bw, 1.2 * cm, "marker_mission_c2   ·   fleet  :8090",
          "polls every drone  ·  relays commands  ·  live state",
          fill=NAVY, stroke=NAVY, tcolor=WHITE)
    _arrow(g, cx, fy, cx, c2y + 1.2 * cm, NAVY, 1.1, 6)
    _text(g, cx + 1.7 * cm, fy - 0.30 * cm, "HTTP / JSON", 6.3, MUTED,
          "Helvetica-Oblique")
    # strategy
    sy = 0.55 * cm
    _node(g, bx, sy, bw, 1.2 * cm, "strategy   :8091 red  /  :8092 blue",
          "tracks the boxes  ·  assigns role + target  ·  planner",
          fill=NAVY, stroke=NAVY, tcolor=WHITE)
    _arrow(g, cx, c2y, cx, sy + 1.2 * cm, NAVY, 1.1, 6)
    # Simulator (right)
    simx = 12.2 * cm
    _node(g, simx, fy, 3.4 * cm, 1.25 * cm, "Parrot Sphinx",
          "marker_mission_sim", fill=REDZONE, stroke=RED, tcolor=RED)
    _arrow(g, simx, fy + 0.62 * cm, bx + bw, fy + 0.62 * cm, RED, 1.0, 5,
           dash=[2, 2])
    _text(g, simx + 1.7 * cm, fy - 0.30 * cm, "drop-in flight controllers",
          6.3, RED, "Helvetica-Oblique")
    d.add(g)
    return d


# ---------------------------------------------------------------------------
# Figure 2 — Localisation pipeline
# ---------------------------------------------------------------------------

def fig_localisation():
    W, H = 16.4 * cm, 6.0 * cm
    d = Drawing(W, H)
    d.hAlign = "CENTER"
    _frame(d)
    g = Group()
    midy = 3.55 * cm
    bw, bh = 2.5 * cm, 1.2 * cm
    xs = [0.35, 3.0, 5.65]
    labs = [("Camera", "~25 fps"), ("ArUco detect", "DICT_4X4_50"),
            ("IPPE pose", "planar PnP")]
    for (lab, sub), xc in zip(labs, xs):
        _node(g, xc * cm, midy, bw, bh, lab, sub, fill=SKY, stroke=ACCENT)
    _cam(g, 1.0 * cm, midy + bh + 0.28 * cm, ACCENT)
    _marker(g, 5.65 * cm + bw + 0.08 * cm, midy + bh - 0.1 * cm, 10, NAVY)
    _arrow(g, 2.85 * cm, midy + bh / 2, 3.0 * cm, midy + bh / 2, NAVY, 1.0, 5)
    _arrow(g, 5.5 * cm, midy + bh / 2, 5.65 * cm, midy + bh / 2, NAVY, 1.0, 5)

    # fork into two branches
    _arrow(g, 8.15 * cm, midy + bh / 2, 8.7 * cm, midy + bh + 0.3 * cm, NAVY,
           1.0, 5)
    _arrow(g, 8.15 * cm, midy + bh / 2, 8.7 * cm, midy - 0.35 * cm, NAVY, 1.0, 5)
    _node(g, 8.75 * cm, midy + bh - 0.05 * cm, 1.95 * cm, 0.66 * cm,
          "branch A", fill=WHITE, stroke=MUTED, tcolor=INK, tsize=7.5)
    _node(g, 8.75 * cm, midy - 0.72 * cm, 1.95 * cm, 0.66 * cm,
          "branch B = mirror", fill=WHITE, stroke=MUTED, tcolor=INK, tsize=6.7)
    _node(g, 11.05 * cm, midy + 0.5 * cm, 2.55 * cm, 1.35 * cm, "Pick the real one",
          "compass + altimeter", fill=BLUEZONE, stroke=ACCENT, ssize=6.6)
    _arrow(g, 10.7 * cm, midy + bh, 11.05 * cm, midy + 1.05 * cm, ACCENT, 1.0, 5)
    _arrow(g, 10.7 * cm, midy - 0.4 * cm, 11.05 * cm, midy + 0.85 * cm, ACCENT,
           1.0, 5)

    lowy = 0.95 * cm
    _node(g, 11.05 * cm, lowy, 2.55 * cm, 1.2 * cm, "Fuse markers",
          "by nearness + anchor", fill=SKY, stroke=ACCENT, ssize=6.6)
    _arrow(g, 12.3 * cm, midy + 0.5 * cm, 12.3 * cm, lowy + 1.2 * cm, NAVY, 1.0, 5)
    _node(g, 13.85 * cm, lowy, 2.3 * cm, 1.2 * cm, "Kalman filter",
          "6-state, +IMU vel.", fill=NAVY, stroke=NAVY, tcolor=WHITE, ssize=6.6)
    _arrow(g, 13.6 * cm, lowy + 0.6 * cm, 13.85 * cm, lowy + 0.6 * cm, NAVY,
           1.0, 5)
    _text(g, 15.0 * cm, lowy - 0.32 * cm, "position (x, y, z)", 6.6, NAVY,
          "Helvetica-Bold")
    _text(g, 12.9 * cm, midy + 2.15 * cm,
          "Anafi velocity  →  arena frame", 6.6, MUTED, "Helvetica-Oblique")
    _arrow(g, 15.0 * cm, midy + 1.95 * cm, 15.0 * cm, lowy + 1.2 * cm, MUTED,
           0.9, 5, dash=[2, 2])
    d.add(g)
    return d


# ---------------------------------------------------------------------------
# Figure 3 — Attack trajectory + emitted script
# ---------------------------------------------------------------------------

def fig_attack():
    W, H = 16.4 * cm, 8.0 * cm
    d = Drawing(W, H)
    d.hAlign = "CENTER"
    _frame(d)
    g = Group()
    ax, ay, aw, ah = 0.7 * cm, 0.7 * cm, 5.5 * cm, 6.4 * cm
    mx, my = _arena(g, ax, ay, aw, ah)
    _text(g, ax + aw / 2, ay + ah + 0.18 * cm, "RED home", 6.5, RED,
          "Helvetica-Bold")
    _text(g, ax + aw / 2, ay - 0.30 * cm, "20 m × 10 m, top-down", 6, MUTED,
          "Helvetica-Oblique")
    for sl, (X, Y) in SLOTS.items():
        _boxglyph(g, mx(X), my(Y), RED if sl <= 3 else ACCENT, str(sl))
    start = (mx(-1.5), my(-6.0))
    appr = (mx(0), my(7.4))
    over = (mx(0), my(9))
    g.add(Circle(start[0], start[1], 5, fillColor=RED, strokeColor=NAVY))
    _arrow(g, start[0], start[1], appr[0], appr[1] - 0.25 * cm, NAVY, 1.5, 7)
    _arrow(g, appr[0], appr[1], over[0], over[1], ACCENT, 1.5, 6)
    _star(g, over[0], over[1], 7, RED)
    rp = PolyLine([over[0], over[1], mx(2.3), my(2), mx(0), my(-6.5)],
                  strokeColor=MUTED, strokeWidth=1.2)
    rp.strokeDashArray = [4, 3]
    g.add(rp)
    _arrow(g, mx(1.5), my(-3), mx(0.1), my(-6.3), MUTED, 1.2, 6, dash=[4, 3])
    _drone(g, appr[0], appr[1] - 0.18 * cm, 6, NAVY, ACCENT)

    sx = 6.95 * cm
    _text(g, sx, ay + ah + 0.18 * cm, "What the strategy server sends  (attack slot 5)",
          7.5, NAVY, "Helvetica-Bold", "start")
    script = [
        ("TAKEOFF", ""),
        ("YAW 0", "face the enemy half"),
        ("HEIGHT 2.20", "climb above the 0.73 m boxes"),
        ("APPROACH 35 1.00", "home to 1 m off the box, by sight"),
        ("HEIGHT 1.50", "drop into the 1–2 m capture band"),
        ("FB_IMU 0.90", "one bounded step over the centre"),
        ("HOOVER 3.5", "hover ≥2 s  →  35 flips to 45"),
        ("HEIGHT 2.20", "climb to clear the boxes"),
        ("YAW 180", "turn toward home"),
        ("APPROACH 13 3.50", "home onto our wall marker"),
        ("YAW 0", "re-face enemy, ready again"),
    ]
    ly = ay + ah - 0.28 * cm
    for code, note in script:
        g.add(String(sx, ly, code, fontName="Courier-Bold", fontSize=8,
                     fillColor=NAVY))
        if note:
            g.add(String(sx + 3.0 * cm, ly, "# " + note, fontName="Courier",
                         fontSize=7, fillColor=MUTED))
        ly -= 0.46 * cm
    ly -= 0.12 * cm
    g.add(Line(sx, ly, sx + 0.5 * cm, ly, strokeColor=NAVY, strokeWidth=1.5))
    _text(g, sx + 0.62 * cm, ly - 2.5, "bounded move", 7, INK, "Helvetica",
          "start")
    g.add(Line(sx + 3.0 * cm, ly, sx + 3.5 * cm, ly, strokeColor=ACCENT,
               strokeWidth=1.5))
    _text(g, sx + 3.62 * cm, ly - 2.5, "vision homing", 7, INK, "Helvetica",
          "start")
    ly -= 0.42 * cm
    lr = Line(sx, ly, sx + 0.5 * cm, ly, strokeColor=MUTED, strokeWidth=1.2)
    lr.strokeDashArray = [4, 3]
    g.add(lr)
    _text(g, sx + 0.62 * cm, ly - 2.5, "return home", 7, INK, "Helvetica",
          "start")
    _star(g, sx + 3.2 * cm, ly + 1, 5, RED)
    _text(g, sx + 3.55 * cm, ly - 2.5, "capture (2 s hover)", 7, INK,
          "Helvetica", "start")
    d.add(g)
    return d


# ---------------------------------------------------------------------------
# Figure 4 — Swarm roles + planner ladder
# ---------------------------------------------------------------------------

def fig_swarm():
    W, H = 16.4 * cm, 8.0 * cm
    d = Drawing(W, H)
    d.hAlign = "CENTER"
    _frame(d)
    g = Group()
    ax, ay, aw, ah = 0.7 * cm, 0.7 * cm, 5.5 * cm, 6.4 * cm
    mx, my = _arena(g, ax, ay, aw, ah)
    _text(g, ax + aw / 2, ay + ah + 0.18 * cm, "Red team, mid-match", 7,
          NAVY, "Helvetica-Bold")
    for sl, (X, Y) in SLOTS.items():
        _boxglyph(g, mx(X), my(Y), RED if sl <= 3 else ACCENT, str(sl))
    g.add(Circle(mx(0), my(0), 13, fillColor=None, strokeColor=ACCENT,
                 strokeWidth=0.8, strokeDashArray=[2, 2]))
    _drone(g, mx(0), my(0), 6, NAVY, ACCENT)
    _text(g, mx(0), my(0) - 0.62 * cm, "scout", 6.3, NAVY, "Helvetica-Bold")
    _drone(g, mx(-2.0), my(-4.2), 6, RED, WHITE)
    _text(g, mx(-2.0), my(-4.2) - 0.6 * cm, "defender", 6.3, RED,
          "Helvetica-Bold")
    for (sx_, sy_), (tx, ty) in (((1.8, -4.0), (3, 6.5)), ((3.0, -1.5), (0, 9))):
        _drone(g, mx(sx_), my(sy_), 5, NAVY, ACCENT)
        _arrow(g, mx(sx_), my(sy_) + 0.2 * cm, mx(tx), my(ty) - 0.28 * cm,
               NAVY, 1.0, 5, dash=[3, 2])
    _text(g, mx(2.7), my(-3.0) - 0.15 * cm, "attackers", 6.3, NAVY,
          "Helvetica-Bold")

    lx, lw = 6.95 * cm, 8.95 * cm
    _text(g, lx, ay + ah + 0.18 * cm,
          "Every tick, the first rule that fits wins", 7.5, NAVY,
          "Helvetica-Bold", "start")
    rungs = [
        ("Guard", "dead / low battery / lost link  →  go home or idle",
         MIST, MUTED),
        ("Defend", "our box turned enemy  →  nearest free drone recaptures NOW",
         REDZONE, RED),
        ("5-point sortie", "≥2 drones can each reach an enemy box  →  go "
         "together, whole team out of home", BLUEZONE, ACCENT),
        ("Attack", "otherwise  →  each free drone to its nearest enemy box "
         "(red→4-6, blue→1-3)", SKY, ACCENT),
        ("Bank", "after a big attempt  →  all home together before the enemy "
         "flips back", MIST, NAVY),
    ]
    ry = ay + ah - 0.40 * cm
    rh = 1.06 * cm
    for i, (title, body, fill, stroke) in enumerate(rungs):
        _card(g, lx, ry - rh, lw, rh, fill, stroke, 1.0)
        g.add(Circle(lx + 0.42 * cm, ry - 0.42 * cm, 7, fillColor=stroke,
                     strokeColor=stroke))
        _text(g, lx + 0.42 * cm, ry - 0.55 * cm, str(i), 8, WHITE,
              "Helvetica-Bold")
        g.add(String(lx + 0.78 * cm, ry - 0.36 * cm, title,
                     fontName="Helvetica-Bold", fontSize=9, fillColor=NAVY))
        words, line, lines = body.split(), "", []
        for wd in words:
            if len(line) + len(wd) + 1 > 62:
                lines.append(line); line = wd
            else:
                line = (line + " " + wd).strip()
        if line:
            lines.append(line)
        yy = ry - 0.62 * cm
        for li in lines[:2]:
            g.add(String(lx + 0.78 * cm, yy, li, fontName="Helvetica",
                         fontSize=7, fillColor=INK))
            yy -= 0.30 * cm
        if i < len(rungs) - 1:
            _arrow(g, lx + lw / 2, ry - rh, lx + lw / 2, ry - rh - 0.18 * cm,
                   MUTED, 0.8, 4)
        ry -= rh + 0.22 * cm
    d.add(g)
    return d


# ---------------------------------------------------------------------------
# Footer
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

    def _stamp(self, i, n):
        if i == 1:
            return
        self.setStrokeColor(RULE)
        self.setLineWidth(0.4)
        self.line(2 * cm, 1.55 * cm, A4[0] - 2 * cm, 1.55 * cm)
        self.setFillColor(MUTED)
        self.setFont("Helvetica-Oblique", 8)
        self.drawString(2 * cm, 1.05 * cm,
                        "SDC26 — Team ToBeDefined — Technical Report")
        self.drawRightString(A4[0] - 2 * cm, 1.05 * cm, f"Page {i} of {n}")


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def cover():
    img = Image(str(LOGO), width=5.8 * cm, height=5.8 * cm)
    img.hAlign = "CENTER"
    return [
        Spacer(1, 1.5 * cm), img, Spacer(1, 0.7 * cm),
        P("SDC26", "Tag"),
        P("Technical Report", "TitleBig"),
        P("Team ToBeDefined", "Sub"),
        Spacer(1, 0.2 * cm),
        P("Swarm Drone Challenge 2026 — ILA Berlin, 11 June 2026", "Sub"),
        Spacer(1, 1.3 * cm),
        P("Abstract", "AbstractTitle"),
        P("Team ToBeDefined flies a swarm of Parrot Anafi drones in the Swarm "
          "Drone Challenge 2026. Every drone runs the same software — on the "
          "real flight controller and in the Parrot Sphinx simulator — so we "
          "test in simulation exactly what we fly. Positioning is vision-only "
          "(ArUco markers, no GPS), manoeuvres are written in a small "
          "mission-scripting language, and a central command-and-control "
          "overlay coordinates the swarm. One rule runs through the whole "
          "design: move by what the camera can see, in small bounded steps, "
          "rather than steering to absolute coordinates we cannot trust "
          "indoors. The full stack — about 90,000 lines of Python — is "
          "open-source."),
    ]


def section_intro_and_architecture():
    return [
        PageBreak(),
        P("1.  Introduction", "H1"),
        P("The Swarm Drone Challenge 2026 is an indoor capture-the-flag for "
          "drones. Two teams of up to five Parrot Anafi share a 20 m × 10 m "
          "arena. Each team defends three target boxes and tries to flip the "
          "three enemy boxes by hovering over them — within a 1–2 m band — for "
          "at least two seconds. The boxes carry ArUco markers (DICT_4X4_50) "
          "whose colour flips on capture (leading digit 4 = red, 3 = blue; "
          "trailing digit is the box, 1–6). Points come in tiers:"),
        BUL("<b>1 point</b> — a plain capture."),
        BUL("<b>5 points</b> — a capture with the whole team already outside "
            "its home zone."),
        BUL("<b>10 points</b> — two captures by two drones within one second."),
        BUL("<b>Instant win</b> — hold all six boxes for five seconds."),
        P("A match lasts ten minutes, and the hall has no GPS — so every "
          "position comes from the on-board cameras and inertial sensors."),
        HEAD(P("2.  System Architecture", "H1"),
             KEY("Three small services share one code base: a flight "
                 "controller per drone, a fleet hub, and a per-team strategy "
                 "brain — all plain HTTP/JSON.")),
        FIG(fig_architecture(),
            CAP("Figure 1 — The runtime. One flight controller per caged "
                "Anafi; the C2 aggregates the fleet; the per-team strategy "
                "servers decide each drone's job. The Sphinx simulator drops "
                "in for the real controllers during development.")),
        BUL("<b>marker_mission</b> (:8080), one per drone — the ArUco vision "
            "loop, the mission language, and the Olympe link to the Anafi."),
        BUL("<b>marker_mission_c2</b> (:8090) — polls every drone, relays "
            "operator commands, and shows live state."),
        BUL("<b>strategy</b> (:8091 red / :8092 blue) — tracks the six boxes "
            "and assigns each drone a role and a target."),
        P("The same code drives the Parrot Sphinx simulator, so a change is "
          "proven in simulation before it flies. One operator switch re-points "
          "a drone from the simulator to a real controller by IP; another "
          "hot-swaps the whole stack between the competition arena and a "
          "smaller test arena — live, with no restart."),
        P("Manoeuvres are written in a small text language. Its commands fall "
          "into four families:"),
        BUL("<b>Bounded moves</b> "
            "(<font face='Courier'>FB_IMU, LR_IMU, UD_IMU, YAW_IMU</font>) — "
            "go an exact distance or angle on the drone's own sensors, then "
            "stop."),
        BUL("<b>Heading hold</b> (<font face='Courier'>YAW</font>) — face a "
            "fixed arena direction, so a move never inherits a stale heading."),
        BUL("<b>Vision homing</b> "
            "(<font face='Courier'>APPROACH, FB_BRAKE</font>) — fly toward a "
            "named marker by sight; drift-free, no matter the world-frame fix."),
        BUL("<b>Stick windows &amp; holds</b> "
            "(<font face='Courier'>FB_RC, UD_RC, HOOVER</font>) — push a stick "
            "for a set time, or hover in place."),
    ]


def section_localisation():
    return [
        HEAD(P("3.  Localisation", "H1"),
             KEY("Two cameras and a compass turn wall markers into a position "
                 "— and we throw away any fix the sensors disagree on.")),
        P("There is no GPS, so each drone works out where it is from the ArUco "
          "markers on the arena walls: eight pillars, sixteen markers (one at "
          "2 m and one at 4 m on each), 0.5 m squares. The vision loop detects "
          "them and solves each marker's pose (Figure 2)."),
        FIG(fig_localisation(),
            CAP("Figure 2 — From camera to position. The pose solver gives two "
                "mirror-image answers; a compass and an altimeter pick the "
                "real one; nearby markers are fused around an anchor and "
                "smoothed by a six-state Kalman filter that also reads the "
                "Anafi's velocity.")),
        P("3.1  Resolving the mirror ambiguity", "H2"),
        P("A single marker's pose solver returns two answers that are mirror "
          "images of each other — and only one is real (the other puts the "
          "drone on the far side of the marker). We pick the right one with two "
          "independent checks:"),
        BUL("<b>Compass</b> — keep the branch whose implied heading matches the "
            "magnetometer (within about 12°). This needs no altitude, so "
            "take-off starts from a clean fix."),
        BUL("<b>Altimeter</b> — once airborne, the Anafi's downward ultrasound "
            "confirms the height of the chosen branch."),
        P("If neither check agrees and only one marker is in view, we reject "
          "the fix and let the last good estimate fade — better than jumping to "
          "a mirrored pose."),
        P("3.2  Smoothing and multiple markers", "H2"),
        P("A six-state Kalman filter — position and velocity, "
          "<font face='Courier'>[px, py, pz, vx, vy, vz]</font> — blends the "
          "slow but accurate marker fixes with the Anafi's fast velocity "
          "telemetry (rotated into the arena frame using the calibrated "
          "magnetic-north offset). When several markers are visible, their "
          "positions are averaged by nearness and tied to a remembered "
          "<i>anchor</i>, so one noisy reading cannot drag the estimate across "
          "the field. The filter also supplies the velocity the controller "
          "uses to damp side-to-side sway while homing."),
        P("3.3  No GPS", "H2"),
        P("No GPS is used in flight — only as off-line ground truth in the "
          "simulator, per §4.4. And we never steer open-loop on an absolute "
          "fix: every precise move homes on a marker the camera can actually "
          "see (Section 4), which stays correct no matter which mirror the "
          "estimator picked."),
    ]


def section_flight_control():
    return [
        HEAD(P("4.  Flight Control and Trajectory Planning", "H1"),
             KEY("Every precise move is bounded and vision-relative — never an "
                 "open-loop dash across the field.")),
        P("The Anafi accepts both continuous stick commands and exact "
          "“move this far and stop” commands; we combine them with vision "
          "homing as building blocks (Section 2). Every attack is generated "
          "automatically for the chosen target and follows one pattern "
          "(Figure 3):"),
        BUL("Climb above the 0.73 m boxes."),
        BUL("<font face='Courier'>APPROACH</font> the enemy marker by sight, "
            "stopping 1 m short."),
        BUL("Rise into the 1–2 m capture band."),
        BUL("One bounded <font face='Courier'>FB_IMU</font> step over the box "
            "centre."),
        BUL("Hover at least two seconds — the marker flips."),
        BUL("Climb, turn, and <font face='Courier'>APPROACH</font> our own "
            "wall marker to return."),
        FIG(fig_attack(),
            CAP("Figure 3 — The attack manoeuvre (left) and the exact script "
                "the strategy server emits (right). Navy = bounded move, blue "
                "= vision homing, dashed = vision-homed return, star = the "
                "2-second capture hover.")),
        P("Three lessons shaped this pattern:"),
        BUL("<b>No open-loop cruises.</b> Pitching forward tilts the camera "
            "down and hides the target marker, so a vision brake never fires "
            "and the drone runs to its timeout at full stick. Bounded steps and "
            "vision homing avoid this entirely."),
        BUL("<b>A bounded step over the box.</b> Sized to the 1 m stand-off, so "
            "even a noisy fix stops over the target rather than carrying on "
            "into the wall behind it."),
        BUL("<b>Closed-loop turns.</b> "
            "<font face='Courier'>YAW</font>/<font face='Courier'>YAW_IMU</font> "
            "hold an exact heading; a yaw-rate stick drifted in testing and "
            "sent the next leg sideways."),
        P("4.1  Safety", "H2"),
        BUL("A hard altitude ceiling clamps the throttle channel."),
        BUL("A watchdog auto-lands a drone if the operator link drops for more "
            "than two seconds (§3.3)."),
        BUL("A single key-press lands the whole fleet — the Safety Officer's "
            "kill-all (§2.2)."),
        BUL("Every drone flies inside a protective cage (§3.2)."),
        BUL("The simulator enforces the same arena walls, so a script that "
            "would hit one is caught before it ever flies."),
    ]


def section_swarm():
    return [
        HEAD(P("5.  Swarm Strategy", "H1"),
             KEY("A simple priority ladder, re-decided every tick — defend "
                 "first, then go for points.")),
        P("A small, rule-based planner runs on the strategy server every tick. "
          "It reads each drone (link, battery, position, phase) and the live "
          "box map, then hands every drone a role and a target. There is no "
          "learned policy — the rules are fixed and easy to follow, which "
          "matters because the code is open (Figure 4). The first rule that "
          "applies wins:"),
        FIG(fig_swarm(),
            CAP("Figure 4 — Left: a typical mix — a scout refreshing the box "
                "map from the centre, two attackers vectored at enemy boxes, a "
                "defender holding in the neutral zone. Right: the per-tick "
                "priority ladder.")),
        P("A recapture inside our own home zone scores nothing (§1.4.3), so the "
          "planner treats it as defence, not points. Holding all six boxes for "
          "five seconds is the instant win and overrides everything when it is "
          "within reach."),
        P("5.1  Roles and fair play", "H2"),
        BUL("<b>Four roles:</b> attacker, scout (spins at the centre to keep "
            "the box map fresh), defender, and idle/return."),
        BUL("<b>No double-booking.</b> Three running sets — slots already taken "
            "this tick, slots an in-flight drone is already heading to, and "
            "slots inside the 5-second post-capture lock — stop two drones "
            "chasing the same box."),
        BUL("<b>Asymmetric attack.</b> Red may target only boxes 4–6, blue only "
            "1–3; the planner refuses any operator override that breaks this."),
        P("5.2  The defender", "H2"),
        P("The defender never camps over a box — that risks the "
          "dead-drone-over-target rule and forfeits the coordination bonus. "
          "Instead it hovers just inside the neutral zone, facing its own back "
          "wall and holding station by sight on the back-wall marker. Because "
          "the box map is shared, it sees a loss the instant a marker flips, "
          "darts in to recapture, then returns to station."),
        P("5.3  Open source", "H2"),
        P("Everything is open at "
          "<font face='Courier'>github.com/otiedemann/sdc-tobe</font> — about "
          "90,000 lines of Python across the flight controller, the C2, the "
          "strategy layer, and the simulator, sharing one configuration and one "
          "environment. Every flight is logged tick-by-tick and recorded to "
          "video for an in-browser replay viewer, so any result in this report "
          "can be reproduced."),
        P("6.  Conclusion", "H1"),
        P("One idea runs through the whole stack: move by what the camera can "
          "see, in small bounded steps, instead of steering to coordinates we "
          "cannot trust indoors. That is what keeps the swarm flying safely and "
          "repeatably. What remains — a sanity check that lets the active wall "
          "guard switch back on once a fix is trusted, and more tuning of the "
          "coordinated five-point sortie — is scoped for the final at ILA "
          "Berlin on 11 June 2026."),
    ]


def main():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
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
