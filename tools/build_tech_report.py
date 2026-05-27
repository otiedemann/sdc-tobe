#!/usr/bin/env python3
"""Build the SDC26 ToBeDefined Technical Report as a 5-page A4 PDF.

Pure ``reportlab`` (no LaTeX, no external HTML→PDF tools). Embeds the
team logo and follows the three required sections from §1.6.3 of the
regulations: localisation, flight control & trajectory planning, swarm
strategy. Output:

    1_Doc/SDC26_ToBeDefined_Technical_Report.pdf
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as canvas_mod
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
)

ROOT = Path(__file__).resolve().parents[1]
LOGO = ROOT / "1_Doc" / "team_logo_pdf.png"
OUT = ROOT / "1_Doc" / "SDC26_ToBeDefined_Technical_Report.pdf"

# Brand palette
NAVY = HexColor("#10243d")
ACCENT = HexColor("#3b82c4")
MUTED = HexColor("#6b7280")
RULE = HexColor("#cbd5e1")
CODE_BG = HexColor("#f3f4f6")


def make_styles():
    s = {}
    s["TitleBig"] = ParagraphStyle(
        "TitleBig", fontName="Times-Bold", fontSize=28, leading=34,
        alignment=TA_CENTER, textColor=NAVY, spaceAfter=8,
    )
    s["Sub"] = ParagraphStyle(
        "Sub", fontName="Times-Roman", fontSize=13, leading=17,
        alignment=TA_CENTER, textColor=MUTED, spaceAfter=4,
    )
    s["Tag"] = ParagraphStyle(
        "Tag", fontName="Helvetica-Bold", fontSize=10, leading=12,
        alignment=TA_CENTER, textColor=ACCENT, spaceAfter=2,
    )
    s["AbstractTitle"] = ParagraphStyle(
        "AbstractTitle", fontName="Helvetica-Bold", fontSize=10,
        leading=12, alignment=TA_LEFT, textColor=NAVY, spaceAfter=4,
    )
    s["H1"] = ParagraphStyle(
        "H1", fontName="Times-Bold", fontSize=15, leading=19,
        textColor=NAVY, spaceBefore=2, spaceAfter=6,
    )
    s["H2"] = ParagraphStyle(
        "H2", fontName="Times-Bold", fontSize=11.5, leading=14,
        textColor=NAVY, spaceBefore=6, spaceAfter=3,
    )
    s["Body"] = ParagraphStyle(
        "Body", fontName="Times-Roman", fontSize=10, leading=13,
        alignment=TA_JUSTIFY, spaceAfter=4,
    )
    s["Code"] = ParagraphStyle(
        "Code", fontName="Courier", fontSize=8.5, leading=11,
        backColor=CODE_BG, borderColor=RULE, borderWidth=0.4,
        borderPadding=6, leftIndent=0, rightIndent=0,
        spaceBefore=2, spaceAfter=6,
    )
    return s


S = make_styles()


def P(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, S[style])


class FooterCanvas(canvas_mod.Canvas):
    """Stamps a Page X of N footer on every page except the cover."""

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
            return  # leave the cover clean
        self.setStrokeColor(RULE)
        self.setLineWidth(0.4)
        self.line(2 * cm, 1.55 * cm, A4[0] - 2 * cm, 1.55 * cm)
        self.setFillColor(MUTED)
        self.setFont("Times-Italic", 8.5)
        self.drawString(2 * cm, 1.05 * cm,
                        "SDC26 — Team ToBeDefined — Technical Report")
        self.drawRightString(A4[0] - 2 * cm, 1.05 * cm, f"Page {i} of {n}")


def cover() -> list:
    story = [Spacer(1, 1.6 * cm)]
    img = Image(str(LOGO), width=6.2 * cm, height=6.2 * cm)
    img.hAlign = "CENTER"
    story += [img, Spacer(1, 0.8 * cm)]
    story += [
        P("SDC26", "Tag"),
        P("Technical Report", "TitleBig"),
        P("Team ToBeDefined", "Sub"),
        Spacer(1, 0.3 * cm),
        P("Swarm Drone Challenge 2026 — ILA Berlin, 11 June 2026", "Sub"),
        Spacer(1, 1.5 * cm),
        P("Abstract", "AbstractTitle"),
        P(
            "This report describes the technical solution developed by "
            "Team ToBeDefined for the Swarm Drone Challenge 2026. Each "
            "flight controller runs an on-board mission engine driven "
            "by ArUco-only positioning, a domain-specific mission "
            "scripting language, and a shared command-and-control "
            "overlay that coordinates the swarm across the 20 m × 10 m "
            "arena. The design prioritises drift-free, vision-relative "
            "homing over absolute world-frame steering, uses bounded "
            "closed-loop primitives for repeatable manoeuvres, and "
            "exposes a rule-based strategy layer that translates the "
            "game's scoring tiers into per-drone role and target "
            "assignments. The system is implemented across roughly "
            "96,000 lines of code (≈84,000 in Python) and is "
            "open-sourced in line with the competition's encouragement "
            "of shared code."
        ),
    ]
    return story


def section_intro_and_architecture() -> list:
    return [
        PageBreak(),
        P("1.  Introduction", "H1"),
        P(
            "The Swarm Drone Challenge 2026 (SDC26) pits two teams of "
            "up to five Parrot Anafi drones against each other in a "
            "20 m × 10 m indoor arena. Each side defends three "
            "target boxes in its home zone and attempts to flip the "
            "three enemy boxes by hovering over them for at least two "
            "seconds. The boxes carry DICT_4X4_50 ArUco markers whose "
            "encoding (leading digit – 4 for red, 3 for blue – "
            "indicates owner; trailing digit is the box ID 1–6) "
            "flips on capture. Points accrue per attempt at three tiers "
            "(1, 5, or 10) with an instant-win Special Maneuver awarded "
            "for controlling all six boxes for five continuous seconds. "
            "A match lasts ten minutes; GPS is unavailable in the "
            "indoor venue, so all positioning is derived from on-board "
            "cameras and inertial sensors."
        ),
        P(
            "Team ToBeDefined attacks the problem with a single, "
            "uniform software stack that runs both on each drone's "
            "flight controller (an x86 Linux companion computer flying "
            "the Anafi via Olympe) and on a lightweight central "
            "command-and-control (C2) overlay. The same code base "
            "powers the Parrot Sphinx simulator, allowing every change "
            "to be validated against the same mission scripts and the "
            "same telemetry contracts before flying real hardware."
        ),
        P("2.  System Architecture", "H1"),
        P(
            "Three processes form the runtime. <b>marker_mission</b> "
            "on port 8080 is the per-drone flight controller – it "
            "owns the Olympe connection, runs the ArUco vision loop, "
            "executes the mission scripting language, and exposes a "
            "small HTTP API for telemetry, RC, and mission control. "
            "<b>marker_mission_c2</b> on port 8090 is the fleet "
            "dashboard: it aggregates the five flight controllers, "
            "proxies operator commands, and serves the live state UI. "
            "The <b>strategy</b> module on port 8091 layers on top of "
            "the C2, providing slot tracking, manual attack triggers, "
            "and the entry point for the autonomous Commander "
            "described in Section 5."
        ),
        P(
            "All inter-process traffic is HTTP/JSON, and the entire "
            "fleet – five flight controllers and the operator's "
            "laptop – is joined into a single Tailscale tailnet so "
            "the operator UIs are reachable by MagicDNS name from "
            "anywhere, while still cleanly air-gapped from the public "
            "internet. Mission behaviour itself is expressed in a "
            "small text-based scripting language (the <i>mission DSL</i>) "
            "parsed and executed by the flight controller. Eighteen "
            "primitives cover discrete IMU closed-loop moves "
            "(<font face='Courier'>FB_IMU</font>, "
            "<font face='Courier'>YAW_IMU</font>), open-loop stick "
            "windows (<font face='Courier'>FB_RC</font>, "
            "<font face='Courier'>UD_RC</font>), vision homing "
            "(<font face='Courier'>APPROACH</font>, "
            "<font face='Courier'>FB_BRAKE</font>), and world-frame "
            "navigation (<font face='Courier'>TO</font>, "
            "<font face='Courier'>TO_HOME</font>). Each step compiles "
            "to a state-machine phase that emits a 20 Hz RC stream, "
            "gives vision its own thread, and writes a structured "
            "per-tick log used by the offline replay viewer for tuning."
        ),
    ]


def section_localisation() -> list:
    return [
        PageBreak(),
        P("3.  Localisation", "H1"),
        P(
            "The arena provides eight pillars carrying sixteen ArUco "
            "markers – an upper marker at 4 m and a lower marker "
            "at 2 m per pillar; IDs 1–8 upper, 9–16 lower, "
            "0.5 × 0.5 m, DICT_4X4_50. Each flight controller's "
            "vision loop runs OpenCV ArUco detection on the on-board "
            "camera and computes a per-marker pose with the IPPE_SQUARE "
            "planar-pose algorithm. Planar PnP returns two equally-"
            "valid solutions related by a reflection about the marker "
            "plane; we disambiguate with two independent priors."
        ),
        P("3.1  Magnetometer prior", "H2"),
        P(
            "The arena's magnetic-north heading is calibrated once and "
            "stored in the active arena configuration. The two IPPE "
            "branches yield two candidate drone headings; the branch "
            "whose implied heading agrees with the live magnetometer "
            "reading within a configurable slack (default 35°) is "
            "accepted. This works at every altitude, including on the "
            "ground, which is critical for a clean state at takeoff."
        ),
        P("3.2  Altimeter prior", "H2"),
        P(
            "When the drone is airborne, the Anafi's ultrasound "
            "altimeter provides a second consistency check on the "
            "vertical component of each branch. We require agreement "
            "from at least one of the two priors; if only one marker "
            "is visible and the magnetometer is uncalibrated, the fix "
            "is rejected and the previous estimate decays."
        ),
        P("3.3  Kalman filter and anchor", "H2"),
        P(
            "Per-tick raw fixes are fused with Anafi-reported body "
            "velocities through a position Kalman filter (constant-"
            "velocity model, configurable process and measurement "
            "noise). The filter both smooths short-term measurement "
            "jitter and supplies a velocity estimate that the "
            "controller uses to damp lateral oscillation during "
            "APPROACH. When multiple reference markers are "
            "simultaneously visible, the per-marker world positions are "
            "weighted by inverse distance and fused into a single "
            "arena fix; the dominant contributor is remembered as the "
            "<i>anchor</i> across frames so that noisy single-marker "
            "fixes cannot drag the estimate in the opposite direction."
        ),
        P("3.4  Failure modes", "H2"),
        P(
            "In practice the ArUco pipeline can still mislabel the "
            "IPPE branch in metal-rich indoor environments where the "
            "magnetometer is unreliable. We learned the hard way that "
            "an active reverse-brake wall guard acting on a mirrored "
            "single-marker fix will drive the drone <i>into</i> the "
            "wall it appears to be near, at saturated RC. The arena "
            "guard is therefore disabled by default in the current "
            "code base; the planned mitigation is a position-sanity "
            "gate that requires a fresh, low-jump fix before allowing "
            "any active correction, so wall protection can be safely "
            "re-enabled once positioning is trusted. No GPS information "
            "is used anywhere in the live positioning or control path, "
            "as required by §4.4 of the regulations – GPS is "
            "used only as an off-line ground-truth reference in the "
            "simulator during tuning."
        ),
    ]


def section_flight_control() -> list:
    code = (
        "TAKEOFF<br/>"
        "HEIGHT&nbsp;1.5<br/>"
        "FB_IMU&nbsp;4 &nbsp;&nbsp;# chain N steps to cover most of the gap<br/>"
        "APPROACH&nbsp;&lt;enemy_face&gt;&nbsp;1.2<br/>"
        "HEIGHT&nbsp;1.8<br/>"
        "FB_IMU&nbsp;1.2 &nbsp;# drift exactly over the box<br/>"
        "HOOVER&nbsp;3<br/>"
        "YAW_IMU&nbsp;180<br/>"
        "FB_IMU&nbsp;4 &nbsp;&nbsp;# chain back home<br/>"
        "APPROACH&nbsp;&lt;home_marker&gt;&nbsp;2.0<br/>"
        "YAW_IMU&nbsp;180<br/>"
        "LAND"
    )
    return [
        PageBreak(),
        P("4.  Flight Control and Trajectory Planning", "H1"),
        P(
            "The Anafi exposes both a piloting (PCMD) interface for "
            "continuous roll/pitch/yaw/gaz commands and a discrete "
            "move-by interface that closes a position loop on the "
            "drone's own IMU and optical flow. Our mission engine "
            "treats these as complementary primitives. <b>IMU "
            "primitives</b> (<font face='Courier'>FB_IMU</font>, "
            "<font face='Courier'>LR_IMU</font>, "
            "<font face='Courier'>UD_IMU</font>, "
            "<font face='Courier'>YAW_IMU</font>) execute an exact "
            "bounded displacement and stop. <b>RC primitives</b> "
            "(<font face='Courier'>FB_RC</font>, "
            "<font face='Courier'>UD_RC</font>, etc.) pin a stick "
            "value for a duration and then auto-brake. <b>Vision "
            "primitives</b> (<font face='Courier'>APPROACH</font>, "
            "<font face='Courier'>FB_BRAKE</font>) home on a named "
            "ArUco marker's measured bearing and distance, which is "
            "drift-free regardless of the world-frame fix."
        ),
        P("4.1  The canonical attack manoeuvre", "H2"),
        P(
            "All test flights converge on the following composition, "
            "generated programmatically by the C2 dashboard for the "
            "selected target slot:"
        ),
        Paragraph(code, S["Code"]),
        P(
            "This pattern reflects three hard-won lessons. First, "
            "every horizontal displacement is either a bounded IMU "
            "step or a vision-homed APPROACH – never an open-loop "
            "RC cruise of non-trivial distance, because forward pitch "
            "tilts the camera downward and hides the destination "
            "marker, so the vision-tripped brake never fires and the "
            "drone reaches the timeout at full stick. Second, the "
            "over-the-box translation is a bounded "
            "<font face='Courier'>FB_IMU</font> sized to the APPROACH "
            "stand-off, so even with a noisy fix the drone stops above "
            "the target rather than continuing into the wall behind "
            "it. Third, the closing 180° rotation uses the "
            "IMU-closed-loop <font face='Courier'>YAW_IMU</font> rather "
            "than a yaw-rate stick; the RC variant rolled off in "
            "timing tests and sent the next leg sideways."
        ),
        P("4.2  Safety layers", "H2"),
        P(
            "Several overlays guard the runtime regardless of which "
            "script is loaded. A hard altitude ceiling clamps the gaz "
            "channel above a configurable height. A C2 watchdog forces "
            "an automatic land if the operator's heartbeat goes "
            "silent for longer than the regulation's two-second "
            "threshold (§3.3). A master kill-all command lands "
            "the entire fleet from a single key-press and is the "
            "responsibility of the team's Safety Officer per §2.2. "
            "Each drone in flight is shrouded in a spherical protective "
            "cage as recommended in §3.2. Crucially, no script "
            "step can bypass the kill-switch path, and the kill-switch "
            "is wired both physically and virtually so it remains "
            "actionable if any one component fails."
        ),
    ]


def section_strategy_and_conclusion() -> list:
    code = (
        "U(d, t) = value(t)<br/>"
        "&nbsp;&nbsp;− w_time&nbsp;·&nbsp;eta(d → t)<br/>"
        "&nbsp;&nbsp;− w_risk&nbsp;·&nbsp;risk(t)<br/>"
        "&nbsp;&nbsp;+ w_ready&nbsp;·&nbsp;readiness(d)<br/>"
        "&nbsp;&nbsp;+ w_hold&nbsp;·&nbsp;[t is the drone's current task]<br/>"
        "&nbsp;&nbsp;− w_switch&nbsp;·&nbsp;[d is mid-task and t differs]"
    )
    return [
        PageBreak(),
        P("5.  Swarm Strategy", "H1"),
        P(
            "Strategic decisions are made by a small <b>Commander</b> "
            "module that runs as part of the strategy server. It is "
            "deliberately rule-based and deterministic: each tick it "
            "inspects every drone's status (connection, battery, "
            "magnetometer calibration, current mission phase, position) "
            "and the live slot map (the holder of each of the six "
            "boxes, and how recently we have observed it), and produces "
            "a (role, target) assignment per drone that maximises a "
            "transparent utility function:"
        ),
        Paragraph(code, S["Code"]),
        P(
            "The <i>value</i> term encodes the regulations' scoring "
            "tiers directly. Recapturing an enemy-held box inside our "
            "own home zone is worth zero (the anti-farming rule of "
            "§1.4.3) and is filtered out of the task set entirely. "
            "A baseline capture is worth one game point. A capture in "
            "which every team drone is already outside the home zone "
            "at the moment of trigger is worth five and adds a "
            "coordination bonus. A pair of captures by two different "
            "drones inside one second is worth ten and is pursued "
            "opportunistically when two drones are simultaneously "
            "close to two different enemy boxes. The Special Maneuver "
            "– controlling all six boxes for at least five "
            "seconds – is treated as a terminal value and "
            "short-circuits role selection when it is within reach."
        ),
        P("5.1  Role set and asymmetric attack", "H2"),
        P(
            "The Commander assigns one of four roles per drone: "
            "<b>attacker</b> (drive a capture or recapture), "
            "<b>scout</b> (refresh stale slots in the enemy zone so "
            "the slot map remains current), <b>defender</b> (re-flip a "
            "recently-lost home box), and <b>return/idle</b> (low "
            "battery, lost link, or no useful work). The attack set is "
            "asymmetric per team – red drones may only target "
            "boxes 4–6 (the blue zone) and blue drones boxes "
            "1–3 – and the Commander rejects any operator "
            "override that violates the constraint. A staggered "
            "take-off schedule and per-slot airspace deconfliction "
            "prevent two drones from arriving at the same box."
        ),
        P("5.2  Open source and reproducibility", "H2"),
        P(
            "The entire stack is open-sourced at "
            "<font face='Courier'>github.com/otiedemann/sdc-tobe</font>: "
            "approximately 96,000 lines of code, of which 84,000 are "
            "Python. The mission DSL parser ships with a test harness; "
            "the strategy module, the flight controller, and the C2 "
            "dashboard share a single configuration file and a single "
            "venv; and every flight is recorded into a per-tick CSV "
            "plus an annotated MP4 that the in-browser replay viewer "
            "plays back synchronously."
        ),
        P("6.  Conclusion", "H1"),
        P(
            "Our design philosophy across every layer of the stack is "
            "the same: prefer drift-free, vision-relative homing and "
            "bounded closed-loop motion over absolute world-frame "
            "steering and open-loop cruises. This choice is forced by "
            "the realities of ArUco-only indoor positioning, and it is "
            "what kept us flying safely through the bug hunts that "
            "preceded this report. The remaining work – a "
            "position-sanity gate that lets the wall guard be safely "
            "re-enabled, full Commander auto-dispatch, and a defender "
            "role tuned to the recapture window – is scoped and "
            "queued, and will be delivered in time for the final at "
            "ILA Berlin on 11 June 2026."
        ),
    ]


def main():
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2.0 * cm,
        title="SDC26 Technical Report — Team ToBeDefined",
        author="Team ToBeDefined",
        subject="Swarm Drone Challenge 2026 — Technical Report",
    )
    story = []
    story += cover()
    story += section_intro_and_architecture()
    story += section_localisation()
    story += section_flight_control()
    story += section_strategy_and_conclusion()
    doc.build(story, canvasmaker=FooterCanvas)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
