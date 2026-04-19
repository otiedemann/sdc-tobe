import base64
import json
import logging
import math
import os
import platform
import socket
import threading
import time
from typing import Any, Dict, Optional

# Point Qt's font lookup at the standard dejavu location before importing cv2.
# Fixes: "Qt no longer ships fonts" warning on embedded Linux / Raspberry Pi.
# No-op on systems where the path doesn't exist.
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")

import cv2
import numpy as np
from cv2 import aruco

import aruco_position as aruco_pos
import prediction as predict
from fusion import fuse_delayed_vision_update
from motion_estimator import MotionEstimator, MotionState

import olympe

# --- CONFIGURATION ---
UDP_DEST_IP = "127.0.0.1"  # Default IP des Laptops (Relay)
UDP_PORT = 5005
UDP_CMD_PORT = 5006  # Port für eingehende Befehle vom Relay
CAMERA_SOURCE = 0
HEARTBEAT_INTERVAL = 1.0  # Sekunden: Status senden auch ohne Marker
TARGET_Z_POS = -1.5  # fixed target height position (internal Z axis)
MARKER_SIZE = 0.5

# Pose robustness settings
MIN_REF_WEIGHT = 0.00  # Ignore very weak refs, but keep detection usable
MIN_REF_COUNT = 1  # Allow single-marker pose as fallback
POSE_HOLD_SEC = 0.8  # Hold last valid pose briefly when refs drop out
OUTLIER_POS_THRESH = 2.5  # meters: looser outlier reject for real-world noise

# Motion model / delayed-measurement tuning
MAX_STATE_DT = 1.0
VEL_BLEND = 0.25
MEAS_BLEND_MIN = 0.35
MEAS_BLEND_MAX = 0.85

# ============================================================
# Optional imports for future modules
# ============================================================

try:
    import input as input_module
except Exception:
    input_module = None

try:
    import controller as ctrl_module
except Exception:
    ctrl_module = None

# Suppress olympe / arsdk / pdraw log noise (H264/AVCC decoder warnings, etc.)
logging.getLogger("olympe").setLevel(logging.CRITICAL)
logging.getLogger("ulog").setLevel(logging.CRITICAL)

# ============================================================
# Fallbacks
# ============================================================

def fallback_get_motion_input() -> Dict[str, Any]:
    """
    Erwartetes späteres Format aus input-Modul:
    {
        "timestamp": float,
        "vx_body": float,
        "vy_body": float,
        "vz_body": float,
        "yaw_rate": float,
    }
    """
    return {
        "timestamp": time.monotonic(),
        "vx_body": 0.0,
        "vy_body": 0.0,
        "vz_body": 0.0,
        "yaw_rate": 0.0,
    }


def estimate_velocity_from_history(history: list[MotionState]) -> tuple[float, float, float]:
    """
    Einfache Geschwindigkeitsabschätzung aus den letzten zwei Zuständen.
    """
    if len(history) < 2:
        return 0.0, 0.0, 0.0

    s0 = history[-2]
    s1 = history[-1]
    dt = s1.timestamp - s0.timestamp
    if dt <= 1e-6:
        return 0.0, 0.0, 0.0

    vx = (s1.x - s0.x) / dt
    vy = (s1.y - s0.y) / dt
    vz = (s1.z - s0.z) / dt
    return vx, vy, vz

def _is_anafi_source(camera_source):
    source = str(camera_source).strip().lower()
    return source in {"anafi", "parrot", "parrot-anafi"} or source.startswith("anafi:") or source.startswith("anafi://")


def _parse_anafi_ip(camera_source):
    source = str(camera_source).strip()
    source_lower = source.lower()
    if source_lower in {"anafi", "parrot", "parrot-anafi"}:
        return os.getenv("ANAFI_IP") or os.getenv("DRONE_IP") or "192.168.42.1"
    if source_lower.startswith("anafi://"):
        return source.split("://", 1)[1].strip() or "192.168.42.1"
    if source_lower.startswith("anafi:"):
        return source.split(":", 1)[1].strip() or "192.168.42.1"
    return "192.168.42.1"


def _anafi_flush_cb(stream):
    try:
        if hasattr(stream, "empty"):
            while not stream.empty():
                try:
                    stream.get(timeout=0.005).unref()
                except Exception:
                    break
        elif hasattr(stream, "get"):
            while True:
                try:
                    stream.get(timeout=0.005).unref()
                except Exception:
                    break
    except Exception:
        pass
    return True

def has_gui():
    system = platform.system().lower()
    if system == "windows":
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

# ============================================================
# Autonomous Mission
# ============================================================

TARGET_MARKER_ID = 15          # ArUco marker to find and approach
APPROACH_DISTANCE = 5.0        # metres to hover in front of the marker face
TAKEOFF_HEIGHT_Z = 1.2         # world-frame Z (up) to wait for after takeoff
TAKEOFF_TIMEOUT_S = 15.0       # abort takeoff after this many seconds
SEARCH_YAW_RATE = 0.3          # rad/s slow rotation during search
HOLD_ARRIVE_RADIUS = 0.25      # m – radius at which HOLD is declared

# Inward wall normals matching arena_config.json:
#   front wall  at y= 0  → normal +Y
#   back  wall  at y=10  → normal -Y
#   right wall  at x=10  → normal -X
#   left  wall  at x=-10 → normal +X
_WALL_NORMALS = {
    "front":  np.array([ 0.0,  1.0, 0.0]),
    "back":   np.array([ 0.0, -1.0, 0.0]),
    "left":   np.array([ 1.0,  0.0, 0.0]),
    "right":  np.array([-1.0,  0.0, 0.0]),
}


class MissionPhase:
    IDLE     = "idle"
    TAKEOFF  = "takeoff"
    SEARCH   = "search"
    APPROACH = "approach"
    HOLD     = "hold"
    FAILED   = "failed"
    ABORTED  = "aborted"


class AutonomousMission:
    """
    State machine that:
      1. Commands the drone to take off.
      2. Slowly rotates (SEARCH) until ArUco marker <target_marker_id> is seen.
      3. Flies to <approach_distance> metres in front of the marker at a
         90-degree angle (perpendicular to the marker face).
      4. Holds position indefinitely.

    Integration in main.py
    ----------------------
    After the prediction block and before the outgoing payload:

        mission.tick(last_prediction, last_vision_update, anafi_drone)

    The mission drives ctrl_module.set_target() internally and sends
    Olympe TakeOff / PCMD commands through the drone handle.
    """

    def __init__(
        self,
        target_marker_id: int,
        approach_distance: float,
        vision_processor,               # HeadlessAruCoPositioning instance
        ctrl_module,                    # controller module (may be None)
        takeoff_height_z: float = TAKEOFF_HEIGHT_Z,
        dry_run: bool = False,          # log RC/commands, send nothing to drone
        auto_confirm: bool = False,     # skip SPACE confirmations, advance automatically
    ) -> None:
        self.target_id        = target_marker_id
        self.approach_dist    = approach_distance
        self.vision           = vision_processor
        self.ctrl             = ctrl_module
        self.takeoff_height_z = takeoff_height_z
        self.dry_run          = dry_run
        self.auto_confirm     = auto_confirm

        self._phase: str               = MissionPhase.IDLE
        self._phase_start: float       = 0.0
        self._approach_target: Optional[tuple] = None  # (x, y, z, yaw)
        self._last_seen_ts: float      = 0.0
        self._search_yaw_sent: bool    = False
        self._pending_phase: Optional[str] = None   # next phase awaiting SPACE confirm
        self._pending_prompt: str      = ""          # message shown while waiting

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def start(self, drone) -> None:
        """Call once to kick off the mission (issues TakeOff)."""
        if self._phase != MissionPhase.IDLE:
            return
        print(f"[mission] Starting – target marker {self.target_id}")
        self._transition(MissionPhase.TAKEOFF)
        self._send_takeoff(drone)

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def waiting_for_confirm(self) -> bool:
        """True when the mission is paused and waiting for a SPACE keypress."""
        return self._pending_phase is not None

    @property
    def pending_prompt(self) -> str:
        return self._pending_prompt

    def confirm(self, drone=None) -> None:
        """Advance to the queued next phase (called when SPACE is pressed)."""
        if self._pending_phase is None:
            return
        phase = self._pending_phase
        # Kick off phase-specific actions before the transition.
        if phase == MissionPhase.SEARCH:
            self._begin_search(drone)
        elif phase == MissionPhase.APPROACH and self._approach_target is not None:
            x, y, z, yaw = self._approach_target
            if self.ctrl is not None:
                self.ctrl.set_target(x, y, z, hold_yaw=yaw)
        print(f"[mission] Confirmed → {phase}")
        self._transition(phase)

    def abort(self, drone) -> None:
        """
        Emergency abort: stop the mission, command the drone to land, and
        clear the position controller target.  Safe to call from any thread.
        """
        if self._phase in (MissionPhase.IDLE, MissionPhase.ABORTED):
            return
        print("\n🚨 [mission] ABORT – sending Landing command")
        self._transition(MissionPhase.ABORTED)
        if self.ctrl is not None:
            self.ctrl.clear_target()
        if self.dry_run:
            print("[DRY RUN] Landing skipped")
            return
        if drone is not None:
            try:
                from olympe.messages.ardrone3.Piloting import Landing
                drone(Landing())
                print("[mission] Landing command sent")
            except Exception as exc:
                print(f"[mission] Landing command failed: {exc}")

    def tick(
        self,
        state: Optional[dict],
        vision_result: Optional[dict],
        drone,
    ) -> None:
        """Call every control tick (25 Hz) after the prediction step.

        Position decisions use vision_result["cam"] exclusively.
        `state` (prediction) is only kept as a takeoff-altitude fallback
        when no ArUco markers are visible yet.
        """
        if self._phase in (MissionPhase.IDLE, MissionPhase.FAILED, MissionPhase.ABORTED):
            return

        # Paused: waiting for operator to press SPACE before the next step.
        if self._pending_phase is not None:
            return

        now = time.monotonic()

        if self._phase == MissionPhase.TAKEOFF:
            self._tick_takeoff(vision_result, state, now, drone)

        elif self._phase == MissionPhase.SEARCH:
            self._tick_search(vision_result, now, drone)

        elif self._phase == MissionPhase.APPROACH:
            self._tick_approach(vision_result, now, drone)

        elif self._phase == MissionPhase.HOLD:
            self._tick_hold(vision_result, now)

    # ------------------------------------------------------------------ #
    # Phase handlers                                                       #
    # ------------------------------------------------------------------ #

    def _tick_takeoff(self, vision_result, state_fallback, now, drone):
        elapsed = now - self._phase_start
        if elapsed > TAKEOFF_TIMEOUT_S:
            print("[mission] Takeoff timeout – aborting")
            self._transition(MissionPhase.FAILED)
            return

        # Prefer CAM-derived altitude; fall back to motion-estimator z when
        # no markers are visible yet (drone still on ground / spinning up).
        cam = self._get_cam_pos(vision_result)
        if cam is not None:
            z = cam[2]
        elif state_fallback is not None:
            z = float(state_fallback.get("z", 0.0))
        else:
            return

        if z >= self.takeoff_height_z * 0.7:
            self._queue_confirm(
                MissionPhase.SEARCH,
                f"Airborne – cam z={z:.2f} m.  Ready to start SEARCH.",
                drone,
            )

    def _tick_search(self, vision_result, now, drone):
        seen = self._seen_markers(vision_result)
        if self.target_id in seen:
            approach = self._compute_approach(vision_result)
            if approach is not None:
                self._approach_target = approach
                x, y, z, yaw = approach
                self._queue_confirm(
                    MissionPhase.APPROACH,
                    f"Marker {self.target_id} found – "
                    f"approach → ({x:.2f}, {y:.2f}, {z:.2f})  "
                    f"yaw={math.degrees(yaw):.1f}°.  Ready to APPROACH.",
                    drone,
                )
                return

        # Rotate in place: pure yaw PCMD only — no position controller target.
        if drone is not None:
            self._send_search_yaw(drone)

    def _tick_approach(self, vision_result, now, drone=None):
        if self._approach_target is None:
            self._transition(MissionPhase.SEARCH)
            return

        seen = self._seen_markers(vision_result)
        if self.target_id in seen:
            self._last_seen_ts = now

        # Primary: controller HOVER phase
        if self.ctrl is not None and hasattr(self.ctrl, "position_controller"):
            from controller import Phase
            if self.ctrl.position_controller.phase == Phase.HOVER:
                self._queue_confirm(
                    MissionPhase.HOLD,
                    "Arrived at approach point.  Ready to HOLD.",
                    drone,
                )
            return

        # Fallback: distance check using cam position
        cam = self._get_cam_pos(vision_result)
        if cam is not None and self._approach_target is not None:
            tx, ty, tz, _ = self._approach_target
            dist = math.sqrt((tx - cam[0])**2 + (ty - cam[1])**2 + (tz - cam[2])**2)
            if dist < HOLD_ARRIVE_RADIUS:
                self._queue_confirm(
                    MissionPhase.HOLD,
                    f"Arrived (cam dist={dist:.2f} m).  Ready to HOLD.",
                    drone,
                )

    def _tick_hold(self, vision_result, now):
        # Controller holds position; just track marker visibility.
        seen = self._seen_markers(vision_result)
        if self.target_id in seen:
            self._last_seen_ts = now
        elif now - self._last_seen_ts > 5.0:
            print(
                f"[mission] HOLD – marker {self.target_id} not seen for "
                f"{now - self._last_seen_ts:.1f} s"
            )

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _queue_confirm(self, next_phase: str, prompt: str, drone=None) -> None:
        """Pause mission and wait for SPACE before entering next_phase.

        When auto_confirm is True the pause is skipped and the mission
        transitions immediately (no operator key-press required).
        """
        if self._pending_phase == next_phase:
            return   # already queued – don't spam the console
        self._pending_phase = next_phase
        self._pending_prompt = prompt
        if self.auto_confirm:
            print(f"\n▶  [mission] auto-confirm → {next_phase.upper()}  ({prompt})")
            self.confirm(drone)
            return
        print(f"\n⏸  [mission] {prompt}")
        print(f"   → Press SPACE to continue to [{next_phase.upper()}]  (ESC to abort)")

    def _transition(self, phase: str) -> None:
        self._pending_phase = None
        self._pending_prompt = ""
        print(f"[mission] {self._phase} → {phase}")
        self._phase = phase
        self._phase_start = time.monotonic()

    def _send_takeoff(self, drone) -> None:
        if self.dry_run:
            print("[DRY RUN] TakeOff skipped – lift drone by hand")
            return
        if drone is None:
            print("[mission] No drone handle – skipping TakeOff command")
            return
        try:
            from olympe.messages.ardrone3.Piloting import TakeOff
            drone(TakeOff())
            print("[mission] TakeOff command sent")
        except Exception as exc:
            print(f"[mission] TakeOff failed: {exc}")

    _search_seq: int = 0   # class-level PCMD sequence counter for search

    def _send_search_yaw(self, drone) -> None:
        """Send a pure yaw PCMD — zero roll, pitch, and gaz — to rotate in place."""
        if self.dry_run:
            return   # drone is moved by hand; no PCMD
        try:
            from olympe.messages.ardrone3.Piloting import PCMD
            AutonomousMission._search_seq = (AutonomousMission._search_seq + 1) & 0x7FFFFFFF
            yaw_pct = int(SEARCH_YAW_RATE / 0.8 * 100)  # scale: 0.8 rad/s → 100 %
            drone(PCMD(0, 0, 0, yaw_pct, 0, AutonomousMission._search_seq))
        except Exception:
            pass

    @staticmethod
    def _get_cam_pos(vision_result: Optional[dict]) -> Optional[list]:
        """Return [x, y, z] from vision_result['cam'], or None if unavailable."""
        if vision_result is None:
            return None
        cam = vision_result.get("cam")
        if cam is None or len(cam) < 3:
            return None
        return [float(cam[0]), float(cam[1]), float(cam[2])]

    @staticmethod
    def _seen_markers(vision_result: Optional[dict]):
        if vision_result is None:
            return set()
        return set(int(m) for m in vision_result.get("seen_markers", []))

    def _begin_search(self, drone) -> None:
        print(f"[mission] SEARCH – yaw-rotating to find marker {self.target_id}")
        # Clear any active controller target so the position controller outputs
        # nothing and the pure-yaw PCMD is the only command sent to the drone.
        if self.ctrl is not None:
            self.ctrl.clear_target()

    def _compute_approach(self, vision_result: Optional[dict]) -> Optional[tuple]:
        """
        Return (x, y, z, yaw) for hovering APPROACH_DISTANCE metres straight in
        front of the target marker face.

        Primary: uses the marker's known world-frame position (from arena config)
        and the wall's inward normal to place the hover point perpendicular to the
        marker face.  Drone yaw is set so it faces toward the marker.

        Fallback (no arena data): fly forward from current cam position along the
        camera's forward direction (original behaviour).
        """
        # -- primary: marker world position from arena config --
        marker_pos = None
        wall_type  = None
        if self.vision is not None:
            mp = getattr(self.vision, "marker_positions", {})
            wt = getattr(self.vision, "marker_wall_type", {})
            if self.target_id in mp:
                marker_pos = mp[self.target_id]
                wall_type  = wt.get(self.target_id)

        if marker_pos is not None:
            # Inward wall normal (points from the wall face into the arena).
            normal = _WALL_NORMALS.get(wall_type, np.array([1.0, 0.0, 0.0]))
            nx, ny = float(normal[0]), float(normal[1])

            # Hover point: step approach_dist along the inward normal from the marker.
            target_x = float(marker_pos[0]) + nx * self.approach_dist
            target_y = float(marker_pos[1]) + ny * self.approach_dist
            target_z = float(marker_pos[2])   # keep marker's world height

            # Drone yaw: face back toward the marker (opposite of inward normal).
            yaw = math.atan2(-ny, -nx)

            print(
                f"[mission] Approach: marker {self.target_id} at "
                f"({marker_pos[0]:.2f}, {marker_pos[1]:.2f}, {marker_pos[2]:.2f})  "
                f"wall={wall_type} normal=({nx:+.2f},{ny:+.2f})  "
                f"hover → ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})  "
                f"yaw={math.degrees(yaw):.1f}°"
            )
            return (float(target_x), float(target_y), float(target_z), float(yaw))

        # -- fallback: no arena config data available --
        cam = self._get_cam_pos(vision_result)
        if cam is None:
            print("[mission] No cam position available – cannot compute approach")
            return None

        raw_dir = vision_result.get("dir") if vision_result is not None else None
        if raw_dir is not None and len(raw_dir) >= 2:
            dx, dy = float(raw_dir[0]), float(raw_dir[1])
        else:
            dx, dy = 1.0, 0.0

        mag = math.sqrt(dx * dx + dy * dy)
        if mag < 1e-6:
            dx, dy = 1.0, 0.0
        else:
            dx, dy = dx / mag, dy / mag

        target_x = cam[0] + dx * self.approach_dist
        target_y = cam[1] + dy * self.approach_dist
        target_z = cam[2]
        yaw      = math.atan2(dy, dx)

        print(
            f"[mission] Approach (fallback – no arena config): "
            f"fly {self.approach_dist} m forward from cam "
            f"({cam[0]:.2f}, {cam[1]:.2f}, {cam[2]:.2f})  "
            f"dir=({dx:+.2f}, {dy:+.2f})  "
            f"→ ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})"
        )
        return (float(target_x), float(target_y), float(target_z), float(yaw))


# ============================================================
# Main
# ============================================================

def main():
    import sys

    # --------------------------------------------------------
    # CLI args
    # --------------------------------------------------------
    camera_src = CAMERA_SOURCE
    target_ip = UDP_DEST_IP
    verbose_mode = False

    # Default: look for arena_config.json next to this script
    _default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arena_config.json")
    arena_config_path = _default_config if os.path.isfile(_default_config) else None

    if "--src" in sys.argv:
        try:
            src_val = sys.argv[sys.argv.index("--src") + 1]
            camera_src = int(src_val) if src_val.isdigit() else src_val
        except Exception:
            print("❌ Invalid source provided, using default.")

    if "--target-ip" in sys.argv:
        try:
            target_ip = sys.argv[sys.argv.index("--target-ip") + 1]
        except Exception:
            print("❌ Invalid target IP provided, using default.")

    if "--verbose" in sys.argv:
        verbose_mode = True

    if "--arena-config" in sys.argv:
        try:
            arena_config_path = sys.argv[sys.argv.index("--arena-config") + 1]
        except Exception:
            print("❌ Invalid --arena-config value, using default.")

    min_ref_weight = MIN_REF_WEIGHT
    min_ref_count = MIN_REF_COUNT
    outlier_pos_thresh = OUTLIER_POS_THRESH
    pose_hold_sec = POSE_HOLD_SEC
    target_z_pos = TARGET_Z_POS
    enable_kalman_filter = False

    if "--min-ref-weight" in sys.argv:
        try:
            min_ref_weight = float(sys.argv[sys.argv.index("--min-ref-weight") + 1])
        except Exception:
            print("⚠️ Invalid --min-ref-weight value, using default.")

    if "--min-ref-count" in sys.argv:
        try:
            min_ref_count = int(sys.argv[sys.argv.index("--min-ref-count") + 1])
        except Exception:
            print("⚠️ Invalid --min-ref-count value, using default.")

    if "--outlier-thresh" in sys.argv:
        try:
            outlier_pos_thresh = float(sys.argv[sys.argv.index("--outlier-thresh") + 1])
        except Exception:
            print("⚠️ Invalid --outlier-thresh value, using default.")

    if "--pose-hold" in sys.argv:
        try:
            pose_hold_sec = float(sys.argv[sys.argv.index("--pose-hold") + 1])
        except Exception:
            print("⚠️ Invalid --pose-hold value, using default.")

    if "--target-z-pos" in sys.argv:
        try:
            target_z_pos = float(sys.argv[sys.argv.index("--target-z-pos") + 1])
        except Exception:
            print("⚠️ Invalid --target-z-pos value, using default.")

    if '--pos-kalman' in sys.argv:
        enable_kalman_filter = True

    enable_motion_input = "--motion-input" in sys.argv

    mission_enabled  = "--mission" in sys.argv
    mission_dry_run  = "--dry-run" in sys.argv
    mission_auto_confirm = "--yes" in sys.argv or "-y" in sys.argv
    mission_marker_id = TARGET_MARKER_ID
    if "--mission-marker" in sys.argv:
        try:
            mission_marker_id = int(sys.argv[sys.argv.index("--mission-marker") + 1])
        except Exception:
            print("⚠️ Invalid --mission-marker value, using default.")

    # --ctrl-target x,y,z   e.g. --ctrl-target 3.0,1.5,1.2
    ctrl_target: Optional[tuple] = None
    if "--ctrl-target" in sys.argv:
        try:
            raw = sys.argv[sys.argv.index("--ctrl-target") + 1]
            parts = [float(v) for v in raw.split(",")]
            if len(parts) == 3:
                ctrl_target = (parts[0], parts[1], parts[2])
            else:
                print("⚠️ --ctrl-target expects x,y,z  e.g. 3.0,1.5,1.2")
        except Exception:
            print("⚠️ Invalid --ctrl-target value, controller disabled.")

    preview_requested = ("--preview" in sys.argv)
    gui_enabled = preview_requested and has_gui() and ("--force-headless" not in sys.argv)
    gui_available = True

    detect_profile = "balanced"
    if "--detect" in sys.argv:
        try:
            val = sys.argv[sys.argv.index("--detect") + 1].strip().lower()
            if val in ("sensitive", "balanced", "strict"):
                detect_profile = val
            else:
                print("⚠️ Unknown detect profile, using 'balanced'.")
        except Exception:
            print("⚠️ Missing value for --detect, using 'balanced'.")

    print(f"🚀 Node -> {target_ip}:{UDP_PORT} (Debug CMD on {UDP_CMD_PORT})")
    print(f"📷 Camera Source: {camera_src}")
    print(f"📝 Verbose Mode: {'ON' if verbose_mode else 'OFF'}")
    print(f"🔎 Detect Profile: {detect_profile}")
    print(
        f"⚙️ min_ref_weight={min_ref_weight} "
        f"min_ref_count={min_ref_count} "
        f"outlier={outlier_pos_thresh} "
        f"pose_hold={pose_hold_sec} "
        f"target_z_pos={target_z_pos}"
    )
    print(f"📉 Per-axis Kalman: {'ON' if enable_kalman_filter else 'OFF'}")
    print(f"📡 Motion Input: {'ON' if enable_motion_input else 'OFF'}")
    print(f"🎯 Ctrl Target: {ctrl_target if ctrl_target else 'none'}")
    print(f"🖥️ Preview Requested: {'YES' if preview_requested else 'NO'}")
    print(f"🖥️ GUI Overlay: {'ON' if gui_enabled else 'OFF'}")
    print(f"🎯 Autonomous Mission: {'ON (marker ' + str(mission_marker_id) + ')' if mission_enabled else 'OFF'}")
    if mission_dry_run:
        print("🧪 DRY RUN – no commands sent to drone; move it by hand")

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------
    cm = np.array(
        [[850.0, 0.0, 320.0],
         [0.0, 850.0, 240.0],
         [0.0, 0.0, 1.0]],
        dtype=float,
    )
    dc = np.zeros(5, dtype=float)

    if "--calib" in sys.argv:
        try:
            d = np.load(sys.argv[sys.argv.index("--calib") + 1])
            cm, dc = d["camera_matrix"], d["dist_coeffs"]
            print("✅ Calibration loaded.")
        except Exception:
            print("❌ Failed to load calibration.")

    # --------------------------------------------------------
    # Vision processor
    # --------------------------------------------------------
    vision_processor = aruco_pos.HeadlessAruCoPositioning(
        cm,
        dc,
        detect_profile=detect_profile,
        enable_kalman_filter=enable_kalman_filter,
        arena_config_path=arena_config_path,
        min_ref_weight=min_ref_weight,
        min_ref_count=max(1, min_ref_count),
        pose_hold_sec=max(0.0, pose_hold_sec),
        outlier_pos_thresh=max(0.1, outlier_pos_thresh),
        target_z_pos=target_z_pos,
    )

    # --------------------------------------------------------
    # Autonomous mission (optional)
    # --------------------------------------------------------
    mission: Optional[AutonomousMission] = None
    if mission_enabled:
        mission = AutonomousMission(
            target_marker_id=mission_marker_id,
            approach_distance=APPROACH_DISTANCE,
            vision_processor=vision_processor,
            ctrl_module=ctrl_module,
            takeoff_height_z=abs(target_z_pos),   # use configured flight height
            dry_run=mission_dry_run,
            auto_confirm=mission_auto_confirm,
        )

    # --------------------------------------------------------
    # Mission log (JSONL, one entry per control tick)
    # --------------------------------------------------------
    _log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"mission_{time.strftime('%Y%m%d_%H%M%S')}.jsonl",
    )
    _log_file = open(_log_path, "w", encoding="utf-8")
    print(f"📋 Mission log → {_log_path}")

    # --------------------------------------------------------
    # Rates
    # --------------------------------------------------------
    motion_rate_hz = 25.0
    vision_rate_hz = 10.0

    motion_dt = 1.0 / motion_rate_hz
    vision_dt = 1.0 / vision_rate_hz

    # --------------------------------------------------------
    # Motion estimator
    # --------------------------------------------------------
    motion_estimator = MotionEstimator(
        history_seconds=2.0,
        nominal_dt=motion_dt,
        initial_pose=(0.0, 0.0, 0.0, 0.0),
        initial_variance=(0.01, 0.01, 0.01, 0.01),
        yaw_positive_is_ccw=True,
    )

    last_motion_state: Optional[MotionState] = motion_estimator.get_current_state()
    last_vision_update: Optional[Dict[str, Any]] = None
    pending_vision_update: Optional[Dict[str, Any]] = None
    last_fusion_result: Optional[Dict[str, Any]] = None
    _last_rc: Optional[Dict] = None

    # --------------------------------------------------------
    # Video source setup
    # --------------------------------------------------------
    use_anafi_stream = _is_anafi_source(camera_src)

    anafi_drone = None
    anafi_stream_api = None
    anafi_frame_state = {"frame": None}
    anafi_frame_lock = threading.Lock()

    cap = None

    motion_listener = None
    if use_anafi_stream:
        if olympe is None:
            raise RuntimeError("Parrot Olympe not installed. Install with: pip install parrot-olympe")

        anafi_ip = _parse_anafi_ip(camera_src)


        print(f"🛩️ Connecting to Parrot Anafi at {anafi_ip}…")
        anafi_drone = olympe.Drone(anafi_ip)

        if enable_motion_input and input_module is not None:
            motion_listener = input_module.MotionListener(anafi_drone)
            motion_listener.subscribe()

        anafi_drone.connect()


        def _anafi_frame_cb(yuv_frame):
            try:
                yuv_frame.ref()
            except Exception:
                pass

            try:
                cv_cvt = cv2.COLOR_YUV2BGR_I420
                try:
                    info = yuv_frame.info()
                    yuv_fmt = None

                    if isinstance(info, dict):
                        if "yuv" in info and isinstance(info["yuv"], dict):
                            yuv_fmt = info["yuv"].get("format")
                        elif "format" in info:
                            yuv_fmt = info["format"]
                        elif "raw" in info and isinstance(info["raw"], dict):
                            yuv_fmt = info["raw"].get("format")

                    if yuv_fmt is not None:
                        for attr, flag in (
                            ("VDEF_I420", cv2.COLOR_YUV2BGR_I420),
                            ("VDEF_NV12", cv2.COLOR_YUV2BGR_NV12),
                        ):
                            fmt_const = getattr(olympe, attr, None)
                            if fmt_const is not None and fmt_const == yuv_fmt:
                                cv_cvt = flag
                                break
                except Exception:
                    pass

                frame = cv2.cvtColor(yuv_frame.as_ndarray(), cv_cvt)
                with anafi_frame_lock:
                    anafi_frame_state["frame"] = frame
            finally:
                try:
                    yuv_frame.unref()
                except Exception:
                    pass

        if hasattr(anafi_drone, "streaming") and hasattr(anafi_drone.streaming, "set_callbacks"):
            anafi_drone.streaming.set_callbacks(raw_cb=_anafi_frame_cb, flush_raw_cb=_anafi_flush_cb)
            anafi_drone.streaming.start()
            anafi_stream_api = "modern"
        elif hasattr(anafi_drone, "set_streaming_callbacks"):
            anafi_drone.set_streaming_callbacks(raw_cb=_anafi_frame_cb)
            anafi_drone.start_video_streaming()
            anafi_stream_api = "legacy"
        else:
            raise RuntimeError("No compatible Olympe streaming API found")

        time.sleep(0.2)
        print("✅ Anafi videostream active")

        if ctrl_target is not None and ctrl_module is not None:
            ctrl_module.set_target(*ctrl_target)

        if mission is not None:
            mission.start(anafi_drone)

    else:
        cap = cv2.VideoCapture(camera_src)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # --------------------------------------------------------
    # UDP
    # --------------------------------------------------------
    sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_cmd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_cmd.bind(("0.0.0.0", UDP_CMD_PORT))
    sock_cmd.setblocking(False)

    # --------------------------------------------------------
    # Emergency abort flag + keyboard listener (headless)
    # Trigger: press ESC in the GUI window, send {"abort":true}
    # via UDP, or press ESC / 'q' in the terminal (headless).
    # --------------------------------------------------------
    _abort_flag   = threading.Event()
    _confirm_flag = threading.Event()

    # Save terminal state at main-thread level so we can restore it even if
    # the keyboard thread is killed before its own finally block runs.
    _saved_term = None
    if platform.system().lower() != "windows":
        try:
            import termios as _termios
            _saved_term = _termios.tcgetattr(sys.stdin.fileno())
        except Exception:
            pass   # stdin not a tty (piped / CI)

    def _keyboard_listener():
        """Non-blocking terminal key listener for headless / no-GUI mode.

        SPACE  → confirm next mission step (_confirm_flag)
        ESC/q  → emergency abort         (_abort_flag)

        Uses msvcrt on Windows (no terminal-mode changes needed).
        On POSIX, switches to cbreak mode (single-char reads without echo)
        then restores the saved settings on exit.  The main finally block
        also restores settings as a guaranteed backstop.
        """
        try:
            if platform.system().lower() == "windows":
                import msvcrt
                while not _abort_flag.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch in (b'\x1b', b'q', b'Q'):
                            print("\n[abort] ESC / q detected in terminal")
                            _abort_flag.set()
                            return
                        if ch == b' ':
                            print("\n[mission] SPACE – confirm")
                            _confirm_flag.set()
                    time.sleep(0.05)
            else:
                import select
                import sys as _sys
                import tty
                import termios
                fd = _sys.stdin.fileno()
                try:
                    tty.setcbreak(fd)   # cbreak: chars available immediately, Ctrl+C still works
                    while not _abort_flag.is_set():
                        r, _, _ = select.select([_sys.stdin], [], [], 0.1)
                        if r:
                            ch = _sys.stdin.read(1)
                            if ch in ('\x1b', 'q', 'Q'):
                                print("\n[abort] ESC / q detected in terminal")
                                _abort_flag.set()
                                return
                            if ch == ' ':
                                print("\n[mission] SPACE – confirm")
                                _confirm_flag.set()
                finally:
                    if _saved_term is not None:
                        try:
                            termios.tcsetattr(fd, termios.TCSADRAIN, _saved_term)
                        except Exception:
                            pass
        except Exception:
            pass   # stdin not a tty (piped / background / non-interactive SSH)

    # Non-daemon so Python waits for it to exit (and restore the terminal)
    # before shutting down.  We signal it via _abort_flag in the finally block.
    _kb_thread = threading.Thread(target=_keyboard_listener, name="kb_listener", daemon=False)
    _kb_thread.start()

    debug_mode = False
    last_send_time = 0.0
    last_img_time = 0.0
    last_heartbeat_time = 0.0
    last_motion_sample: Optional[Dict] = None

    next_motion_time = time.monotonic()
    next_vision_time = time.monotonic()

    def read_frame():
        if use_anafi_stream:
            with anafi_frame_lock:
                frame = anafi_frame_state.get("frame")
                if frame is not None:
                    frame = frame.copy()
            return (frame is not None), frame

        if cap is not None:
            ret, frame = cap.read()
            return ret, frame

        return False, None

    try:
        while True:
            now = time.monotonic()

            # --------------------------------------
            # Debug command socket
            # --------------------------------------
            try:
                data, _ = sock_cmd.recvfrom(1024)
                cmd = json.loads(data.decode())
                if "debug" in cmd:
                    debug_mode = bool(cmd["debug"])
                if cmd.get("abort"):
                    print("[abort] Abort received via UDP command")
                    _abort_flag.set()
            except BlockingIOError:
                pass
            except Exception:
                pass

            # --------------------------------------
            # Emergency abort check
            # --------------------------------------
            if _abort_flag.is_set() and mission is not None and mission.phase not in (
                MissionPhase.ABORTED, MissionPhase.IDLE
            ):
                mission.abort(anafi_drone)

            # --------------------------------------
            # Manual SPACE confirm
            # --------------------------------------
            if _confirm_flag.is_set():
                _confirm_flag.clear()
                if mission is not None and mission.waiting_for_confirm:
                    mission.confirm(drone=anafi_drone)

            # --------------------------------------
            # Motion update @ 25 Hz
            # --------------------------------------
            if now >= next_motion_time:

                if enable_motion_input and motion_listener is not None:
                    last_motion_sample = motion_listener.get_motion_input()
                else:
                    last_motion_sample = fallback_get_motion_input()

                # ts = float(last_motion_sample.get("timestamp", now))
                # vx_body = float(last_motion_sample.get("vx_body", 0.0))
                # vy_body = float(last_motion_sample.get("vy_body", 0.0))
                # vz_body = float(last_motion_sample.get("vz_body", 0.0))
                # yaw_rate = float(last_motion_sample.get("yaw_rate", 0.0))
                #
                # last_motion_state = motion_estimator.update_body_frame(
                #     timestamp=ts,
                #     vx_body=vx_body,
                #     vy_body=vy_body,
                #     vz_body=vz_body,
                #     yaw_rate=yaw_rate,
                # )

                next_motion_time += motion_dt
                if now - next_motion_time > motion_dt:
                    next_motion_time = now + motion_dt

            # --------------------------------------
            # Vision update @ 5 Hz
            # --------------------------------------
            current_frame = None

            if now >= next_vision_time:
                ret, current_frame = read_frame()

                if ret and current_frame is not None:
                    result = vision_processor.process_frame(
                        current_frame,
                        frame_ts=now,
                        now_ts=now,
                    )

                    if result is None:
                        result = {"cam": None, "dir": None, "targets": {}}

                    result["timestamp"] = now
                    result["debug"] = debug_mode

                    last_vision_update = result

                    # Nur wirklich neue verwertbare Vision-Updates in die Fusion geben
                    if result.get("cam") is not None:
                        pending_vision_update = result

                next_vision_time += vision_dt
                if now - next_vision_time > vision_dt:
                    next_vision_time = now + vision_dt

            # Falls wir für Preview/Debug ein Frame brauchen
            if current_frame is None and (gui_enabled or debug_mode):
                ret, current_frame = read_frame()
                if not ret:
                    current_frame = None

            # --------------------------------------
            # Fusion nur bei neuem Vision-Update
            # --------------------------------------
            if pending_vision_update is not None:
                last_fusion_result = fuse_delayed_vision_update(
                    motion_history=motion_estimator.get_history(),
                    vision_update=pending_vision_update,
                    use_yaw_if_available=True,
                    prefer_repropagation=True,
                )

                updated_history = last_fusion_result.get("updated_history")
                if updated_history:
                    last_motion_state = motion_estimator.apply_fused_history(updated_history)

                pending_vision_update = None

            # --------------------------------------
            # Aktuellen gefuseten Zustand aufbauen
            # --------------------------------------
            current_state = motion_estimator.get_current_state()
            history = motion_estimator.get_history()
            est_vx, est_vy, est_vz = estimate_velocity_from_history(history)

            last_fused_state = {
                "timestamp": current_state.timestamp,
                "x": current_state.x,
                "y": current_state.y,
                "z": current_state.z,
                "yaw": current_state.yaw,
                "vx": est_vx,
                "vy": est_vy,
                "vz": est_vz,
                "var_x": current_state.var_x,
                "var_y": current_state.var_y,
                "var_z": current_state.var_z,
                "var_yaw": current_state.var_yaw,
                "source": "fusion" if last_vision_update is not None else "motion_only",
                "history": [
                    {
                        "timestamp": s.timestamp,
                        "x": s.x,
                        "y": s.y,
                        "z": s.z,
                        "yaw": s.yaw,
                    }
                    for s in history
                ],
            }

            # --------------------------------------
            # Prediction auf jetzt
            # --------------------------------------
            last_prediction = predict.predict_to_now(
                fused_state=last_fused_state,
                now_ts=now,
            )

            # --------------------------------------
            # Drone Position Controller  (cam coordinates only)
            # --------------------------------------

            _ctrl_state = None
            if last_vision_update is not None:
                _cam = last_vision_update.get("cam")
                _dir = last_vision_update.get("dir")
                if _cam is not None and len(_cam) >= 3:
                    _yaw = math.atan2(_dir[1], _dir[0]) if (_dir is not None and len(_dir) >= 2) else 0.0
                    _ctrl_state = {
                        "x":       float(_cam[0]),
                        "y":       float(_cam[1]),
                        "z":       float(_cam[2]),
                        "yaw":     _yaw,
                        "vx":      0.0,
                        "vy":      0.0,
                        "vz":      0.0,
                        "std_pos": 0.0,
                    }

            if ctrl_module is not None and _ctrl_state is not None:
                rc = ctrl_module.update(_ctrl_state)
                if rc is not None:
                    _last_rc = rc
                    if mission_dry_run:
                        print(
                            f"\r[DRY RUN] RC  fb={rc['forward_back']:+4d}  "
                            f"lr={rc['left_right']:+4d}  "
                            f"ud={rc['up_down']:+4d}  "
                            f"yaw={rc['yaw']:+4d}  | "
                            f"phase={rc.get('_phase','?')}  "
                            f"err_xy={rc.get('_err_xy',0):.2f}m  "
                            f"err_z={rc.get('_err_z',0):+.2f}m  "
                            f"yaw_err={rc.get('_yaw_err_deg',0):+.1f}°"
                            f"  cam=({_ctrl_state['x']:+.2f},{_ctrl_state['y']:+.2f},{_ctrl_state['z']:+.2f})",
                            end="",
                        )
                    elif anafi_drone is not None:
                        ctrl_module.send_pcmd_olympe(anafi_drone, rc)

            # --------------------------------------
            # Autonomous mission tick
            # --------------------------------------
            if mission is not None:
                mission.tick(last_prediction, last_vision_update, anafi_drone)

            # --------------------------------------
            # Mission log entry (written every tick the controller is active)
            # --------------------------------------
            if _last_rc is not None or mission is not None:
                _log_cam = None
                if _ctrl_state is not None:
                    _log_cam = {
                        "x": round(_ctrl_state["x"], 4),
                        "y": round(_ctrl_state["y"], 4),
                        "z": round(_ctrl_state["z"], 4),
                        "yaw_deg": round(math.degrees(_ctrl_state["yaw"]), 2),
                    }

                _log_tgt = None
                if mission is not None and mission._approach_target is not None:
                    tx, ty, tz, tyaw = mission._approach_target
                    _log_tgt = {
                        "x": round(tx, 4),
                        "y": round(ty, 4),
                        "z": round(tz, 4),
                        "yaw_deg": round(math.degrees(tyaw), 2),
                    }
                elif ctrl_module is not None and ctrl_module.position_controller.target is not None:
                    t = ctrl_module.position_controller.target
                    _log_tgt = {
                        "x": round(t.x, 4),
                        "y": round(t.y, 4),
                        "z": round(t.z, 4),
                        "yaw_deg": round(math.degrees(t.hold_yaw), 2) if t.hold_yaw is not None else None,
                    }

                _log_entry = {
                    "ts": round(time.time(), 4),
                    "mission_phase": mission.phase if mission is not None else None,
                    "cam": _log_cam,
                    "target": _log_tgt,
                    "ctrl_phase": _last_rc.get("_phase") if _last_rc else None,
                    "err_xy": _last_rc.get("_err_xy") if _last_rc else None,
                    "err_z": _last_rc.get("_err_z") if _last_rc else None,
                    "yaw_err_deg": _last_rc.get("_yaw_err_deg") if _last_rc else None,
                    "rc": {
                        "forward_back": _last_rc["forward_back"],
                        "left_right":   _last_rc["left_right"],
                        "up_down":      _last_rc["up_down"],
                        "yaw":          _last_rc["yaw"],
                    } if _last_rc else None,
                }
                _log_file.write(json.dumps(_log_entry) + "\n")

            # --------------------------------------
            # Outgoing payload
            # Alte Daten beibehalten + neue ergänzen
            # --------------------------------------
            result = {"cam": None, "dir": None, "targets": {}, "debug": debug_mode}

            if last_vision_update is not None:
                result.update(last_vision_update)
                result["debug"] = debug_mode

            if last_motion_state is not None:
                result["motion"] = {
                    "x": last_motion_state.x,
                    "y": last_motion_state.y,
                    "z": last_motion_state.z,
                    "yaw": last_motion_state.yaw,
                    "var_x": last_motion_state.var_x,
                    "var_y": last_motion_state.var_y,
                    "var_z": last_motion_state.var_z,
                    "var_yaw": last_motion_state.var_yaw,
                    "timestamp": last_motion_state.timestamp,
                }

            if last_fused_state is not None:
                result["fused"] = {
                    "x": last_fused_state["x"],
                    "y": last_fused_state["y"],
                    "z": last_fused_state["z"],
                    "yaw": last_fused_state["yaw"],
                    "vx": last_fused_state["vx"],
                    "vy": last_fused_state["vy"],
                    "vz": last_fused_state["vz"],
                    "timestamp": last_fused_state["timestamp"],
                    "source": last_fused_state["source"],
                }

            if last_prediction is not None:
                result["pred"] = {
                    "x": last_prediction["x"],
                    "y": last_prediction["y"],
                    "z": last_prediction["z"],
                    "yaw": last_prediction["yaw"],
                    "vx": last_prediction["vx"],
                    "vy": last_prediction["vy"],
                    "vz": last_prediction["vz"],
                    "std_x": last_prediction["std_x"],
                    "std_y": last_prediction["std_y"],
                    "std_z": last_prediction["std_z"],
                    "std_pos": last_prediction["std_pos"],
                    "timestamp": last_prediction["timestamp"],
                    "source": last_prediction["source"],
                    "method": last_prediction["method"],
                }

            if last_fusion_result is not None:
                result["fusion"] = {
                    "reference_index": last_fusion_result.get("reference_index"),
                    "correction": last_fusion_result.get("correction"),
                    "innovation": last_fusion_result.get("innovation"),
                }

            if ctrl_module is not None:
                result["ctrl"] = ctrl_module.position_controller.status()

            if mission is not None:
                result["mission"] = {
                    "phase": mission.phase,
                    "target_marker": mission.target_id,
                    "approach_target": list(mission._approach_target) if mission._approach_target else None,
                    "waiting_confirm": mission.waiting_for_confirm,
                    "confirm_prompt": mission.pending_prompt if mission.waiting_for_confirm else None,
                }

            # --------------------------------------
            # Local preview
            # --------------------------------------
            if gui_enabled and gui_available and current_frame is not None:
                preview = current_frame.copy()
                gray = cv2.cvtColor(preview, cv2.COLOR_BGR2GRAY)
                c_dbg, i_dbg, _ = vision_processor.detector.detectMarkers(gray)
                if i_dbg is not None:
                    aruco.drawDetectedMarkers(preview, c_dbg, i_dbg)

                cv2.putText(
                    preview,
                    f"Debug: {'ON' if debug_mode else 'OFF'}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cam = result.get("cam")
                if cam is not None:
                    _dir = result.get("dir")
                    _cam_yaw_deg = math.degrees(math.atan2(_dir[1], _dir[0])) if _dir is not None else 0.0
                    cv2.putText(
                        preview,
                        f"CAM x={cam[0]:+.3f} y={cam[1]:+.3f} z={cam[2]:+.3f} yaw={_cam_yaw_deg:+.1f}°",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 255),
                        2,
                    )

                if last_prediction is not None:
                    txt1 = (
                        f"PRED x={last_prediction['x']:+.2f} "
                        f"y={last_prediction['y']:+.2f} "
                        f"z={last_prediction['z']:+.2f}"
                    )
                    txt2 = (
                        f"{last_prediction['method']} | "
                        f"std={last_prediction['std_pos']:.2f} m"
                    )

                    cv2.putText(
                        preview,
                        txt1,
                        (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        preview,
                        txt2,
                        (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                if last_motion_sample is not None:
                    timestamp = last_motion_sample.get("timestamp", 0.0)
                    vx = last_motion_sample.get("vx_body", 0.0)
                    vy = last_motion_sample.get("vy_body", 0.0)
                    vz = last_motion_sample.get("vz_body", 0.0)
                    yr = last_motion_sample.get("yaw_rate", 0.0)
                    cv2.putText(
                        preview,
                        f"MOT vx={vx:+.3f} vy={vy:+.3f} vz={vz:+.3f} yr={yr:+.3f} time={timestamp}",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 180, 0),
                        2,
                    )

                if _last_rc is not None:
                    rc_txt = (
                        f"RC  fb={_last_rc['forward_back']:+4d}  "
                        f"lr={_last_rc['left_right']:+4d}  "
                        f"ud={_last_rc['up_down']:+4d}  "
                        f"yaw={_last_rc['yaw']:+4d}  "
                        f"| err_xy={_last_rc.get('_err_xy',0):.2f}m  "
                        f"err_z={_last_rc.get('_err_z',0):+.2f}m  "
                        f"desired_yaw={_last_rc.get('_desired_yaw',0):+.1f} deg  "
                        f"yaw_err={_last_rc.get('_yaw_err_deg',0):+.1f} deg  "
                        f"phase={_last_rc.get('_phase')}"
                    )
                    cv2.putText(
                        preview,
                        rc_txt,
                        (10, 170),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 220, 255),
                        2,
                    )

                if mission is not None and mission._approach_target is not None:
                    tx, ty, tz, tyaw = mission._approach_target
                    cv2.putText(
                        preview,
                        f"TGT x={tx:+.2f} y={ty:+.2f} z={tz:+.2f} yaw={math.degrees(tyaw):+.1f} deg",
                        (10, 195),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 100, 255),
                        2,
                    )

                if mission is not None:
                    phase_color = (0, 255, 0) if mission.phase == MissionPhase.HOLD else \
                                  (0, 0, 255) if mission.phase == MissionPhase.ABORTED else \
                                  (0, 165, 255)
                    if mission.waiting_for_confirm:
                        overlay_txt = f"⏸ {mission.pending_prompt}  [SPACE=confirm  ESC=abort]"
                        phase_color = (0, 255, 255)   # yellow while waiting
                    else:
                        overlay_txt = f"MISSION: {mission.phase.upper()}  [ESC=abort]"
                    cv2.putText(
                        preview,
                        overlay_txt,
                        (10, preview.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        phase_color,
                        2,
                    )

                try:
                    cv2.imshow("aruco_position Preview", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == 27:   # ESC
                        print("[abort] ESC pressed in preview window")
                        _abort_flag.set()
                    if key == 32:   # SPACE
                        _confirm_flag.set()
                except cv2.error:
                    gui_available = False
                    print("\n⚠️ OpenCV HighGUI not available. Disabling local preview window.")

            # --------------------------------------
            # Debug image (max 10 FPS)
            # --------------------------------------
            if debug_mode and current_frame is not None and now - last_img_time > 0.1:
                small = cv2.resize(current_frame, (320, 240))
                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 40])
                result["img"] = base64.b64encode(buf).decode()
                last_img_time = now

            # --------------------------------------
            # Send data
            # --------------------------------------
            should_send_tracking = (result.get("cam") is not None) and (now - last_send_time > 0.03)
            should_send_debug = debug_mode and (now - last_send_time > 0.03)
            should_send_heartbeat = (now - last_heartbeat_time > HEARTBEAT_INTERVAL)

            if should_send_tracking or should_send_debug or should_send_heartbeat:
                sock_send.sendto(json.dumps(result).encode(), (target_ip, UDP_PORT))

                if should_send_tracking or should_send_debug:
                    last_send_time = now
                if should_send_heartbeat:
                    last_heartbeat_time = now

                if verbose_mode and result.get("cam") is not None and result.get("dir") is not None:
                    cam = result["cam"]
                    dirv = result["dir"]

                    targets_txt = ""
                    if result.get("targets"):
                        parts = []
                        for tid, tpos in result["targets"].items():
                            parts.append(f"T{tid}:[{tpos[0]:+.2f},{tpos[1]:+.2f},{tpos[2]:+.2f}]")
                        targets_txt = " | " + " ".join(parts)

                    marker_txt = ""
                    if "ref_markers" in result and "marker_weights" in result:
                        marker_parts = []
                        for mid in result["ref_markers"]:
                            w = result["marker_weights"].get(str(mid), 0.0)
                            marker_parts.append(f"M{mid}:{w:.3f}")
                        marker_txt = " | REF: " + " ".join(marker_parts)

                    pred_txt = ""
                    if result.get("pred") is not None:
                        p = result["pred"]
                        pred_txt = f" | PRED:[{p['x']:+.2f},{p['y']:+.2f},{p['z']:+.2f}]"

                    print(
                        f"\rCAM: [{cam[0]:+.3f}, {cam[1]:+.3f}, {cam[2]:+.3f}] "
                        f"DIR: [{dirv[0]:+.3f}, {dirv[1]:+.3f}, {dirv[2]:+.3f}] "
                        f"Targets: {len(result.get('targets', {}))} "
                        f"Debug: {'ON' if debug_mode else 'OFF'}"
                        f"{marker_txt}{targets_txt}{pred_txt}",
                        end="",
                    )
                else:
                    tgt_count = len(result["targets"]) if result.get("targets") else 0
                    src = result.get("pred", {}).get("source", "-")
                    print(
                        f"\rTracking: {tgt_count} Targets | Debug: {'Yes' if debug_mode else 'No'} | PredSrc: {src}",
                        end="",
                    )

            time.sleep(0.001)

    finally:
        try:
            _log_file.close()
            print(f"\n📋 Mission log saved → {_log_path}")
        except Exception:
            pass

        if cap is not None:
            cap.release()

        if anafi_drone is not None:
            try:
                if anafi_stream_api == "modern":
                    anafi_drone.streaming.stop()
                else:
                    anafi_drone.stop_video_streaming()
            except Exception:
                try:
                    anafi_drone.streaming.stop()
                except Exception:
                    try:
                        anafi_drone.stop_video_streaming()
                    except Exception:
                        pass
            try:
                if motion_listener is not None:
                    motion_listener.unsubscribe()
                anafi_drone.disconnect()
            except Exception:
                pass

        sock_send.close()
        sock_cmd.close()

        if gui_enabled and gui_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

        # Signal the keyboard listener to exit its loop, then wait for it to
        # finish so its finally block can restore the terminal before we return.
        _abort_flag.set()
        _kb_thread.join(timeout=1.0)

        # Backstop: restore terminal settings from the main thread in case the
        # keyboard thread was killed before it could run its own restore.
        if _saved_term is not None:
            try:
                import termios as _termios
                _termios.tcsetattr(sys.stdin.fileno(), _termios.TCSADRAIN, _saved_term)
            except Exception:
                pass


if __name__ == "__main__":
    main()
