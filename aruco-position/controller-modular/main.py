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
DRY_RUN = False  # True: log RC/commands but send nothing to drone, no takeoff

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

TAKEOFF_HEIGHT_Z = 1.2         # world-frame Z (up) to wait for after takeoff
TAKEOFF_TIMEOUT_S = 15.0       # abort takeoff after this many seconds
HOLD_ARRIVE_RADIUS = 0.25      # m – radius at which HOLD is declared
CAM_LOST_TIMEOUT_S = 2.0       # seconds without cam position before entering recovery yaw
RECOVERY_YAW_RATE = 0.1        # rad/s slow yaw rotation during recovery
RECOVERY_MAX_YAW_SPEED = 150   # °/s – MaxRotationSpeed sent to drone before recovery PCMD


class MissionPhase:
    IDLE    = "idle"
    TAKEOFF = "takeoff"
    FLY_TO  = "fly_to"
    HOLD    = "hold"
    FAILED  = "failed"
    ABORTED = "aborted"


class AutonomousMission:
    """
    State machine that:
      1. Commands the drone to take off.
      2. Flies to a given (x, y, z) world-frame coordinate.
      3. Holds position indefinitely.

    Integration in main.py
    ----------------------
    After the prediction block and before the outgoing payload:

        mission.tick(last_prediction, last_vision_update, anafi_drone)

    The mission drives ctrl_module.set_target() internally and sends
    Olympe TakeOff commands through the drone handle.
    """

    def __init__(
        self,
        fly_to: tuple,                  # (x, y, z) world-frame target coordinate
        ctrl_module,                    # controller module (may be None)
        takeoff_height_z: float = TAKEOFF_HEIGHT_Z,
        dry_run: bool = False,          # log RC/commands, send nothing to drone
        auto_confirm: bool = False,     # skip SPACE confirmations, advance automatically
    ) -> None:
        self.fly_to           = fly_to  # (x, y, z)
        self.ctrl             = ctrl_module
        self.takeoff_height_z = takeoff_height_z
        self.dry_run          = dry_run
        self.auto_confirm     = auto_confirm

        self._phase: str               = MissionPhase.IDLE
        self._phase_start: float       = 0.0
        self._pending_phase: Optional[str] = None   # next phase awaiting SPACE confirm
        self._pending_prompt: str      = ""          # message shown while waiting

        # FLY_TO recovery state
        self._last_cam_ts: float       = 0.0         # monotonic timestamp of last valid cam pos
        self._recovering: bool         = False        # True while yaw-searching for markers
        self._recovery_seq: int        = 0            # PCMD sequence counter during recovery

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def start(self, drone) -> None:
        """Call once to kick off the mission (issues TakeOff)."""
        if self._phase != MissionPhase.IDLE:
            return
        x, y, z = self.fly_to
        print(f"[mission] Starting – fly to ({x:.2f}, {y:.2f}, {z:.2f})")
        if self.dry_run:
            print("[DRY RUN] Takeoff skipped – jumping straight to FLY_TO")
            self._begin_fly_to(drone)
            self._transition(MissionPhase.FLY_TO)
            return
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
        if phase == MissionPhase.FLY_TO:
            self._begin_fly_to(drone)
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

        elif self._phase == MissionPhase.FLY_TO:
            self._tick_fly_to(vision_result, now, drone)

        elif self._phase == MissionPhase.HOLD:
            self._tick_hold(now)

    # ------------------------------------------------------------------ #
    # Phase handlers                                                       #
    # ------------------------------------------------------------------ #

    def _tick_takeoff(self, vision_result, state_fallback, now, drone):
        elapsed = now - self._phase_start
        if elapsed > TAKEOFF_TIMEOUT_S:
            print("[mission] Takeoff timeout – aborting")
            self._transition(MissionPhase.FAILED)
            return

        if drone is None:
            return

        try:
            from olympe.messages.ardrone3.PilotingState import FlyingStateChanged
            flying = drone.get_state(FlyingStateChanged)["state"]
            flying_str = flying.name if hasattr(flying, "name") else str(flying)
        except Exception:
            return

        if flying_str in ("hovering", "flying"):
            x, y, z = self.fly_to
            self._queue_confirm(
                MissionPhase.FLY_TO,
                f"Airborne – flying state={flying_str}.  "
                f"Ready to fly to ({x:.2f}, {y:.2f}, {z:.2f}).",
                drone,
            )

    def _tick_fly_to(self, vision_result, now, drone=None):
        cam = self._get_cam_pos(vision_result)

        if cam is not None:
            # Position visible — update timestamp.
            self._last_cam_ts = now

            # If we were recovering, resume flying to target.
            if self._recovering:
                self._recovering = False
                print("[mission] Cam position regained – resuming FLY_TO")
                self._begin_fly_to(drone)

            # Primary: controller HOVER phase signals arrival.
            if self.ctrl is not None and hasattr(self.ctrl, "position_controller"):
                from controller import Phase
                if self.ctrl.position_controller.phase == Phase.HOVER:
                    self._queue_confirm(
                        MissionPhase.HOLD,
                        "Arrived at target coordinate.  Ready to HOLD.",
                        drone,
                    )
                return

            # Fallback: distance check.
            tx, ty, tz = self.fly_to
            dist = math.sqrt((tx - cam[0])**2 + (ty - cam[1])**2 + (tz - cam[2])**2)
            if dist < HOLD_ARRIVE_RADIUS:
                self._queue_confirm(
                    MissionPhase.HOLD,
                    f"Arrived (cam dist={dist:.2f} m).  Ready to HOLD.",
                    drone,
                )
            return

        # No cam position — check if timeout exceeded.
        lost_for = now - self._last_cam_ts
        if lost_for < CAM_LOST_TIMEOUT_S:
            return  # brief dropout, keep flying

        # Timeout exceeded: enter/continue recovery yaw.
        if not self._recovering:
            self._recovering = True
            print(
                f"[mission] FLY_TO – cam lost for {lost_for:.1f} s, "
                "stopping and yaw-searching for position markers"
            )
            if self.ctrl is not None:
                self.ctrl.clear_target()
            self._set_recovery_yaw_speed(drone)

        self._send_recovery_yaw(drone)

    def _tick_hold(self, now):
        pass  # Hold position indefinitely; controller keeps target active

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

    @staticmethod
    def _get_cam_pos(vision_result: Optional[dict]) -> Optional[list]:
        """Return [x, y, z] from vision_result['cam'], or None if unavailable."""
        if vision_result is None:
            return None
        cam = vision_result.get("cam")
        if cam is None or len(cam) < 3:
            return None
        return [float(cam[0]), float(cam[1]), float(cam[2])]

    def _begin_fly_to(self, drone) -> None:
        x, y, z = self.fly_to
        print(f"[mission] FLY_TO – heading to ({x:.2f}, {y:.2f}, {z:.2f})")
        self._last_cam_ts = time.monotonic()  # reset so timeout doesn't fire immediately
        self._recovering = False
        if self.ctrl is not None:
            self.ctrl.set_target(x, y, z)

    def _set_recovery_yaw_speed(self, drone) -> None:
        """Set drone MaxRotationSpeed once when entering recovery so yaw PCMD is effective."""
        if drone is None or self.dry_run:
            return
        try:
            from olympe.messages.ardrone3.SpeedSettings import MaxRotationSpeed
            drone(MaxRotationSpeed(RECOVERY_MAX_YAW_SPEED))
        except Exception:
            pass

    def _send_recovery_yaw(self, drone) -> None:
        """Send a pure yaw PCMD — zero roll, pitch, gaz — to rotate in place."""
        if drone is None or self.dry_run:
            return
        try:
            from olympe.messages.ardrone3.Piloting import PCMD
            self._recovery_seq = (self._recovery_seq + 1) & 0x7FFFFFFF
            yaw_pct = int(RECOVERY_YAW_RATE / 0.8 * 100)  # 0.8 rad/s → 100 %
            drone(PCMD(0, 0, 0, yaw_pct, 0, self._recovery_seq))
        except Exception:
            pass


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
    mission_dry_run  = DRY_RUN or "--dry-run" in sys.argv
    mission_auto_confirm = "--yes" in sys.argv or "-y" in sys.argv
    mission_fly_to: Optional[tuple] = None
    if "--fly-to" in sys.argv:
        try:
            raw = sys.argv[sys.argv.index("--fly-to") + 1]
            parts = [float(v) for v in raw.split(",")]
            if len(parts) == 3:
                mission_fly_to = (parts[0], parts[1], parts[2])
            else:
                print("❌ --fly-to expects x,y,z  e.g. --fly-to 3.0,2.0,1.5")
                sys.exit(1)
        except Exception:
            print("❌ Invalid --fly-to value.  Expected x,y,z  e.g. --fly-to 3.0,2.0,1.5")
            sys.exit(1)

    if mission_enabled and mission_fly_to is None:
        print("❌ --mission requires --fly-to x,y,z  e.g.:  python main.py --mission --fly-to 3.0,2.0,1.5")
        sys.exit(1)

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
    fly_to_str = f"({mission_fly_to[0]:.2f}, {mission_fly_to[1]:.2f}, {mission_fly_to[2]:.2f})" if mission_fly_to else "not set"
    print(f"🎯 Autonomous Mission: {'ON fly-to ' + fly_to_str if mission_enabled else 'OFF'}")
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
            fly_to=mission_fly_to,
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
        from olympe.messages.ardrone3.SpeedSettings import MaxRotationSpeed
        anafi_drone(MaxRotationSpeed(150))

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

    MANUAL_SPEED    = 30    # % stick for manual W/A/S/D/Q/E override
    MANUAL_TIMEOUT  = 0.15  # s: auto-stop when key not pressed for this long
    _manual_rc_lock = threading.Lock()
    _manual_rc      = {"forward_back": 0, "left_right": 0, "yaw": 0, "up_down": 0}
    _manual_rc_time = 0.0

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
        nonlocal _manual_rc_time
        try:
            if platform.system().lower() == "windows":
                import msvcrt
                while not _abort_flag.is_set():
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch == b'\x1b':
                            print("\n[abort] ESC detected in terminal")
                            _abort_flag.set()
                            return
                        elif ch == b' ':
                            print("\n[mission] SPACE – confirm")
                            _confirm_flag.set()
                        elif ch in (b'w', b'W'):
                            with _manual_rc_lock:
                                _manual_rc.update({"forward_back": MANUAL_SPEED, "left_right": 0, "yaw": 0, "up_down": 0})
                            _manual_rc_time = time.monotonic()
                        elif ch in (b's', b'S'):
                            with _manual_rc_lock:
                                _manual_rc.update({"forward_back": -MANUAL_SPEED, "left_right": 0, "yaw": 0, "up_down": 0})
                            _manual_rc_time = time.monotonic()
                        elif ch in (b'a', b'A'):
                            with _manual_rc_lock:
                                _manual_rc.update({"forward_back": 0, "left_right": -MANUAL_SPEED, "yaw": 0, "up_down": 0})
                            _manual_rc_time = time.monotonic()
                        elif ch in (b'd', b'D'):
                            with _manual_rc_lock:
                                _manual_rc.update({"forward_back": 0, "left_right": MANUAL_SPEED, "yaw": 0, "up_down": 0})
                            _manual_rc_time = time.monotonic()
                        elif ch in (b'q', b'Q'):
                            with _manual_rc_lock:
                                _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": -MANUAL_SPEED, "up_down": 0})
                            _manual_rc_time = time.monotonic()
                        elif ch in (b'e', b'E'):
                            with _manual_rc_lock:
                                _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": MANUAL_SPEED, "up_down": 0})
                            _manual_rc_time = time.monotonic()
                        elif ch in (b'r', b'R'):
                            with _manual_rc_lock:
                                _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": 0, "up_down": MANUAL_SPEED})
                            _manual_rc_time = time.monotonic()
                        elif ch in (b'f', b'F'):
                            with _manual_rc_lock:
                                _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": 0, "up_down": -MANUAL_SPEED})
                            _manual_rc_time = time.monotonic()
                    time.sleep(0.05)
            else:
                import select as _select
                import tty
                import termios
                # Open /dev/tty directly so keyboard works even when stdin is
                # redirected, piped, or running over SSH without a PTY.
                _tty_fd   = None
                _tty_file = None
                _orig_attrs = None
                try:
                    _tty_fd = os.open('/dev/tty', os.O_RDONLY)
                    _orig_attrs = termios.tcgetattr(_tty_fd)
                    tty.setcbreak(_tty_fd)
                    _tty_file = os.fdopen(_tty_fd, 'rb', buffering=0)
                    print("[keyboard] Ready – ESC=land  SPACE=confirm  W/S=fwd/back  A/D=left/right  Q/E=yaw")
                    while not _abort_flag.is_set():
                        r, _, _ = _select.select([_tty_file], [], [], 0.1)
                        if r:
                            ch = _tty_file.read(1)
                            if ch == b'\x1b':
                                print("\n[abort] ESC – landing now")
                                _abort_flag.set()
                                return
                            elif ch == b' ':
                                print("\n[mission] SPACE – confirm")
                                _confirm_flag.set()
                            elif ch in (b'w', b'W'):
                                with _manual_rc_lock:
                                    _manual_rc.update({"forward_back": MANUAL_SPEED, "left_right": 0, "yaw": 0, "up_down": 0})
                                _manual_rc_time = time.monotonic()
                            elif ch in (b's', b'S'):
                                with _manual_rc_lock:
                                    _manual_rc.update({"forward_back": -MANUAL_SPEED, "left_right": 0, "yaw": 0, "up_down": 0})
                                _manual_rc_time = time.monotonic()
                            elif ch in (b'a', b'A'):
                                with _manual_rc_lock:
                                    _manual_rc.update({"forward_back": 0, "left_right": -MANUAL_SPEED, "yaw": 0, "up_down": 0})
                                _manual_rc_time = time.monotonic()
                            elif ch in (b'd', b'D'):
                                with _manual_rc_lock:
                                    _manual_rc.update({"forward_back": 0, "left_right": MANUAL_SPEED, "yaw": 0, "up_down": 0})
                                _manual_rc_time = time.monotonic()
                            elif ch in (b'q', b'Q'):
                                with _manual_rc_lock:
                                    _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": -MANUAL_SPEED, "up_down": 0})
                                _manual_rc_time = time.monotonic()
                            elif ch in (b'e', b'E'):
                                with _manual_rc_lock:
                                    _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": MANUAL_SPEED, "up_down": 0})
                                _manual_rc_time = time.monotonic()
                            elif ch in (b'r', b'R'):
                                with _manual_rc_lock:
                                    _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": 0, "up_down": MANUAL_SPEED})
                                _manual_rc_time = time.monotonic()
                            elif ch in (b'f', b'F'):
                                with _manual_rc_lock:
                                    _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": 0, "up_down": -MANUAL_SPEED})
                                _manual_rc_time = time.monotonic()
                except Exception as _kb_err:
                    print(f"[keyboard] Not available: {_kb_err}")
                finally:
                    if _orig_attrs is not None and _tty_fd is not None:
                        try:
                            termios.tcsetattr(_tty_fd, termios.TCSADRAIN, _orig_attrs)
                        except Exception:
                            pass
                    if _tty_file is not None:
                        try:
                            _tty_file.close()
                        except Exception:
                            pass
        except Exception as _outer_kb_err:
            print(f"[keyboard] Thread error: {_outer_kb_err}")

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

            _manual_active = (time.monotonic() - _manual_rc_time < MANUAL_TIMEOUT)
            # if _manual_active:
            #     print(f"[debug] manual_active=True  ctrl={ctrl_module is not None}  "
            #           f"ctrl_state={_ctrl_state is not None}  drone={anafi_drone is not None}  "
            #           f"dry={mission_dry_run}  age={time.monotonic()-_manual_rc_time:.3f}s", flush=True)

            if _manual_active and anafi_drone is not None and not mission_dry_run:
                # Manual RC takes priority over autonomous controller
                with _manual_rc_lock:
                    _mrc = {
                        "forward_back": _manual_rc["forward_back"],
                        "left_right":   _manual_rc["left_right"],
                        "up_down":      _manual_rc["up_down"],
                        "yaw":          _manual_rc["yaw"],
                    }
                if ctrl_module is not None:
                    ctrl_module.send_pcmd_olympe(anafi_drone, _mrc)
                else:
                    from olympe.messages.ardrone3.Piloting import PCMD as _PCMD
                    anafi_drone(_PCMD(1, _mrc["left_right"], _mrc["forward_back"], _mrc["yaw"], 0, 0))
            elif ctrl_module is not None and _ctrl_state is not None:
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
                if mission is not None and mission.fly_to is not None:
                    tx, ty, tz = mission.fly_to
                    _log_tgt = {
                        "x": round(tx, 4),
                        "y": round(ty, 4),
                        "z": round(tz, 4),
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
                    "fly_to": list(mission.fly_to),
                    "recovering": mission._recovering,
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

                if mission is not None and mission.fly_to is not None:
                    tx, ty, tz = mission.fly_to
                    cv2.putText(
                        preview,
                        f"TGT x={tx:+.2f} y={ty:+.2f} z={tz:+.2f}",
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
                    if key != 0xFF:
                        print(f"[key] cv2 key={key} chr={chr(key) if 32<=key<127 else '?'}", flush=True)
                    if key == 27:   # ESC – emergency land
                        print("[abort] ESC pressed in preview window")
                        _abort_flag.set()
                    elif key == 32:   # SPACE
                        _confirm_flag.set()
                    elif key in (ord('w'), ord('W')):
                        with _manual_rc_lock:
                            _manual_rc.update({"forward_back": MANUAL_SPEED, "left_right": 0, "yaw": 0, "up_down": 0})
                        _manual_rc_time = time.monotonic()
                    elif key in (ord('s'), ord('S')):
                        with _manual_rc_lock:
                            _manual_rc.update({"forward_back": -MANUAL_SPEED, "left_right": 0, "yaw": 0, "up_down": 0})
                        _manual_rc_time = time.monotonic()
                    elif key in (ord('a'), ord('A')):
                        with _manual_rc_lock:
                            _manual_rc.update({"forward_back": 0, "left_right": -MANUAL_SPEED, "yaw": 0, "up_down": 0})
                        _manual_rc_time = time.monotonic()
                    elif key in (ord('d'), ord('D')):
                        with _manual_rc_lock:
                            _manual_rc.update({"forward_back": 0, "left_right": MANUAL_SPEED, "yaw": 0, "up_down": 0})
                        _manual_rc_time = time.monotonic()
                    elif key in (ord('q'), ord('Q')):
                        with _manual_rc_lock:
                            _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": -MANUAL_SPEED, "up_down": 0})
                        _manual_rc_time = time.monotonic()
                    elif key in (ord('e'), ord('E')):
                        with _manual_rc_lock:
                            _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": MANUAL_SPEED, "up_down": 0})
                        _manual_rc_time = time.monotonic()
                    elif key in (ord('r'), ord('R')):
                        with _manual_rc_lock:
                            _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": 0, "up_down": MANUAL_SPEED})
                        _manual_rc_time = time.monotonic()
                    elif key in (ord('f'), ord('F')):
                        with _manual_rc_lock:
                            _manual_rc.update({"forward_back": 0, "left_right": 0, "yaw": 0, "up_down": -MANUAL_SPEED})
                        _manual_rc_time = time.monotonic()
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
