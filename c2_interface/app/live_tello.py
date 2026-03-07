import threading
import time
from dataclasses import dataclass

from djitellopy import Tello
from PIL import Image


@dataclass
class LiveDroneConfig:
    drone_id: str
    host: str = "192.168.10.1"


class TelloHandle:
    def __init__(self, cfg: LiveDroneConfig) -> None:
        self.cfg = cfg
        self.tello: Tello | None = None
        self.connected = False
        self._jpeg = b""
        self._stop = False
        self._thread: threading.Thread | None = None

    def connect(self) -> None:
        if self.connected:
            return
        self.tello = Tello(host=self.cfg.host)
        self.tello.connect()
        self.tello.streamon()
        self.connected = True
        self._thread = threading.Thread(target=self._frame_loop, daemon=True)
        self._thread.start()

    def disconnect(self) -> None:
        self._stop = True
        if self.tello is not None:
            try:
                self.tello.streamoff()
            except Exception:
                pass
            try:
                self.tello.end()
            except Exception:
                pass
        self.connected = False

    def battery(self) -> int | None:
        if not self.connected:
            return None
        if self.tello is None:
            return None
        try:
            return int(self.tello.get_battery())
        except Exception:
            return None

    def _frame_loop(self) -> None:
        if self.tello is None:
            return
        frame_read = self.tello.get_frame_read()
        while not self._stop:
            frame = frame_read.frame
            if frame is None:
                time.sleep(0.02)
                continue
            try:
                # BGR (OpenCV) -> RGB (Pillow)
                rgb = frame[:, :, ::-1]
                img = Image.fromarray(rgb)
                from io import BytesIO
                b = BytesIO()
                img.save(b, format="JPEG", quality=80)
                self._jpeg = b.getvalue()
            except Exception:
                pass
            time.sleep(0.03)

    def latest_jpeg(self) -> bytes:
        return self._jpeg


class LiveTelloManager:
    def __init__(self) -> None:
        self.handles: dict[str, TelloHandle] = {}

    def set_drones(self, configs: list[LiveDroneConfig]) -> None:
        existing = set(self.handles.keys())
        incoming = {c.drone_id for c in configs}
        for removed in existing - incoming:
            self.handles[removed].disconnect()
            del self.handles[removed]
        for cfg in configs:
            if cfg.drone_id not in self.handles:
                self.handles[cfg.drone_id] = TelloHandle(cfg)

    def connect_all(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for drone_id, h in self.handles.items():
            try:
                h.connect()
                out[drone_id] = "connected"
            except Exception as e:
                out[drone_id] = f"error: {e}"
        return out

    def disconnect_all(self) -> None:
        for h in self.handles.values():
            h.disconnect()

    def latest_jpeg(self, drone_id: str) -> bytes | None:
        h = self.handles.get(drone_id)
        if not h:
            return None
        return h.latest_jpeg()
