"""Per-drone websocket client. Maintains three long-lived WS to each Pi
(/ws/telemetry, /ws/position, /ws/rc) so the C2 dodges per-call HTTP
framing overhead and can answer telemetry/position queries from RAM."""
from __future__ import annotations

import json
import threading
import time

try:
    import websocket as _wsclient  # websocket-client package
    HAS_WSCLIENT = True
except Exception as _e:
    _wsclient = None
    HAS_WSCLIENT = False


class DroneWS:
    """One WS channel (really three sockets) to a single Pi. All
    connections auto-reconnect with 1s backoff up to 5s. When the
    websocket-client package isn't present (HAS_WSCLIENT=False) all
    sends become no-ops and callers fall back to HTTP."""

    def __init__(self, drone_id: str, base_http_url: str):
        self.drone_id = str(drone_id)
        self.base_http = base_http_url.rstrip("/")
        # Convert http(s):// → ws(s):// for the WS URL
        if self.base_http.startswith("https://"):
            self.ws_base = "wss://" + self.base_http[len("https://"):]
        elif self.base_http.startswith("http://"):
            self.ws_base = "ws://"  + self.base_http[len("http://"):]
        else:
            self.ws_base = "ws://"  + self.base_http

        self._lock = threading.Lock()
        self._latest_tel: dict | None = None
        self._latest_tel_ts: float = 0.0
        self._latest_pos: dict | None = None
        self._latest_pos_ts: float = 0.0
        self._rc_ws = None                    # websocket.WebSocket
        self._rc_ws_lock = threading.Lock()
        self._rc_seq = 0
        self._last_rc_send_ts: float = 0.0
        self._last_rc_send_ms: float = 0.0      # wall-clock of the most recent ws.send()
        self._rc_rtt_ms: float = 0.0
        self._ws_connected = {"telemetry": False, "position": False, "rc": False}
        # Log-suppression state: only print on state transitions, not every
        # reconnect attempt. Offline hosts would otherwise produce 3 lines
        # every 5 s per drone = a flood that obscures every real log.
        self._was_connected = {"telemetry": False, "position": False, "rc": False}
        self._consec_failures = {"telemetry": 0, "position": 0, "rc": 0}
        self._running = False

    @staticmethod
    def _sockopt_low_latency():
        """TCP_NODELAY kills Nagle's 40 ms batching delay on small
        frames — RC/key events are tiny (50-80 bytes) so without this
        the OS would buffer them. SO_KEEPALIVE on the socket helps us
        notice dead links faster than a timeout."""
        import socket as _sock
        return [
            (_sock.IPPROTO_TCP, _sock.TCP_NODELAY, 1),
            (_sock.SOL_SOCKET,  _sock.SO_KEEPALIVE, 1),
        ]

    # --- Public API ---
    def start(self):
        if not HAS_WSCLIENT or self._running:
            return
        self._running = True
        for name, target in (
            ("telemetry", self._rx_telemetry_loop),
            ("position",  self._rx_position_loop),
            ("rc",        self._rc_connect_loop),
        ):
            t = threading.Thread(target=target, daemon=True,
                                  name=f"ws-{name}-{self.drone_id}")
            t.start()

    def stop(self):
        self._running = False
        with self._rc_ws_lock:
            if self._rc_ws is not None:
                try: self._rc_ws.close()
                except Exception: pass
                self._rc_ws = None

    def latest_telemetry(self) -> tuple[dict | None, float]:
        """Return (cached_telemetry, age_seconds) or (None, +inf) when
        no WS frame has arrived."""
        with self._lock:
            if self._latest_tel is None:
                return None, float("inf")
            return dict(self._latest_tel), time.time() - self._latest_tel_ts

    def latest_position(self) -> tuple[dict | None, float]:
        with self._lock:
            if self._latest_pos is None:
                return None, float("inf")
            return dict(self._latest_pos), time.time() - self._latest_pos_ts

    def send_rc(self, lr: int, fb: int, ud: int, yaw: int,
                duration_ms: int = 250) -> bool:
        """Send an RC frame over WS. Returns True on success, False if
        the socket is not currently connected — caller should fall back
        to HTTP."""
        return self._send_rc_message({
            "type": "rc", "lr": int(lr), "fb": int(fb),
            "ud": int(ud), "yaw": int(yaw),
            "duration_ms": int(duration_ms),
        })

    def send_key(self, key: str, event: str) -> bool:
        """Send a key_down / key_up over WS. event must be 'down' or 'up'."""
        if event not in ("down", "up"):
            return False
        return self._send_rc_message({
            "type": "key", "key": str(key).lower(), "event": event,
        })

    def status(self) -> dict:
        """Connection snapshot for /proxy/ws/status + UI badge."""
        _, tel_age = self.latest_telemetry()
        _, pos_age = self.latest_position()
        with self._lock:
            rc_rtt     = self._rc_rtt_ms
            rc_send_ms = self._last_rc_send_ms
        return {
            "drone_id": self.drone_id,
            "rc":        self._ws_connected["rc"],
            "telemetry": self._ws_connected["telemetry"],
            "position":  self._ws_connected["position"],
            "telemetry_age_ms": int(tel_age * 1000) if tel_age < 1e6 else None,
            "position_age_ms":  int(pos_age * 1000) if pos_age < 1e6 else None,
            "rc_rtt_ms":        rc_rtt     if rc_rtt > 0 else None,
            "rc_send_ms":       rc_send_ms if rc_send_ms > 0 else None,
        }

    # --- Internals ---
    def _send_rc_message(self, msg: dict) -> bool:
        """Send fire-and-forget with a tight timeout budget. If the send
        takes noticeably longer than a round-trip (~50 ms), we treat the
        socket as stalled, abandon it, and let the reconnect loop spin
        up a fresh one — otherwise TCP back-pressure from a wedged
        server can queue RC frames for seconds of perceived lag.

        No sequence numbers / no ACKs — RC is idempotent and bandwidth
        is tiny; any "lost" frame is corrected on the next 100 ms tick.
        """
        if not HAS_WSCLIENT:
            return False
        ws = self._rc_ws        # snapshot ref — no lock needed
        if ws is None:
            return False
        t0 = time.time()
        try:
            ws.send(json.dumps(msg))
            dt_ms = (time.time() - t0) * 1000.0
            with self._lock:
                self._last_rc_send_ts = time.time()
                self._last_rc_send_ms = dt_ms
            # Any RC send over 250 ms is anomalous — the socket is
            # almost certainly wedged by TCP back-pressure. Kill it so
            # the reconnect loop replaces it, otherwise every
            # subsequent send waits behind the same clogged buffer.
            if dt_ms > 250.0:
                print(f"[WS] {self.drone_id} rc send SLOW {dt_ms:.0f}ms — "
                      f"dropping socket")
                with self._rc_ws_lock:
                    if self._rc_ws is ws:
                        try: ws.close()
                        except Exception: pass
                        self._rc_ws = None
                        self._ws_connected["rc"] = False
                return False
            return True
        except Exception as e:
            print(f"[WS] {self.drone_id} rc send failed: {e}")
            with self._rc_ws_lock:
                if self._rc_ws is ws:
                    try: ws.close()
                    except Exception: pass
                    self._rc_ws = None
                    self._ws_connected["rc"] = False
            return False

    def _rx_telemetry_loop(self):
        self._rx_pull_loop("telemetry", f"{self.ws_base}/ws/telemetry",
                            self._on_telemetry_msg)

    def _rx_position_loop(self):
        self._rx_pull_loop("position", f"{self.ws_base}/ws/position",
                            self._on_position_msg)

    def _rx_pull_loop(self, name: str, url: str, handler):
        """Generic pull loop — opens a WS, reads until it closes, marks
        disconnected, retries with backoff.

        recv timeout must be comfortably longer than the server's
        _WS_PING_INTERVAL_S (currently 3 s) so idle channels don't
        false-positive as dead links. 30 s gives ~10× headroom and
        still catches real link losses in under a minute.

        Logging discipline: an offline host would otherwise log every
        reconnect attempt forever. We log only on state transitions
        (first failure after connected, or first success after
        failing). Backoff also ramps to 60 s so the log stays quiet
        and bandwidth is minimal when a drone is simply offline."""
        backoff = 1.0
        while self._running:
            try:
                ws = _wsclient.create_connection(
                    url, timeout=4, sockopt=self._sockopt_low_latency())
                ws.settimeout(30.0)
                self._ws_connected[name] = True
                # Transition FAIL → OK — log only once, reset counters.
                if not self._was_connected[name]:
                    if self._consec_failures[name] > 0:
                        print(f"[WS] {self.drone_id} {name} connected "
                              f"(after {self._consec_failures[name]} failure(s))")
                    self._was_connected[name] = True
                    self._consec_failures[name] = 0
                backoff = 1.0          # reset on successful connect
                while self._running:
                    try:
                        msg = ws.recv()
                    except Exception:
                        break
                    if not msg:
                        break
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    try:
                        handler(data)
                    except Exception as he:
                        print(f"[WS] {self.drone_id} {name} handler error: {he}")
                try: ws.close()
                except Exception: pass
            except Exception as e:
                if self._running:
                    self._consec_failures[name] += 1
                    # Log once on: (a) transition from connected → failed,
                    # or (b) the very first failure at boot so the operator
                    # knows a drone is unreachable. Subsequent reconnect
                    # attempts stay silent so the log doesn't flood while
                    # the drone sits offline.
                    first_failure = (self._consec_failures[name] == 1
                                      and not self._was_connected[name])
                    if self._was_connected[name] or first_failure:
                        self._was_connected[name] = False
                        msg = str(e)
                        if ("Connection refused" not in msg
                                and "Connection closed" not in msg
                                and "1000" not in msg):
                            print(f"[WS] {self.drone_id} {name} disconnect: {e}")
                        else:
                            print(f"[WS] {self.drone_id} {name} offline "
                                  f"(retries will stay silent until recovery)")
            finally:
                self._ws_connected[name] = False
            if self._running:
                time.sleep(backoff)
                # Fast retries while we're likely just between frames (1-5 s),
                # slow retries while the host is clearly offline (>10 failures
                # → 60 s cap). Keeps the reconnect alive without spamming.
                if self._consec_failures[name] <= 3:
                    backoff = min(5.0, backoff * 1.6)
                else:
                    backoff = min(60.0, backoff * 2.0)

    def _rc_connect_loop(self):
        """Keeps the RC send socket alive.

        Crucial latency details:
          - TCP_NODELAY via sockopt — without it, Nagle's 40 ms delay
            batches small RC frames and key events feel sluggish.
          - settimeout(0.3) — short. The timeout applies to BOTH recv
            and send on the socket, so a big value (old: 2 s) meant a
            single stalled send blocked the caller for 2 s. 0.3 s is
            long enough for LAN round-trips but short enough that a
            dropped link fails fast and falls back to HTTP.
          - Ping every ~2 s and measure the round-trip — exposed as
            rc_rtt_ms so operators can see the actual latency."""
        url = f"{self.ws_base}/ws/rc"
        backoff = 1.0
        while self._running:
            try:
                ws = _wsclient.create_connection(
                    url, timeout=4, sockopt=self._sockopt_low_latency())
                ws.settimeout(0.3)
                with self._rc_ws_lock:
                    self._rc_ws = ws
                self._ws_connected["rc"] = True
                # Transition FAIL → OK — log only once.
                if not self._was_connected["rc"]:
                    if self._consec_failures["rc"] > 0:
                        print(f"[WS] {self.drone_id} rc connected "
                              f"(after {self._consec_failures['rc']} failure(s))")
                    self._was_connected["rc"] = True
                    self._consec_failures["rc"] = 0
                backoff = 1.0
                last_ping = 0.0
                ping_pending: dict[int, float] = {}  # client_ts → send_monotonic
                while self._running:
                    # Opportunistic ping for RTT measurement
                    now = time.time()
                    if now - last_ping > 2.0:
                        last_ping = now
                        mono = time.monotonic()
                        try:
                            ws.send(json.dumps({
                                "type": "ping",
                                "client_ts": now,
                            }))
                            ping_pending[int(now * 1000)] = mono
                        except Exception:
                            break
                    try:
                        msg = ws.recv()
                    except _wsclient._exceptions.WebSocketTimeoutException:
                        continue
                    except Exception:
                        break
                    if not msg:
                        break
                    try:
                        data = json.loads(msg)
                    except Exception:
                        continue
                    if data.get("type") == "pong":
                        echo = data.get("echo")
                        if echo is not None:
                            sent_mono = ping_pending.pop(int(float(echo) * 1000), None)
                            if sent_mono is not None:
                                with self._lock:
                                    self._rc_rtt_ms = round((time.monotonic() - sent_mono) * 1000.0, 1)
            except Exception as e:
                if self._running:
                    self._consec_failures["rc"] += 1
                    first_failure = (self._consec_failures["rc"] == 1
                                      and not self._was_connected["rc"])
                    if self._was_connected["rc"] or first_failure:
                        self._was_connected["rc"] = False
                        m = str(e)
                        if ("Connection refused" not in m
                                and "Connection closed" not in m
                                and "1000" not in m):
                            print(f"[WS] {self.drone_id} rc disconnect: {e}")
                        else:
                            print(f"[WS] {self.drone_id} rc offline "
                                  f"(retries silent until recovery)")
            finally:
                with self._rc_ws_lock:
                    self._rc_ws = None
                self._ws_connected["rc"] = False
            if self._running:
                time.sleep(backoff)
                if self._consec_failures["rc"] <= 3:
                    backoff = min(5.0, backoff * 1.6)
                else:
                    backoff = min(60.0, backoff * 2.0)

    def _on_telemetry_msg(self, data: dict):
        if data.get("type") not in (None, "telemetry"):
            return
        # Strip framing field; store the rest
        snap = {k: v for k, v in data.items() if k not in ("type",)}
        with self._lock:
            self._latest_tel = snap
            self._latest_tel_ts = time.time()

    def _on_position_msg(self, data: dict):
        if data.get("type") not in (None, "position"):
            return
        snap = {k: v for k, v in data.items() if k not in ("type",)}
        with self._lock:
            self._latest_pos = snap
            self._latest_pos_ts = time.time()



# ── Fleet of clients ──────────────────────────────────────────────
# Populated by start_fleet(). The dict is exported so other parts of
# the app (heartbeat loop, route handlers) can look up the per-drone
# client by id.
drone_ws: dict[str, DroneWS] = {}


def start_fleet(drones: dict) -> dict[str, "DroneWS"]:
    """One DroneWS per configured drone. Each client connects on its
    own schedule — failing to reach a drone never blocks the C2 boot.
    Returns the populated drone_ws dict."""
    for did, info in drones.items():
        base = (info or {}).get("base")
        if not base:
            continue
        try:
            client = DroneWS(str(did), base)
            client.start()
            drone_ws[str(did)] = client
            print(f"[WS] started client for drone {did} → {client.ws_base}")
        except Exception as e:
            print(f"[WS] failed to start client for drone {did}: {e}")
    return drone_ws
