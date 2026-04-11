import logging
import time

import cv2
import olympe

# Suppress olympe / arsdk / pdraw log noise (H264/AVCC decoder warnings, etc.)
logging.getLogger("olympe").setLevel(logging.CRITICAL)
logging.getLogger("ulog").setLevel(logging.CRITICAL)


class LatencyMeter:
    def __init__(self, drone_ip="192.168.42.1"):
        self.drone_ip = drone_ip
        self.drone = olympe.Drone(drone_ip)
        self.latencies = []
        self._running = False

    def _extract_timestamp_us(self, info):
        """Try common locations for capture_timestamp in olympe frame info dict."""
        if not isinstance(info, dict):
            return None
        for key in ("frame", "coded", "raw"):
            sub = info.get(key, {})
            if isinstance(sub, dict):
                ts = sub.get("capture_timestamp")
                if ts:
                    return ts
        return info.get("capture_timestamp")

    def _on_h264_frame(self, h264_frame):
        """Timing only - no ref/unref/byte-extraction to avoid packed-buffer conflicts."""
        t_received = time.time()
        try:
            t_captured_us = self._extract_timestamp_us(h264_frame.info())
            if t_captured_us and t_captured_us > 0:
                latency_ms = (t_received - t_captured_us / 1_000_000) * 1000
                if 0 < latency_ms < 2000:
                    self.latencies.append(latency_ms)
                    avg = sum(self.latencies) / len(self.latencies)
                    print(f"Frame latency: {latency_ms:.1f} ms | Avg: {avg:.1f} ms")
        except Exception:
            pass

    def _on_flush(self, stream):
        try:
            while True:
                try:
                    stream.get(timeout=0.005).unref()
                except Exception:
                    break
        except Exception:
            pass
        return True

    def start_measurement(self, duration_sec=10):
        self.drone.connect()
        self._running = True

        if hasattr(self.drone, "streaming") and hasattr(self.drone.streaming, "set_callbacks"):
            self.drone.streaming.set_callbacks(
                h264_cb=self._on_h264_frame,
                flush_h264_cb=self._on_flush,
            )
            self.drone.streaming.start()
        else:
            self.drone.set_streaming_callbacks(raw_cb=self._on_h264_frame)
            self.drone.start_video_streaming()

        # Brief delay to let olympe negotiate the stream before cv2 connects
        time.sleep(1.0)

        # Display via cv2/ffmpeg RTSP - bypasses pdraw's broken H264/AVCC decoder
        rtsp_url = f"rtsp://{self.drone_ip}/live"
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"[display] Could not open {rtsp_url} - running without display")
            cap = None

        print(f"Measuring latency for {duration_sec} seconds... (press 'q' to quit early)")
        t_end = time.time() + duration_sec

        while time.time() < t_end and self._running:
            if cap is not None:
                ret, frame = cap.read()
                if ret:
                    cv2.imshow("Anafi - Latency Check", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.05)

        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

        try:
            self.drone.streaming.stop()
        except Exception:
            try:
                self.drone.stop_video_streaming()
            except Exception:
                pass
        self.drone.disconnect()

        if self.latencies:
            avg = sum(self.latencies) / len(self.latencies)
            print(f"\nResult: avg={avg:.1f} ms  min={min(self.latencies):.1f} ms  max={max(self.latencies):.1f} ms  n={len(self.latencies)}")
        else:
            print("\nNo latency samples collected (capture timestamps may not be in stream metadata).")


if __name__ == "__main__":
    meter = LatencyMeter()
    meter.start_measurement()
