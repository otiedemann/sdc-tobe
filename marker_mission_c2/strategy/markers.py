"""Marker observation tracker.

Aggregates per-FC ``visible_marker_ids`` from the C2 overview into a single
table of marker status:

    marker_id -> {
        last_seen_unix_s: float,
        last_seen_by: str,                  # which FC saw it most recently
        seen_by_recent: list[str],          # FCs that saw it within freshness window
        team: "red"|"blue"|"wall"|"other",
        captured: bool,                     # toggled when no scout sees it for cooldown
        last_captured_unix_s: float|None
    }

The tracker is intentionally lightweight: it doesn't read camera frames, it
just trusts what the FCs report. The strategy runner ticks it once per loop
with the latest overview.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)

# How long an observation stays "fresh" (visible) before we consider the
# marker out of view for that FC.
FRESHNESS_S = 1.5

# After we lose sight of a marker for this many seconds (and no scout has
# reported it recently), we flag it as `captured` for UI purposes. The
# attacker role uses this signal to decide it's safe to RTH.
CAPTURE_COOLDOWN_S = 4.0


def _classify(marker_id: int, red_ids: set[int], blue_ids: set[int]) -> str:
    if marker_id in red_ids:
        return "red"
    if marker_id in blue_ids:
        return "blue"
    if 1 <= marker_id <= 30:
        return "wall"
    return "other"


@dataclass
class MarkerStatus:
    marker_id: int
    team: str = "other"
    last_seen_unix_s: float = 0.0
    last_seen_by: Optional[str] = None
    seen_by_recent: List[str] = field(default_factory=list)
    captured: bool = False
    last_captured_unix_s: Optional[float] = None


class MarkerTracker:
    """Thread-safe marker visibility + capture-state tracker."""

    def __init__(
        self,
        *,
        red_live_ids: Iterable[int] = (44, 45, 46),
        blue_live_ids: Iterable[int] = (31, 32, 33),
    ) -> None:
        self._lock = threading.RLock()
        self._red = set(red_live_ids)
        self._blue = set(blue_live_ids)
        self._markers: Dict[int, MarkerStatus] = {}
        # Per-FC last-seen for each marker; used to compute seen_by_recent.
        self._per_fc_last_seen: Dict[tuple[str, int], float] = {}

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def update_team_ids(
        self,
        *,
        red_live_ids: Iterable[int],
        blue_live_ids: Iterable[int],
    ) -> None:
        with self._lock:
            self._red = set(red_live_ids)
            self._blue = set(blue_live_ids)
            for status in self._markers.values():
                status.team = _classify(status.marker_id, self._red, self._blue)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(self, fc_name: str, visible_ids: Iterable[int]) -> None:
        """Record observations from a single FC."""
        if not fc_name:
            return
        now = time.time()
        ids = {int(x) for x in visible_ids}
        with self._lock:
            for mid in ids:
                self._per_fc_last_seen[(fc_name, mid)] = now
                status = self._markers.get(mid)
                if status is None:
                    status = MarkerStatus(
                        marker_id=mid,
                        team=_classify(mid, self._red, self._blue),
                    )
                    self._markers[mid] = status
                status.last_seen_unix_s = now
                status.last_seen_by = fc_name
                # Coming back into view => no longer captured.
                if status.captured:
                    status.captured = False
                    status.last_captured_unix_s = None
            # Recompute seen_by_recent and capture flag for every marker we know.
            self._recompute_locked(now)

    def _recompute_locked(self, now: float) -> None:
        for status in self._markers.values():
            mid = status.marker_id
            recent: List[str] = []
            for (fc, m_id), ts in self._per_fc_last_seen.items():
                if m_id == mid and (now - ts) <= FRESHNESS_S:
                    recent.append(fc)
            status.seen_by_recent = sorted(recent)
            # Capture rule: only meaningful for live target markers. A marker
            # is "captured" if nobody has seen it for at least CAPTURE_COOLDOWN_S
            # AND it was seen at some point (last_seen_unix_s > 0). We don't
            # flag wall markers — they stay visible by definition.
            if status.team in ("red", "blue") and status.last_seen_unix_s > 0:
                staleness = now - status.last_seen_unix_s
                if not recent and staleness >= CAPTURE_COOLDOWN_S:
                    if not status.captured:
                        status.captured = True
                        status.last_captured_unix_s = now

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[int, MarkerStatus]:
        with self._lock:
            # Return copies so callers don't mutate our state.
            return {
                mid: MarkerStatus(
                    marker_id=s.marker_id,
                    team=s.team,
                    last_seen_unix_s=s.last_seen_unix_s,
                    last_seen_by=s.last_seen_by,
                    seen_by_recent=list(s.seen_by_recent),
                    captured=s.captured,
                    last_captured_unix_s=s.last_captured_unix_s,
                )
                for mid, s in self._markers.items()
            }

    def status_for(self, marker_id: int) -> Optional[MarkerStatus]:
        with self._lock:
            s = self._markers.get(int(marker_id))
            if s is None:
                return None
            return MarkerStatus(
                marker_id=s.marker_id,
                team=s.team,
                last_seen_unix_s=s.last_seen_unix_s,
                last_seen_by=s.last_seen_by,
                seen_by_recent=list(s.seen_by_recent),
                captured=s.captured,
                last_captured_unix_s=s.last_captured_unix_s,
            )

    def to_dict(self) -> Dict[str, dict]:
        """JSON-friendly view of all known markers, keyed by id (string)."""
        with self._lock:
            now = time.time()
            self._recompute_locked(now)
            out: Dict[str, dict] = {}
            for mid, s in self._markers.items():
                age = (
                    now - s.last_seen_unix_s if s.last_seen_unix_s > 0 else None
                )
                out[str(mid)] = {
                    "id": s.marker_id,
                    "team": s.team,
                    "last_seen_unix_s": s.last_seen_unix_s,
                    "last_seen_age_s": age,
                    "last_seen_by": s.last_seen_by,
                    "seen_by_recent": list(s.seen_by_recent),
                    "captured": s.captured,
                    "last_captured_unix_s": s.last_captured_unix_s,
                }
            return out
