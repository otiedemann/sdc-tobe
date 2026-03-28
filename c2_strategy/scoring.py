"""
Score tracker for SDC26.

Implements capture timing, bonus calculations, and score tracking per the v4.0 regulations:
- Normal capture: hover 5s over enemy box → 1 point
- Special capture: entire team returns to home zone after capture → 5 points
- Simultaneous capture: 2+ boxes captured at same time → 10 point bonus
- Instant win: control all 6 boxes for 5 seconds
"""

import time
from dataclasses import dataclass, field

from .arena_config import ArenaConfig
from .models import CaptureEvent


@dataclass
class ActiveCapture:
    """A capture in progress (drone hovering over target)."""
    box_id: str
    drone_id: str
    start_time: float
    required_seconds: float = 2.0


@dataclass
class ScoreTracker:
    config: ArenaConfig
    our_score: int = 0
    their_score: int = 0
    active_captures: dict[str, ActiveCapture] = field(default_factory=dict)
    completed_captures: list[CaptureEvent] = field(default_factory=list)
    captures_since_last_return: list[CaptureEvent] = field(default_factory=list)
    all_boxes_controlled_since: float | None = None

    def start_capture(self, box_id: str, drone_id: str) -> None:
        """Begin a capture attempt (drone started hovering over box)."""
        if box_id not in self.active_captures:
            self.active_captures[box_id] = ActiveCapture(
                box_id=box_id,
                drone_id=drone_id,
                start_time=time.monotonic(),
                required_seconds=self.config.capture_hover_seconds,
            )

    def cancel_capture(self, box_id: str) -> None:
        """Cancel a capture (drone moved away or lost)."""
        self.active_captures.pop(box_id, None)

    def check_captures(self) -> list[CaptureEvent]:
        """Check for newly completed captures. Returns list of just-completed captures."""
        now = time.monotonic()
        completed: list[CaptureEvent] = []
        done_ids: list[str] = []

        for box_id, cap in self.active_captures.items():
            elapsed = now - cap.start_time
            if elapsed >= cap.required_seconds:
                event = CaptureEvent(
                    box_id=box_id,
                    drone_id=cap.drone_id,
                    start_time=cap.start_time,
                    end_time=now,
                    points=self.config.normal_capture_points,
                )
                completed.append(event)
                done_ids.append(box_id)

        for box_id in done_ids:
            del self.active_captures[box_id]

        # Check for simultaneous capture bonus
        if len(completed) >= 2:
            bonus = self.config.simultaneous_capture_bonus
            for event in completed:
                event.points += bonus // len(completed)  # distribute bonus
            self.our_score += bonus

        # Apply normal points
        for event in completed:
            self.our_score += self.config.normal_capture_points
            self.completed_captures.append(event)
            self.captures_since_last_return.append(event)

        return completed

    def apply_team_return_bonus(self) -> int:
        """
        Called when all drones are confirmed in home zone.
        Awards special capture bonus for each capture since the last team return.
        Returns bonus points awarded.
        """
        bonus = 0
        if self.captures_since_last_return:
            bonus = len(self.captures_since_last_return) * self.config.special_capture_points
            self.our_score += bonus
            self.captures_since_last_return.clear()
        return bonus

    def check_instant_win(self, controlled_box_ids: set[str], total_boxes: int = 6) -> bool:
        """
        Check if we control all boxes for the required duration.
        Returns True if instant win condition is met.
        """
        now = time.monotonic()

        if len(controlled_box_ids) >= total_boxes:
            if self.all_boxes_controlled_since is None:
                self.all_boxes_controlled_since = now
            elif (now - self.all_boxes_controlled_since) >= self.config.capture_hover_seconds:
                return True
        else:
            self.all_boxes_controlled_since = None

        return False

    def summary(self) -> dict:
        """Return a score summary."""
        return {
            "our_score": self.our_score,
            "their_score": self.their_score,
            "total_captures": len(self.completed_captures),
            "active_captures": len(self.active_captures),
            "pending_return_captures": len(self.captures_since_last_return),
            "captures": [e.model_dump() for e in self.completed_captures[-10:]],
        }
