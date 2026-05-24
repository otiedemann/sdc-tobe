"""Game-event detection for the SDC26 C2 strategy server.

:func:`detect_events` is a *pure* function: it diffs two snapshots of the 6
target slots and reports the meaningful transitions a commander reacts to in
AUTO mode (e.g. "the enemy just flipped one of our boxes"). It has no side
effects and holds no state -- the caller keeps the previous slot dict and
feeds it the next one each tick.

Event kinds (see ``EVENT_KINDS``):
  * ``enemy_captured_our_slot`` -- one of our home boxes turned enemy colour.
  * ``we_captured_enemy_slot``  -- one of the enemy's home boxes turned ours.
  * ``slot_color_changed``      -- any other colour flip (neutral / re-flip).
  * ``slot_lost``               -- a slot we were tracking has gone stale.
"""

from __future__ import annotations

from dataclasses import dataclass

if __package__ in (None, ""):  # run directly as a script: bootstrap the package
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from c2.config import C2Config
    from c2.models import SlotColor, TargetSlot
else:
    from .config import C2Config
    from .models import SlotColor, TargetSlot

EVENT_KINDS = (
    "enemy_captured_our_slot",
    "we_captured_enemy_slot",
    "slot_color_changed",
    "slot_lost",
)


@dataclass
class GameEvent:
    kind: str          # one of EVENT_KINDS
    slot: int
    detail: str = ""

    def to_dict(self) -> dict:
        return {"kind": self.kind, "slot": self.slot, "detail": self.detail}

    def __str__(self) -> str:
        return f"[{self.kind}] slot {self.slot}: {self.detail}".rstrip(": ").rstrip()


def detect_events(
    prev: dict[int, TargetSlot],
    curr: dict[int, TargetSlot],
    cfg: C2Config,
) -> list[GameEvent]:
    """Diff two slot dicts and return the transitions worth reacting to.

    Pure function: no mutation of inputs, no I/O.
    """
    our_color = SlotColor.from_team(cfg.our_team)
    enemy_color = SlotColor.from_team(cfg.enemy_team())
    our_slots = set(cfg.our_slots())
    enemy_slots = set(cfg.enemy_slots())

    events: list[GameEvent] = []

    for idx, cur in curr.items():
        old = prev.get(idx)
        if old is None:
            continue  # nothing to compare against yet

        # --- colour transitions (both sides must have a defined colour) ---
        both_defined = (
            old.color != SlotColor.UNKNOWN and cur.color != SlotColor.UNKNOWN
        )
        if both_defined and old.color != cur.color:
            if idx in our_slots and cur.color == enemy_color:
                events.append(GameEvent(
                    kind="enemy_captured_our_slot",
                    slot=idx,
                    detail=f"{old.color.value} -> {cur.color.value} (defend!)",
                ))
            elif idx in enemy_slots and cur.color == our_color:
                events.append(GameEvent(
                    kind="we_captured_enemy_slot",
                    slot=idx,
                    detail=f"{old.color.value} -> {cur.color.value}",
                ))
            else:
                events.append(GameEvent(
                    kind="slot_color_changed",
                    slot=idx,
                    detail=f"{old.color.value} -> {cur.color.value}",
                ))

        # --- freshness transition: a tracked slot just went stale ---------
        # Only fire on the fresh -> stale edge (best-effort, once).
        if old.fresh and not cur.fresh and cur.color != SlotColor.UNKNOWN:
            age_txt = (
                f"age {cur.age_s:.1f}s" if cur.age_s is not None else "no recent fix"
            )
            events.append(GameEvent(
                kind="slot_lost",
                slot=idx,
                detail=f"tracking lost ({age_txt})",
            ))

    return events


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from c2.models import Team  # noqa: F811  (Team only needed for the demo)

    cfg = C2Config(our_team=Team.RED)
    print("our_team:", cfg.our_team.value,
          "our_slots:", cfg.our_slots(),
          "enemy_slots:", cfg.enemy_slots())

    def slot(idx, color, *, fresh=True, age=0.5):
        return TargetSlot(
            index=idx,
            home_team=cfg.slot_map.home_team_of(idx),
            color=color,
            fresh=fresh,
            age_s=age,
        )

    # our_team=RED -> our slots are 4,5,6 ; enemy slots 1,2,3.
    prev = {
        4: slot(4, SlotColor.RED),                       # ours, enemy will flip it
        1: slot(1, SlotColor.BLUE),                      # enemy's, we will capture
        6: slot(6, SlotColor.BLUE),                      # ours, we re-cap to red
        5: slot(5, SlotColor.RED, fresh=True, age=0.5),  # ours, will go stale
        3: slot(3, SlotColor.UNKNOWN),                   # undefined, ignored
    }
    curr = {
        4: slot(4, SlotColor.BLUE),                       # enemy_captured_our_slot
        1: slot(1, SlotColor.RED),                        # we_captured_enemy_slot
        6: slot(6, SlotColor.RED),                        # slot_color_changed (re-cap own box)
        5: slot(5, SlotColor.RED, fresh=False, age=9.0),  # slot_lost
        3: slot(3, SlotColor.RED),                        # prev UNKNOWN -> no colour event
    }

    events = detect_events(prev, curr, cfg)
    print(f"detected {len(events)} events:")
    for e in events:
        print("  ", e)

    kinds = {e.kind for e in events}
    assert "enemy_captured_our_slot" in kinds, kinds
    assert "we_captured_enemy_slot" in kinds, kinds
    assert "slot_color_changed" in kinds, kinds
    assert "slot_lost" in kinds, kinds
    print("all four EVENT_KINDS triggered:", sorted(kinds))
    print("OK events smoke test")
