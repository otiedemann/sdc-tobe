"""Lightweight CPU / memory sampler for the marker-mission process.

A background thread polls /proc/stat, /proc/self/stat, /proc/loadavg
and /proc/meminfo at ~1 Hz and publishes the deltas. The result is
exposed through ``last_sample()`` so state.snapshot() can fold the
numbers into every CSV row and the UI can poll them live.

Existence rationale: telemetry stalls coincided with the controller
slamming saturated RC, and we want to know whether the host is CPU /
memory-starved at those moments -- a vision_worker running at 100%
of one core can starve the telemetry callback thread and look exactly
like a wifi stall from the outside. This module gives us the data
to tell those two failure modes apart without adding a psutil
dependency.

Read-only /proc access; thread-safe via a single lock around the
latest CpuSample dataclass.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Optional


# ── /proc parsers (dependency-free) ──────────────────────────────────

def _read_proc_stat() -> Optional[tuple[int, int]]:
    """Return ``(idle_jiffies, total_jiffies)`` from the aggregate
    'cpu' line of /proc/stat, or None on parse failure. ``idle``
    counts both idle and iowait so a disk-bound process doesn't
    register as CPU pressure."""
    try:
        with open("/proc/stat") as f:
            line = f.readline()
        parts = line.split()
        if not parts or parts[0] != "cpu":
            return None
        nums = [int(x) for x in parts[1:]]
        # /proc/stat fields: user nice system idle iowait irq softirq
        # steal guest guest_nice. idle = nums[3], iowait = nums[4].
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        return idle, total
    except Exception:
        return None


def _read_proc_self_cpu() -> Optional[int]:
    """Return this process's accumulated CPU jiffies (utime + stime).

    The 'comm' field at position 1 of /proc/self/stat can contain
    arbitrary characters including parens and spaces. We split on the
    last ')' to skip past it cleanly; after that the field layout is
    space-separated and stable across kernel versions."""
    try:
        with open("/proc/self/stat") as f:
            line = f.read()
        end = line.rfind(")")
        if end < 0:
            return None
        rest = line[end + 2:].split()
        # rest[0] = state. utime is at offset 11, stime at offset 12
        # in the man page numbering (post-comm).
        utime = int(rest[11])
        stime = int(rest[12])
        return utime + stime
    except Exception:
        return None


def _read_loadavg() -> Optional[float]:
    try:
        with open("/proc/loadavg") as f:
            return float(f.read().split()[0])
    except Exception:
        return None


def _read_meminfo_pct() -> Optional[float]:
    """Return percent of memory currently in use. Uses MemAvailable
    (kernel's estimate of "claim without swapping") rather than the
    raw MemFree, which on a healthy Linux box is usually low because
    the kernel caches aggressively."""
    try:
        d = {}
        with open("/proc/meminfo") as f:
            for ln in f:
                k, _, v = ln.partition(":")
                d[k] = int(v.strip().split()[0])
        total = d.get("MemTotal")
        avail = d.get("MemAvailable")
        if total and avail is not None and total > 0:
            return (1.0 - avail / total) * 100.0
    except Exception:
        pass
    return None


# ── Sample container + sampler thread ────────────────────────────────

@dataclass
class CpuSample:
    """Snapshot of host + process resource usage. Any field may be
    None on a sampling failure (e.g. /proc temporarily unreadable)."""
    system_cpu_pct: Optional[float] = None    # % of all cores busy
    process_cpu_pct: Optional[float] = None   # % of ONE core used by this proc (>100 = multi-core)
    load_1m: Optional[float] = None
    mem_used_pct: Optional[float] = None
    sampled_at: float = 0.0                   # process monotonic


class CpuSampler:
    """Background sampler. Single instance owned by mission.cmd_fly;
    state.snapshot calls ``last_sample()`` so the latest reading is
    folded into every CSV row and every /api/state response without
    blocking the caller on /proc I/O."""

    def __init__(self, sample_period_s: float = 1.0):
        self.sample_period_s = float(sample_period_s)
        self._lock = threading.Lock()
        self._sample = CpuSample()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._prev_proc: Optional[tuple[int, int]] = None
        self._prev_self_jif: Optional[int] = None
        self._nproc: int = os.cpu_count() or 1

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="cpu_monitor")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def last_sample(self) -> CpuSample:
        with self._lock:
            return self._sample

    def _run(self) -> None:
        # First reading: store the baseline. We need a delta to derive
        # a meaningful percent, so the first published sample stays at
        # the dataclass defaults (None / 0.0) for one period.
        self._prev_proc = _read_proc_stat()
        self._prev_self_jif = _read_proc_self_cpu()
        while not self._stop.wait(self.sample_period_s):
            now_proc = _read_proc_stat()
            now_self = _read_proc_self_cpu()
            system_pct: Optional[float] = None
            proc_pct: Optional[float] = None
            if now_proc is not None and self._prev_proc is not None:
                d_idle = now_proc[0] - self._prev_proc[0]
                d_total = now_proc[1] - self._prev_proc[1]
                if d_total > 0:
                    system_pct = (1.0 - d_idle / d_total) * 100.0
            if (now_self is not None and self._prev_self_jif is not None
                    and now_proc is not None and self._prev_proc is not None):
                d_self = now_self - self._prev_self_jif
                d_total = now_proc[1] - self._prev_proc[1]
                if d_total > 0:
                    # d_self / d_total = fraction of TOTAL (multi-core)
                    # system CPU used by this proc. Scaling by nproc
                    # rebases that to "% of one core" so 100 = "fully
                    # using one core" and 200 = "fully using two cores".
                    proc_pct = (d_self / d_total) * 100.0 * self._nproc
            self._prev_proc = now_proc
            self._prev_self_jif = now_self
            sample = CpuSample(
                system_cpu_pct=system_pct,
                process_cpu_pct=proc_pct,
                load_1m=_read_loadavg(),
                mem_used_pct=_read_meminfo_pct(),
                sampled_at=time.monotonic(),
            )
            with self._lock:
                self._sample = sample


# ── Module-level singleton ───────────────────────────────────────────
# state.snapshot is reached from many code paths that don't otherwise
# know about CpuSampler. Wiring a sampler through every constructor
# would touch a lot of unrelated call sites; instead the singleton
# lives here and snapshot calls last_sample() directly. start() is
# idempotent so a second call (e.g. test reuse) is a no-op.

_sampler: Optional[CpuSampler] = None


def start(sample_period_s: float = 1.0) -> CpuSampler:
    """Start (or return the running) module-level sampler."""
    global _sampler
    if _sampler is None:
        _sampler = CpuSampler(sample_period_s=sample_period_s)
        _sampler.start()
    return _sampler


def last_sample() -> CpuSample:
    """Return the latest CpuSample. If the sampler hasn't been
    started, returns an empty CpuSample with all fields None /
    sampled_at=0.0 -- callers can treat that as "no data yet"."""
    s = _sampler
    if s is None:
        return CpuSample()
    return s.last_sample()
