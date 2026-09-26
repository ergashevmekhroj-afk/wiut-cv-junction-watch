"""Shared inputs for all event rules."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..flow import FlowPrior
from ..scene import Scene
from ..signal import GREEN, RED
from ..tracks import Track


@dataclass
class Context:
    tracks: list[Track]
    scene: Scene
    duration: float
    phase_t: np.ndarray = field(default_factory=lambda: np.zeros(0))   # signal sample times
    phase: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # main-carriageway vehicle state
    flow: FlowPrior | None = None
    debug: dict = field(default_factory=dict)

    # ---- signal helpers --------------------------------------------------
    def phase_at(self, t: float) -> int:
        if len(self.phase_t) == 0:
            return -1
        return int(self.phase[min(len(self.phase) - 1, int(np.searchsorted(self.phase_t, t)))])

    def red_onsets(self) -> np.ndarray:
        p = self.phase
        idx = np.flatnonzero((p[1:] == RED) & (p[:-1] != RED)) + 1
        return self.phase_t[idx]

    def green_onsets(self) -> np.ndarray:
        p = self.phase
        idx = np.flatnonzero((p[1:] == GREEN) & (p[:-1] != GREEN)) + 1
        return self.phase_t[idx]

    def time_since_red(self, t: float) -> float:
        """Seconds since the current red started (inf if not red / unknown start)."""
        if self.phase_at(t) != RED:
            return -1.0
        on = self.red_onsets()
        on = on[on <= t]
        return float(t - on[-1]) if len(on) else float("inf")

    def next_green(self, t: float) -> float:
        g = self.green_onsets()
        g = g[g > t]
        return float(g[0]) if len(g) else self.duration

    # ---- geometry helpers --------------------------------------------------
    def vehicles(self) -> list[Track]:
        return [k for k in self.tracks if k.is_vehicle]

    def people(self) -> list[Track]:
        return [k for k in self.tracks if k.is_person]
