"""Learned traffic-flow prior: the normal driving direction at every road cell.

Built once from the vehicle tracks of the sample videos (tools/build_flow.py)
and shipped as scene/flow_prior.npz. A cell with a strong, one-directional
flow (coherence close to 1) defines the legal direction there; moving against
it is `wrong_way`. Cells with mixed directions (the junction box, where turns
cross) have low coherence and are ignored.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .scene import REF_H, REF_W, SCENE_DIR

CELL = 32


class FlowPrior:
    def __init__(self, sum_v: np.ndarray, count: np.ndarray):
        self.sum_v, self.count = sum_v, count
        n = np.maximum(count, 1)[..., None]
        mean = sum_v / n
        self.coherence = np.linalg.norm(mean, axis=-1)            # 0 = mixed, 1 = one direction
        self.dir = mean / np.maximum(self.coherence[..., None], 1e-6)

    @classmethod
    def empty(cls) -> "FlowPrior":
        gh, gw = REF_H // CELL + 1, REF_W // CELL + 1
        return cls(np.zeros((gh, gw, 2)), np.zeros((gh, gw)))

    @classmethod
    def load(cls, path: Path | None = None) -> "FlowPrior | None":
        p = path or SCENE_DIR / "flow_prior.npz"
        if not p.exists():
            return None
        d = np.load(p)
        return cls(d["sum_v"], d["count"])

    def save(self, path: Path | None = None) -> None:
        np.savez_compressed(path or SCENE_DIR / "flow_prior.npz", sum_v=self.sum_v, count=self.count)

    def add(self, xy: np.ndarray, v: np.ndarray, min_speed: float = 25.0) -> None:
        sp = np.linalg.norm(v, axis=1)
        m = sp > min_speed
        for (x, y), u in zip(xy[m], v[m] / sp[m, None]):
            i, j = int(y) // CELL, int(x) // CELL
            if 0 <= i < self.count.shape[0] and 0 <= j < self.count.shape[1]:
                self.sum_v[i, j] += u
                self.count[i, j] += 1

    def lookup(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per point: (legal direction (n,2), coherence (n,), support count (n,))."""
        ij = np.clip((np.asarray(xy) // CELL).astype(int), 0, None)
        i = np.clip(ij[:, 1], 0, self.count.shape[0] - 1)
        j = np.clip(ij[:, 0], 0, self.count.shape[1] - 1)
        return self.dir[i, j], self.coherence[i, j], self.count[i, j]
