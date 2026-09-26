"""Turning per-sample flags / per-object intervals into clean event segments."""
from __future__ import annotations

import numpy as np


def runs(t: np.ndarray, flag: np.ndarray, merge_gap: float = 0.0, min_len: float = 0.0) -> list[tuple[float, float]]:
    """Maximal runs where flag is True -> [(start, end)], gaps < merge_gap merged, runs < min_len dropped."""
    out: list[list[float]] = []
    if len(t) == 0:
        return []
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    start = None
    for i in range(len(t)):
        if flag[i] and start is None:
            start = t[i]
        if start is not None and (not flag[i] or i == len(t) - 1):
            end = t[i] if flag[i] else t[i - 1]
            end += dt / 2
            if out and start - out[-1][1] < merge_gap:
                out[-1][1] = end
            else:
                out.append([start, end])
            start = None
    return [(s, e) for s, e in out if e - s >= min_len]


def union(intervals: list[tuple[float, float]], merge_gap: float = 0.0) -> list[tuple[float, float]]:
    """Merge overlapping (or nearly touching) intervals of the same class."""
    iv = sorted((float(s), float(e)) for s, e in intervals if e > s)
    out: list[list[float]] = []
    for s, e in iv:
        if out and s <= out[-1][1] + merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def finalize(events: dict[str, list[tuple[float, float]]], duration: float,
             merge_gap: dict[str, float] | None = None, min_len: dict[str, float] | None = None) -> list[list]:
    """Per class: union, clip to [0, duration], drop blips; returns harness format."""
    merge_gap = merge_gap or {}
    min_len = min_len or {}
    out = []
    for label, iv in events.items():
        for s, e in union(iv, merge_gap.get(label, 0.0)):
            s, e = max(0.0, s), min(duration, e)
            if e - s >= max(min_len.get(label, 0.3), 1e-3):
                out.append([round(s, 2), round(e, 2), label])
    out.sort(key=lambda x: (x[0], x[2]))
    return out
