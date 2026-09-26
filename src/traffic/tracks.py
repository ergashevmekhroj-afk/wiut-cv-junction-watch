"""Observation table -> per-track trajectories in reference coordinates.

Each Track holds, per sampled frame: time, reference-space foot point (bottom
centre of the box, i.e. where the object touches the road), box size, and
smoothed velocity. Speeds are in reference px/s; with the camera fixed that is
a consistent (if uncalibrated) unit across videos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .perception import C, PERSON, BICYCLE, MOTORCYCLE, VEHICLES
from .scene import warp_points


@dataclass
class Track:
    tid: int
    cls: int                 # majority COCO class
    t: np.ndarray            # (n,) seconds
    frame: np.ndarray        # (n,) frame index
    xy: np.ndarray           # (n,2) smoothed foot point, reference px
    box: np.ndarray          # (n,4) native box
    wh: np.ndarray           # (n,2) box size in reference px
    v: np.ndarray            # (n,2) velocity, reference px/s
    conf: float
    edge: np.ndarray | None = None   # (n,) box touches the image border (truncated box)
    rider_hits: int = 0              # person: frames where a bike/motorbike box sits under it

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.v, axis=1)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    @property
    def is_vehicle(self) -> bool:
        return self.cls in VEHICLES

    @property
    def is_person(self) -> bool:
        return self.cls == PERSON

    def at(self, t: float) -> int:
        """Index of the sample closest to time t."""
        return int(np.argmin(np.abs(self.t - t)))


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) < 3 or k <= 1:
        return x.copy()
    k = min(k, len(x) if len(x) % 2 else len(x) - 1)
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(xp[:, j], ker, mode="valid") for j in range(x.shape[1])], 1)


def build_tracks(obs: np.ndarray, H: np.ndarray, frame_wh: tuple[int, int] | None = None, min_len: int = 5,
                 smooth_sec: float = 0.5, vel_sec: float = 0.6) -> list[Track]:
    """Group observations by track id and compute smoothed kinematics."""
    tracks: list[Track] = []
    if len(obs) == 0:
        return tracks
    obs = obs[np.lexsort((obs[:, C["t"]], obs[:, C["tid"]]))]
    ids, starts = np.unique(obs[:, C["tid"]], return_index=True)
    bounds = list(starts) + [len(obs)]
    for k, tid in enumerate(ids):
        o = obs[bounds[k]:bounds[k + 1]]
        if len(o) < min_len:
            continue
        cls_vals, counts = np.unique(o[:, C["cls"]].astype(int), return_counts=True)
        cls = int(cls_vals[np.argmax(counts)])
        # bicycles/motorbikes ridden by a person: keep as the vehicle
        t = o[:, C["t"]]
        box = o[:, [C["x1"], C["y1"], C["x2"], C["y2"]]]
        foot = np.stack([(box[:, 0] + box[:, 2]) / 2, box[:, 3]], 1)
        if cls in VEHICLES:
            # for vehicles seen from above, the box centre is closer to the road
            # contact point than the bottom edge; use a point 25% up from the bottom
            foot = np.stack([(box[:, 0] + box[:, 2]) / 2, box[:, 3] - 0.25 * (box[:, 3] - box[:, 1])], 1)
        ref = warp_points(H, foot)
        tl, br = warp_points(H, box[:, :2]), warp_points(H, box[:, 2:])
        wh = np.abs(br - tl)
        dt = np.median(np.diff(t)) if len(t) > 1 else 0.1
        xy = _smooth(ref, max(1, int(round(smooth_sec / dt)) | 1))
        # velocity: central difference over +-vel_sec/2
        h = max(1, int(round(vel_sec / dt / 2)))
        v = np.zeros_like(xy)
        for i in range(len(xy)):
            a, b = max(0, i - h), min(len(xy) - 1, i + h)
            if b > a and t[b] > t[a]:
                v[i] = (xy[b] - xy[a]) / (t[b] - t[a])
        edge = np.zeros(len(t), bool)
        if frame_wh is not None:
            m = 0.004 * frame_wh[0]
            edge = (box[:, 0] < m) | (box[:, 1] < m) | (box[:, 2] > frame_wh[0] - m) | (box[:, 3] > frame_wh[1] - m)
        tracks.append(Track(int(tid), cls, t, o[:, C["frame"]].astype(int), xy, box, wh, v,
                            float(o[:, C["conf"]].mean()), edge))
    return tracks


def mark_riders(tracks: list[Track], obs: np.ndarray) -> None:
    """Count, per person track, the frames in which a bicycle/motorcycle box overlaps it.

    YOLO labels a scooter/bike rider as `person` (the two-wheeler is often missed),
    so a person with such overlaps is a rider in traffic, not a pedestrian.
    """
    if len(obs) == 0:
        return
    bikes = obs[np.isin(obs[:, C["cls"]], (BICYCLE, MOTORCYCLE))]
    if len(bikes) == 0:
        return
    by_frame: dict[int, np.ndarray] = {}
    for f in np.unique(bikes[:, C["frame"]]).astype(int):
        by_frame[f] = bikes[bikes[:, C["frame"]] == f][:, [C["x1"], C["y1"], C["x2"], C["y2"]]]
    for k in tracks:
        if k.cls != PERSON:
            continue
        hits = 0
        for f, b in zip(k.frame, k.box):
            bb = by_frame.get(int(f))
            if bb is None:
                continue
            ix = np.clip(np.minimum(bb[:, 2], b[2]) - np.maximum(bb[:, 0], b[0]), 0, None) * \
                 np.clip(np.minimum(bb[:, 3], b[3]) - np.maximum(bb[:, 1], b[1]), 0, None)
            below = (bb[:, 1] + bb[:, 3]) / 2 > (b[1] + b[3]) / 2
            if np.any((ix > 0.25 * (b[2] - b[0]) * (b[3] - b[1])) & below):
                hits += 1
        k.rider_hits = hits


def id_switch_near(k: Track, t: float, window: float = 1.2) -> bool:
    """True if the track looks like it jumped to another object around t.

    Signs of an ID switch: the box area changes by more than 40% between
    consecutive samples, or the foot point jumps further than the box size.
    """
    m = (k.t >= t - window) & (k.t <= t + window)
    idx = np.flatnonzero(m)
    if len(idx) < 3:
        return True
    b = k.box[idx]
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    # a rigid object's box changes size slowly; a box that swallows a neighbour grows fast
    if w.max() / max(w.min(), 1e-6) > 1.25 or h.max() / max(h.min(), 1e-6) > 1.25:
        return True
    c = (b[:, :2] + b[:, 2:]) / 2
    jump = np.linalg.norm(np.diff(c, axis=0), axis=1)
    return bool((jump > 0.5 * np.hypot(w[:-1], h[:-1])).any())


def raw_speed(k: Track, t0: float, t1: float) -> float:
    """Median speed of the RAW box centre (native px/s) in [t0, t1]; no smoothing."""
    m = (k.t >= t0) & (k.t <= t1)
    idx = np.flatnonzero(m)
    if len(idx) < 2:
        return 0.0
    c = (k.box[idx, :2] + k.box[idx, 2:]) / 2
    return float(np.median(np.linalg.norm(np.diff(c, axis=0), axis=1) / np.maximum(np.diff(k.t[idx]), 1e-3)))


def predecessor(tracks: list[Track], k: Track, max_gap: float = 12.0, max_dist: float = 60.0) -> Track | None:
    """The track that most plausibly continues into k (lost/re-identified object):
    same kind, ended before k started, within max_gap seconds and max_dist ref px."""
    best, best_d = None, max_dist
    for o in tracks:
        if o is k or o.is_vehicle != k.is_vehicle or o.t[-1] > k.t[0] + 0.6 or k.t[0] - o.t[-1] > max_gap:
            continue
        d = float(np.linalg.norm(o.xy[-1] - k.xy[0]))
        if d < best_d:
            best, best_d = o, d
    return best
