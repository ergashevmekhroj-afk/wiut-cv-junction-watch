"""
solution.py — WIUT Hackathon 2026, CV track.

    detect_events(video_path)  -> [[start_sec, end_sec, label], ...]    # Part A
    RiskEstimator().reset(meta); .step(frame, t_sec) -> float           # Part B

Part A: register the video to the reference view -> YOLO11s + ByteTrack
(detector every 3rd frame, signal lamp read in the same pass) -> trajectories
-> per-class rules on trajectories + scene layout + signal phase -> segments.
Part B: causal time-to-collision risk from its own detector/tracker.
See README.md and src/traffic/ for details.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.traffic.pipeline import ENABLED, detect as _detect  # noqa: E402
from src.traffic.risk import CausalRisk  # noqa: E402

# Official ids, reduced to the classes our rules predict (removing ids is
# allowed; predicting a class that never occurs would add a zero to Score A).
_OFFICIAL = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
             "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
             "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]
CLASSES: list[str] = [c for c in _OFFICIAL if c in ENABLED]

RISK_HORIZON_SEC = 5.0

# determinism
random.seed(0)
np.random.seed(0)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
try:
    import torch
    torch.manual_seed(0)
except Exception:  # pragma: no cover
    pass

_MODEL = None

# Time budget (harness: Part A + Part B <= 3 x video duration, and the harness
# itself decodes every frame again for Part B). Both parts MEASURE the decoding
# speed of the machine instead of assuming it:
#   * Part A stops early (returning the events found so far) if continuing would
#     leave less than RESERVE x (one full decoding pass) before BUDGET x duration;
#   * Part B stops running its detector (returning its last score) as soon as the
#     remaining frames, at the measured harness speed, would not fit.
# On a fast machine neither ever triggers; on a slow one we lose events, not the video.
BUDGET = 2.9      # of the 3.0 x duration allowed
RESERVE = 1.5     # safety factor: the harness read() of Part B costs ~1.3x our measured decode
POST_SEC = lambda dur: 5.0 + 0.12 * dur   # registration refinement + rules after the detection pass
_START: dict[str, tuple[float, float]] = {}   # video file name -> (start time, duration)


def detect_events(video_path: str) -> list[list]:
    """Part A — traffic event detection (see module docstring)."""
    global _MODEL
    import time
    from src.traffic.perception import PerceptionConfig, load_model, video_info
    t0 = time.perf_counter()
    dur = video_info(video_path)["duration"]
    _START[Path(video_path).name] = (t0, dur)
    cfg = PerceptionConfig()
    if _MODEL is None:
        _MODEL = load_model(cfg)
    deadline = None if os.environ.get("TRAFFIC_NO_DEADLINE") else t0 + BUDGET * dur - POST_SEC(dur)   # env: dev on slow CPUs
    events = _detect(video_path, cfg=cfg, model=_MODEL, deadline=deadline, reserve_decode=RESERVE)
    return [e for e in events if e[2] in CLASSES]


class RiskEstimator:
    """Part B — causal accident anticipation (TTC between tracked road users)."""

    _shared = None   # one detector instance reused across videos (tracker is reset per video)

    def reset(self, meta: dict) -> None:
        import time
        if RiskEstimator._shared is None:
            RiskEstimator._shared = CausalRisk()
        self.impl = RiskEstimator._shared
        self.impl.reset(meta)
        dur = meta["n_frames"] / meta["fps"] if meta.get("fps") else 0.0
        t0, _ = _START.get(meta.get("video_id", ""), (time.perf_counter(), dur))
        no_limit = bool(os.environ.get("TRAFFIC_NO_DEADLINE"))       # full reference run (predictions_samples.json)
        self.end_at = t0 + BUDGET * dur if dur and not no_limit else float("inf")
        self.n_frames = int(meta.get("n_frames") or 0)
        self.n = 0
        self.over = False
        self.returned_at = None   # when step() last returned
        self.gap = None           # EMA of (next call - last return) = harness decoding time per frame

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        import time
        now = time.perf_counter()
        if self.returned_at is not None:
            g = now - self.returned_at
            self.gap = g if self.gap is None else 0.97 * self.gap + 0.03 * g
        self.n += 1
        if not self.over and self.gap is not None and self.n % 25 == 0:
            # stop the detector once the remaining frames alone would need the rest of the budget
            self.over = now + RESERVE * max(0, self.n_frames - self.n) * self.gap > self.end_at
        out = self.impl.score if self.over else self.impl.step(frame, t_sec)
        self.returned_at = time.perf_counter()
        return out
