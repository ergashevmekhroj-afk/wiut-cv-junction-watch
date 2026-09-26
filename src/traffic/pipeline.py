"""Part A pipeline: video -> events.

    registration -> detection + tracking (+ signal lamp, same pass) -> trajectories
    -> rules -> segment post-processing
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .flow import FlowPrior
from .perception import PerceptionConfig, REPO, run_perception, video_info
from .rules import detect_all
from .rules.context import Context
from .scene import Scene, register, video_to_reference
from .segments import finalize
from .signal import SignalReader, phase_series, read_phase
from .tracks import build_tracks, mark_riders

# classes we emit. A class we predict that never occurs in the test set scores 0
# and still counts in the macro average, so only classes whose rule was checked
# on the samples are enabled (see README, "What is predicted").
ENABLED = {
    "red_light", "stop_line", "stopped_vehicle", "wrong_way", "illegal_u_turn",
    "jaywalking", "failure_to_yield", "near_miss", "accident", "congestion",
}
MERGE_GAP = {"jaywalking": 4.0, "congestion": 5.0, "failure_to_yield": 3.0, "red_light": 0.5}
MIN_LEN = {"stopped_vehicle": 10.0, "congestion": 20.0, "jaywalking": 6.0}   # jaywalking: labelled events last 11-17 s


def _phase_cache(video_path: str) -> Path | None:
    root = os.environ.get("TRAFFIC_CACHE", str(REPO / ".cache"))
    if root.lower() == "off":
        return None
    st = os.stat(video_path)
    h = hashlib.md5(f"{Path(video_path).name}-{st.st_size}-phase-v1".encode()).hexdigest()[:12]
    return Path(root) / f"{Path(video_path).stem}-{h}-phase.npz"


def _first_frame_H(video_path: str, width: int) -> tuple[np.ndarray, int]:
    """Registration from the first frame only (no seeking): used for the lamp ROI."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return np.diag([1280.0 / width, 1280.0 / width, 1.0]), 0
    small = cv2.resize(f, (1280, int(round(f.shape[0] * 1280 / f.shape[1]))), interpolation=cv2.INTER_AREA)
    H, n = register(small)
    return H @ np.diag([1280.0 / width, 1280.0 / width, 1.0]), n


def _save_phase(pc: Path | None, t: np.ndarray, phase: np.ndarray) -> None:
    if pc is None:
        return
    try:   # caching must never fail a run
        pc.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pc, t=t, phase=phase)
    except OSError:
        pass


def analyse(video_path: str, cfg: PerceptionConfig | None = None, model=None, progress=None,
            deadline: float | None = None, reserve_decode: float = 0.0) -> dict:
    """Everything the rules, the renderer and the website need for one video."""
    scene = Scene.load()
    info = video_info(video_path)
    H, n_inliers = _first_frame_H(video_path, info["width"])
    reader = SignalReader(scene, H)
    lamp_t, lamp_s = [], []

    def hook(idx, t, frame):
        lamp_t.append(t)
        lamp_s.append(reader.ped_state(frame))

    obs, info = run_perception(video_path, cfg, model, progress, frame_hook=hook, deadline=deadline,
                               reserve_decode=reserve_decode)
    # refine the registration on the median background collected during the pass
    if "bg" in info:
        Hb, nb = register(info["bg"])
        if nb >= max(12, n_inliers // 2):
            H, n_inliers = Hb @ np.diag([1280.0 / info["width"]] * 2 + [1.0]), nb
    elif info.get("cache_hit"):
        H, n_inliers = video_to_reference(video_path, info["width"])   # old dev caches
    pc = _phase_cache(video_path)
    if lamp_t:
        phase_t, phase = np.asarray(lamp_t), phase_series(np.asarray(lamp_t), np.asarray(lamp_s))
        if "truncated_at" not in info:
            _save_phase(pc, phase_t, phase)
    elif pc is not None and pc.exists():
        d = np.load(pc)
        phase_t, phase = d["t"], d["phase"]
    else:  # detector cache hit but no stored phase: one extra light pass
        phase_t, phase = read_phase(video_path, reader)
        _save_phase(pc, phase_t, phase)
    tracks = build_tracks(obs, H, (info["width"], info["height"]))
    mark_riders(tracks, obs)
    return {"info": info, "H": H, "n_inliers": n_inliers, "obs": obs, "tracks": tracks,
            "phase_t": phase_t, "phase": phase, "scene": scene}


def events_from_analysis(a: dict, enabled: set[str] | None = None) -> tuple[list[list], dict]:
    ctx = Context(a["tracks"], a["scene"], a["info"]["duration"], a["phase_t"], a["phase"], FlowPrior.load())
    raw = detect_all(ctx, enabled or ENABLED)
    return finalize(raw, ctx.duration, MERGE_GAP, MIN_LEN), ctx.debug


def detect(video_path: str, **kw) -> list[list]:
    a = analyse(video_path, **kw)
    events = events_from_analysis(a)[0]
    cut = a["info"].get("truncated_at")
    if cut is not None:   # stopped early for time: drop events touching the cut (their end is unknown)
        events = [e for e in events if e[1] < cut - 1.0]
    return events
