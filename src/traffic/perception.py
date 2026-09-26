"""Detection + multi-object tracking over a sampled video.

Output is a flat observation table (one row per detected box per sampled
frame) that every rule module reads. It is cached on disk so that rules can
be re-tuned without re-running the detector.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
WEIGHTS = REPO / "weights"
TRACKER_CFG = REPO / "src" / "traffic" / "bytetrack.yaml"

# COCO ids we care about
PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK = 0, 1, 2, 3, 5, 7
VEHICLES = (CAR, MOTORCYCLE, BUS, TRUCK)
ROAD_USERS = (PERSON, BICYCLE, CAR, MOTORCYCLE, BUS, TRUCK)
ANIMALS = tuple(range(14, 24))          # bird .. giraffe (dog, cat, horse, cow, ...)
KEEP_CLASSES = list(ROAD_USERS) + list(ANIMALS)
CLASS_NAMES = {PERSON: "person", BICYCLE: "bicycle", CAR: "car", MOTORCYCLE: "motorcycle",
               BUS: "bus", TRUCK: "truck"}

# columns of the observation table
COLS = ["frame", "t", "tid", "cls", "x1", "y1", "x2", "y2", "conf"]
C = {name: i for i, name in enumerate(COLS)}


@dataclass
class PerceptionConfig:
    model: str = "yolo11s.pt"
    imgsz: int = 960           # CCTV objects are small; 960 keeps far vehicles
    conf: float = 0.25
    target_fps: float = 10.0   # detector rate (every 3rd frame at 29.97 fps)
    device: str | None = None  # auto: cuda if available

    def key(self) -> str:
        return f"{self.model}-{self.imgsz}-{self.conf}-{self.target_fps}-w1280"


def _device(cfg: PerceptionConfig) -> str:
    if cfg.device:
        return cfg.device
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def load_model(cfg: PerceptionConfig):
    from ultralytics import YOLO
    path = WEIGHTS / cfg.model
    return YOLO(str(path if path.exists() else cfg.model))


def reset_tracker(model) -> None:
    """Ultralytics keeps tracker state on the model between calls; clear it per video."""
    pred = getattr(model, "predictor", None)
    for tr in getattr(pred, "trackers", None) or []:
        tr.reset()


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": float(fps), "n_frames": n, "width": w, "height": h, "duration": n / float(fps)}


def _cache_path(video_path: str, cfg: PerceptionConfig) -> Path | None:
    root = os.environ.get("TRAFFIC_CACHE", str(REPO / ".cache"))
    if root.lower() == "off":
        return None
    st = os.stat(video_path)
    h = hashlib.md5(f"{Path(video_path).name}-{st.st_size}-{cfg.key()}".encode()).hexdigest()[:12]
    return Path(root) / f"{Path(video_path).stem}-{h}.npz"


DET_WIDTH = 1280   # frames are downscaled to this width before the detector (4K letterboxing is slow)
BG_FRAMES = 25     # frames kept (downscaled) for the median background


def _reader(cap, stride: int, fps: float, q, stop, frame_hook, bg_every: int, stats: dict):
    """Decode thread: grab every frame, retrieve every `stride`-th, downscale, queue it.

    OpenCV releases the GIL while decoding, so this overlaps with the detector
    running on the GPU in the main thread.
    """
    import time as _time
    import cv2 as _cv2
    idx = 0
    try:
        while not stop.is_set():
            t0 = _time.perf_counter()
            if not cap.grab():
                break
            stats["grab"] += _time.perf_counter() - t0
            stats["n"] = idx + 1
            if idx % stride == 0:
                t1 = _time.perf_counter()
                ok, frame = cap.retrieve()
                stats["retrieve"] += _time.perf_counter() - t1
                if not ok:
                    break
                if frame_hook is not None:
                    frame_hook(idx, idx / fps, frame)          # full-resolution (signal lamp ROI)
                h, w = frame.shape[:2]
                small = frame if w <= DET_WIDTH else _cv2.resize(
                    frame, (DET_WIDTH, int(round(h * DET_WIDTH / w))), interpolation=_cv2.INTER_LINEAR)
                q.put((idx, small, (idx // stride) % bg_every == 0))
            idx += 1
    finally:
        q.put(None)


def run_perception(video_path: str, cfg: PerceptionConfig | None = None, model=None,
                   progress=None, frame_hook=None, deadline: float | None = None,
                   reserve_decode: float = 0.0) -> tuple[np.ndarray, dict]:
    """Detect and track every road user. Returns (obs table [N, 9] in NATIVE px, video info).

    One decoding pass. frame_hook(idx, t, frame) is called on every sampled
    full-resolution frame (signal lamp). info["bg"] is the median of ~25 sampled
    frames (for registration). If `deadline` (time.perf_counter()) passes, the
    pass stops early and returns what it has (info["truncated_at"]).
    reserve_decode > 0 moves that deadline earlier by reserve_decode x the
    measured time to decode (and colour-convert) the whole video once, so a
    second full decoding pass (the harness streaming frames to Part B) still
    fits in the budget on slow machines.
    On a cache hit, no frame is decoded and frame_hook is not called.
    """
    import queue
    import threading
    import time

    cfg = cfg or PerceptionConfig()
    info = video_info(video_path)
    cache = _cache_path(video_path, cfg)
    if cache is not None and cache.exists():
        d = np.load(cache)
        info["cache_hit"] = True
        if "bg" in d.files:
            info["bg"] = d["bg"]
        return d["obs"], info

    model = model or load_model(cfg)
    reset_tracker(model)
    device = _device(cfg)
    stride = max(1, round(info["fps"] / cfg.target_fps))
    info["stride"] = stride
    n_sampled = max(1, info["n_frames"] // stride)
    bg_every = max(1, n_sampled // BG_FRAMES)

    cap = cv2.VideoCapture(video_path)
    q: queue.Queue = queue.Queue(maxsize=6)
    stop = threading.Event()
    stats = {"grab": 0.0, "retrieve": 0.0, "n": 0}
    th = threading.Thread(target=_reader, args=(cap, stride, info["fps"], q, stop, frame_hook, bg_every, stats), daemon=True)
    th.start()
    rows, bg = [], []
    scale = 1.0
    while True:
        item = q.get()
        if item is None:
            break
        idx, small, keep_bg = item
        scale = info["width"] / small.shape[1]
        if keep_bg and len(bg) < BG_FRAMES + 2:
            bg.append(small if small.shape[1] == 1280 else cv2.resize(small, (1280, int(round(small.shape[0] * 1280 / small.shape[1])))))
        res = model.track(small, persist=True, tracker=str(TRACKER_CFG), imgsz=cfg.imgsz,
                          conf=cfg.conf, classes=KEEP_CLASSES, device=device, verbose=False)[0]
        b = res.boxes
        if b is not None and b.id is not None and len(b):
            xyxy = b.xyxy.cpu().numpy() * scale
            ids = b.id.cpu().numpy()
            cls = b.cls.cpu().numpy()
            cf = b.conf.cpu().numpy()
            t = idx / info["fps"]
            for k in range(len(ids)):
                rows.append([idx, t, ids[k], cls[k], *xyxy[k], cf[k]])
        if progress and (idx // stride) % 50 == 0:
            progress(idx / max(1, info["n_frames"]))
        eff_deadline = deadline
        if deadline is not None and reserve_decode > 0 and stats["n"] > 50:
            # per-frame cost of a full read(): grab + colour conversion (measured on the sampled frames)
            per_frame = stats["grab"] / stats["n"] + stats["retrieve"] / max(1, stats["n"] // stride)
            eff_deadline = deadline - reserve_decode * per_frame * info["n_frames"]
            info["decode_ms_per_frame"] = round(1000 * per_frame, 2)
        if eff_deadline is not None and time.perf_counter() > eff_deadline:
            info["truncated_at"] = idx / info["fps"]
            stop.set()
            while q.get() is not None:   # drain so the reader can exit
                pass
            break
    th.join(timeout=5)
    cap.release()
    obs = np.asarray(rows, dtype=np.float64).reshape(-1, len(COLS))
    if bg:
        info["bg"] = np.median(np.stack(bg), 0).astype(np.uint8)
    if cache is not None and "truncated_at" not in info:
        try:   # caching is a dev convenience: it must never fail a run
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, obs=obs, **({"bg": info["bg"]} if "bg" in info else {}))
        except OSError:
            pass
    return obs, info
