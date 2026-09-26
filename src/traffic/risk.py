"""Part B: causal accident-risk score from time-to-collision between tracked road users.

Strictly causal: the estimator only sees frames as the harness streams them.
  * registration to the reference view uses the FIRST frame only (no median
    background, which would need future frames);
  * a small detector + ByteTrack runs every `stride`-th frame on a downscaled copy;
  * per pair of road users we compute the constant-velocity time-to-collision
    (TTC) and the closing speed; the risk is a calibrated function of the most
    dangerous pair, smoothed with a short causal filter.

Calibration: an alarm (score >= 0.5) should mean "a collision is likely within
5 s". Normal dense traffic at this junction constantly has short gaps between
queued or following cars, so only CROSSING conflicts at speed (heading
difference >= 30 deg, or a pedestrian in the path of a moving vehicle) can push
the score past 0.5; everything else stays in the low range, where it still
orders frames for the AP metric.
"""
from __future__ import annotations

import numpy as np

from .perception import PERSON, VEHICLES, WEIGHTS, reset_tracker
from .rules.interaction_rules import ttc
from .scene import REF_W, register, warp_points

RISK_CLASSES = [PERSON, 1, 2, 3, 5, 7]


class CausalRisk:
    def __init__(self, model_name: str = "yolo11s.pt", imgsz: int = 640, target_fps: float = 7.5,
                 device: str | None = None):
        self.model_name, self.imgsz, self.target_fps = model_name, imgsz, target_fps
        self.device = device
        self.model = None

    # ------------------------------------------------------------------
    def reset(self, meta: dict) -> None:
        from ultralytics import YOLO
        if self.model is None:
            p = WEIGHTS / self.model_name
            self.model = YOLO(str(p if p.exists() else self.model_name))
            if self.device is None:
                try:
                    import torch
                    self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
                except Exception:
                    self.device = "cpu"
        reset_tracker(self.model)
        self.fps = float(meta.get("fps") or 25.0)
        self.stride = max(1, int(round(self.fps / self.target_fps)))
        self.width = int(meta.get("width") or 1920)
        self.idx = 0
        self.H = None
        self.hist: dict[int, list[tuple[float, np.ndarray, int]]] = {}   # tid -> [(t, ref xy, cls)]
        self.score = 0.0
        self.raw_hist: list[float] = []

    # ------------------------------------------------------------------
    def step(self, frame: np.ndarray, t_sec: float) -> float:
        i = self.idx
        self.idx += 1
        if i % self.stride:
            return self.score
        import cv2
        scale = REF_W / frame.shape[1]
        small = cv2.resize(frame, (REF_W, int(round(frame.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)
        if self.H is None:
            H, n = register(small)
            self.H = H          # small px -> reference px (identity-scaled fallback if n == 0)
        res = self.model.track(small, persist=True, imgsz=self.imgsz, conf=0.3, classes=RISK_CLASSES,
                               device=self.device, verbose=False, tracker="bytetrack.yaml")[0]
        b = res.boxes
        seen = set()
        if b is not None and b.id is not None and len(b):
            xyxy = b.xyxy.cpu().numpy()
            ids = b.id.cpu().numpy().astype(int)
            cls = b.cls.cpu().numpy().astype(int)
            foot = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, xyxy[:, 3] - 0.25 * (xyxy[:, 3] - xyxy[:, 1])], 1)
            ref = warp_points(self.H, foot)
            size = np.linalg.norm(xyxy[:, 2:] - xyxy[:, :2], axis=1) * np.sqrt(abs(np.linalg.det(self.H[:2, :2])))
            for tid, c, p, s in zip(ids, cls, ref, size):
                h = self.hist.setdefault(int(tid), [])
                h.append((t_sec, p, int(c), float(s)))
                if len(h) > 12:
                    del h[0]
                seen.add(int(tid))
        for tid in list(self.hist):
            if tid not in seen and t_sec - self.hist[tid][-1][0] > 1.0:
                del self.hist[tid]
        raw = self._raw_risk(t_sec, seen)
        # causal smoothing: mean of the last ~0.6 s of raw scores (suppresses one-frame spikes)
        self.raw_hist.append(raw)
        k = max(1, int(round(0.6 * self.target_fps)))
        self.score = float(np.mean(self.raw_hist[-k:]))
        return self.score

    # ------------------------------------------------------------------
    def _state(self, tid: int):
        h = self.hist[tid]
        if len(h) < 4:
            return None
        t = np.array([x[0] for x in h])
        p = np.array([x[1] for x in h])
        if t[-1] - t[0] < 0.3:
            return None
        v = (p[-1] - p[-4]) / max(1e-3, t[-1] - t[-4])
        return p[-1], v, h[-1][2], h[-1][3]

    def _raw_risk(self, t: float, seen: set[int]) -> float:
        return self.calib(self._danger(t, seen))

    def _danger(self, t: float, seen: set[int]) -> float:
        """Most dangerous pair right now: closing speed x urgency, crossing conflicts weighted up."""
        states = {tid: s for tid in seen if (s := self._state(tid)) is not None}
        ids = list(states)
        best = 0.0
        for a in range(len(ids)):
            pa, va, ca, sa = states[ids[a]]
            for b in range(a + 1, len(ids)):
                pb, vb, cb, sb = states[ids[b]]
                if ca not in VEHICLES and cb not in VEHICLES:
                    continue
                d = np.linalg.norm(pb - pa)
                if d > 200:
                    continue
                r = 0.3 * (sa + sb)
                if d < r:                 # already overlapping in the image: queue / occlusion
                    continue
                tc = ttc(pa, va, pb, vb, r)
                if not np.isfinite(tc) or tc > 3.0:
                    continue
                closing = -((pb - pa) @ (vb - va)) / max(d, 1e-6)
                if closing <= 0:
                    continue
                spa, spb = np.linalg.norm(va), np.linalg.norm(vb)
                crossing = ca == PERSON or cb == PERSON
                if spa > 10 and spb > 10:
                    crossing = crossing or (va @ vb) / (spa * spb) < np.cos(np.radians(30))
                w = 1.0 if crossing else 0.4
                best = max(best, w * closing * np.exp(-tc / 1.0))
        return float(best)

    # danger -> probability. Knots are set from the danger distribution of NORMAL
    # traffic in the four sample videos (tools/calibrate_risk.py): everything seen
    # there stays below 0.45; only danger beyond the normal maximum raises an alarm.
    CALIB = None

    def calib(self, danger: float) -> float:
        if CausalRisk.CALIB is None:
            import json
            from .scene import SCENE_DIR
            p = SCENE_DIR / "risk_calibration.json"
            CausalRisk.CALIB = json.loads(p.read_text()) if p.exists() else {"x": [0, 100, 200, 400], "y": [0, 0.2, 0.5, 0.9]}
        c = CausalRisk.CALIB
        return float(np.interp(danger, c["x"], c["y"]))
