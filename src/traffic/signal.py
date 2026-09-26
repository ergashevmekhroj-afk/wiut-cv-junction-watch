"""Traffic-signal phase from the pedestrian signal head that faces the camera.

The vehicle signal heads on the gantry face away from the camera, but the
pedestrian head on the left pole (reference px ~(181, 328) red lamp,
~(181, 340) green lamp) faces it and is clearly readable. It controls the
side-road zebra, which runs in the SAME phase as the main carriageway. We
verified this on sample_04: all 50+ stop-line crossings of the main
carriageway happen while this lamp is green (EDA notebook), so

    pedestrian GREEN  <=>  main-carriageway vehicles GREEN

The reader works on a small ROI around each lamp mapped into native pixels,
so the per-frame cost is negligible even on 4K frames.
"""
from __future__ import annotations

import cv2
import numpy as np

from .scene import Scene, warp_points

RED, GREEN, UNKNOWN = 1, 0, -1


def lamp_scores(frame: np.ndarray, px_red: np.ndarray, px_green: np.ndarray, r: int) -> tuple[float, float]:
    """(redness of red lamp, greenness of green lamp) — max over a small window."""
    def win(p):
        x, y = int(round(p[0])), int(round(p[1]))
        h, w = frame.shape[:2]
        return frame[max(0, y - r):min(h, y + r + 1), max(0, x - r):min(w, x + r + 1)].astype(np.int16)
    a, b = win(px_red), win(px_green)
    red = float((a[..., 2] - np.maximum(a[..., 1], a[..., 0])).max()) if a.size else 0.0
    grn = float((b[..., 1] - np.maximum(b[..., 2], b[..., 0])).max()) if b.size else 0.0
    return red, grn


class SignalReader:
    """Maps lamp positions from reference into native pixels for one video."""

    def __init__(self, scene: Scene, H_native_to_ref: np.ndarray, lamp: str = "ped_left"):
        Hinv = np.linalg.inv(H_native_to_ref)
        self.p_red = warp_points(Hinv, scene.points[f"{lamp}_red"])[0]
        self.p_green = warp_points(Hinv, scene.points[f"{lamp}_green"])[0]
        scale = np.sqrt(abs(np.linalg.det(Hinv[:2, :2])))     # native px per reference px
        self.r = max(2, int(round(4 * scale)))
        self.red_on = scene.meta.get("lamp_red_on", 45.0)
        self.green_on = scene.meta.get("lamp_green_on", 28.0)

    def ped_state(self, frame: np.ndarray) -> int:
        """Pedestrian lamp state: RED, GREEN or UNKNOWN (both dark / both lit)."""
        red, grn = lamp_scores(frame, self.p_red, self.p_green, self.r)
        if red >= self.red_on and grn < self.green_on:
            return RED
        if grn >= self.green_on and red < self.red_on:
            return GREEN
        return UNKNOWN


def phase_series(t: np.ndarray, ped: np.ndarray, hold_sec: float = 3.0) -> np.ndarray:
    """Clean a raw pedestrian-state series: fill UNKNOWN from neighbours, drop blips.

    Returns the main-carriageway VEHICLE state per sample (RED/GREEN/UNKNOWN).
    Flashing pedestrian green (end of the phase) alternates GREEN/UNKNOWN, which
    the fill step keeps as GREEN; the vehicle amber is handled by the rules.
    """
    s = ped.copy()
    # forward-fill unknowns with last known state
    last = UNKNOWN
    for i in range(len(s)):
        if s[i] == UNKNOWN:
            s[i] = last
        else:
            last = s[i]
    # remove runs shorter than hold_sec
    if len(s) > 2:
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            if 0 < i and j < len(s) - 1 and t[j] - t[i] < hold_sec:
                s[i:j + 1] = s[i - 1]
            i = j + 1
    return s  # same phase as the main carriageway (see module docstring)


def read_phase(video_path: str, reader: SignalReader, sample_fps: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Scan the whole video at `sample_fps`; returns (t, vehicle_state)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
    ts, st, i = [], [], 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                ts.append(i / fps)
                st.append(reader.ped_state(f))
        i += 1
    cap.release()
    t = np.asarray(ts)
    return t, phase_series(t, np.asarray(st))
