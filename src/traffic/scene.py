"""Scene geometry of the fixed camera, and per-video registration to it.

All geometry (lanes, stop lines, crossings, signal lamps) is drawn ONCE on a
reference view (`scene/reference.jpg`, 1280x720, the median background of
sample_01). The sample videos are not framed identically (samples 03/04 are
zoomed out and shifted by ~40 px relative to 01/02), so every video is
registered to the reference with a SIFT homography computed on its own median
background. Track points are then mapped into reference coordinates and all
rules work in one coordinate system.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCENE_DIR = REPO / "scene"
REF_W, REF_H = 1280, 720


@dataclass
class Scene:
    polygons: dict[str, np.ndarray] = field(default_factory=dict)   # name -> (N,2) ref px
    lines: dict[str, np.ndarray] = field(default_factory=dict)      # name -> (2,2) ref px
    points: dict[str, np.ndarray] = field(default_factory=dict)     # name -> (2,) ref px
    meta: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None = None) -> "Scene":
        d = json.loads((path or SCENE_DIR / "scene.json").read_text())
        return cls(
            polygons={k: np.asarray(v, float) for k, v in d.get("polygons", {}).items()},
            lines={k: np.asarray(v, float) for k, v in d.get("lines", {}).items()},
            points={k: np.asarray(v, float) for k, v in d.get("points", {}).items()},
            meta=d.get("meta", {}),
        )

    def inside(self, name: str, pts: np.ndarray) -> np.ndarray:
        """Boolean mask: which of pts (N,2) lie inside polygon `name`."""
        poly = self.polygons.get(name)
        pts = np.atleast_2d(pts)
        if poly is None:
            return np.zeros(len(pts), bool)
        path = poly.astype(np.float32).reshape(-1, 1, 2)
        return np.array([cv2.pointPolygonTest(path, (float(x), float(y)), False) >= 0 for x, y in pts])

    def side(self, name: str, pts: np.ndarray) -> np.ndarray:
        """Signed side of directed line `name` (a->b): >0 left, <0 right (image coords)."""
        (ax, ay), (bx, by) = self.lines[name]
        pts = np.atleast_2d(pts)
        return (bx - ax) * (pts[:, 1] - ay) - (by - ay) * (pts[:, 0] - ax)


# ----------------------------------------------------------------------------
# registration
# ----------------------------------------------------------------------------
def median_background(video_path: str, n: int = 25, width: int = REF_W) -> np.ndarray:
    """Median of n evenly spaced frames (moving vehicles vanish), resized to `width`."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(0, total - 2), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            h = int(round(f.shape[0] * width / f.shape[1]))
            frames.append(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA))
    cap.release()
    return np.median(np.stack(frames), 0).astype(np.uint8)


def register(image: np.ndarray, reference: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Homography mapping `image` px -> reference px. Returns (H, n_inliers).

    Falls back to identity (scaled) if registration is unreliable.
    """
    if reference is None:
        reference = cv2.imread(str(SCENE_DIR / "reference.jpg"))
    sx = REF_W / image.shape[1]
    S = np.diag([sx, sx, 1.0])
    small = cv2.resize(image, (REF_W, int(round(image.shape[0] * sx))), interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(6000)
    g1 = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return S, 0
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=64))
    good = [m for m, n in (p for p in flann.knnMatch(d1, d2, k=2) if len(p) == 2) if m.distance < 0.7 * n.distance]
    if len(good) < 15:
        return S, 0
    p1 = np.float32([k1[m.queryIdx].pt for m in good])
    p2 = np.float32([k2[m.trainIdx].pt for m in good])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 3.0)
    if H is None or inl.sum() < 12 or not _plausible(H):
        return S, 0
    return H @ S, int(inl.sum())


def _plausible(H: np.ndarray) -> bool:
    """Same camera, same angle: expect near-similarity with modest zoom/shift."""
    A = H[:2, :2]
    s = np.sqrt(abs(np.linalg.det(A)))
    return 0.7 < s < 1.4 and abs(H[2, 0]) < 1e-3 and abs(H[2, 1]) < 1e-3 and np.abs(H[:2, 2]).max() < 300


def warp_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)


def video_to_reference(video_path: str, native_width: int) -> tuple[np.ndarray, int]:
    """Homography from the video's NATIVE pixel coords to reference coords."""
    bg = median_background(video_path)                     # REF_W wide
    H, n = register(bg)
    s = bg.shape[1] / float(native_width)
    return H @ np.diag([s, s, 1.0]), n


def polygon_depth(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Signed distance (ref px) of pts to the polygon boundary: >0 inside, <0 outside."""
    path = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    return np.array([cv2.pointPolygonTest(path, (float(x), float(y)), True) for x, y in np.atleast_2d(pts)])
