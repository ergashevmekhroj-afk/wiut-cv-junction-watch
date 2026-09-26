"""Contact sheets for reviewing candidate events by eye.

    python tools/review.py VIDEO --tid 1068 --t 73.3 --out review.jpg
Shows 6 frames around t (t-2 .. t+3 s), cropped around the track, with its box drawn.
"""
import argparse, sys
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sheet(video, obs_tracks, tid, t, offsets=(-2, -1, 0, 1, 2, 3), pad=160, others=True, label=""):
    cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS)
    k = next((x for x in obs_tracks if x.tid == tid), None)
    tiles = []
    for o in offsets:
        tt = t + o
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(tt * fps)); ok, f = cap.read()
        if not ok:
            continue
        cx, cy = f.shape[1] // 2, f.shape[0] // 2
        if k is not None:
            j = k.at(tt); b = k.box[j].astype(int)
            cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
            if abs(k.t[j] - tt) < 0.3:
                cv2.rectangle(f, tuple(b[:2]), tuple(b[2:]), (0, 0, 255), 2)
        if others:
            for x in obs_tracks:
                if x is k or not (x.t[0] <= tt <= x.t[-1]):
                    continue
                j = x.at(tt)
                if abs(x.t[j] - tt) < 0.3:
                    b = x.box[j].astype(int)
                    cv2.rectangle(f, tuple(b[:2]), tuple(b[2:]), (0, 255, 0), 1)
                    cv2.putText(f, str(x.tid), (b[0], b[1] - 2), 0, 0.35, (0, 255, 0), 1)
        x0, y0 = max(0, cx - pad * 2), max(0, cy - pad)
        c = f[y0:y0 + 2 * pad, x0:x0 + 4 * pad].copy()
        c = cv2.resize(c, (4 * pad, 2 * pad)) if c.size else np.zeros((2 * pad, 4 * pad, 3), np.uint8)
        cv2.putText(c, f"{label} t={tt:.1f}s", (5, 18), 0, 0.55, (0, 255, 255), 2)
        tiles.append(c)
    cap.release()
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    return np.vstack(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("video"); ap.add_argument("--tid", type=int, required=True)
    ap.add_argument("--t", type=float, required=True); ap.add_argument("--out", default="review.jpg")
    a = ap.parse_args()
    from src.traffic.pipeline import analyse
    res = analyse(a.video)
    cv2.imwrite(a.out, sheet(a.video, res["tracks"], a.tid, a.t))
