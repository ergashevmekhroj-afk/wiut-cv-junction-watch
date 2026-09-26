"""Calibrate the Part B danger->risk mapping on NORMAL traffic (the sample videos).

    python tools/calibrate_risk.py --videos samples     # -> scene/risk_calibration.json

Replays the cached Part A tracks through the Part B danger function at the
Part B rate (causal: only past samples of each track are used), then places the
mapping knots at quantiles of the observed danger:
    median -> 0.02, p99 -> 0.15, p99.9 -> 0.30, max -> 0.45,
    1.5 x max -> 0.60, 3 x max -> 0.90.
So the alarm threshold 0.5 is crossed only by conflicts more extreme than any
moment in the samples.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.traffic.pipeline import analyse
from src.traffic.risk import CausalRisk
from src.traffic.scene import SCENE_DIR


def replay(a, fps=7.5):
    r = CausalRisk()
    ts = np.arange(0, a["info"]["duration"], 1 / fps)
    out = []
    for t in ts:
        r.hist, seen = {}, set()
        for k in a["tracks"]:
            if k.t[0] <= t <= k.t[-1]:
                m = (k.t <= t) & (k.t > t - 1.6)
                if m.sum() >= 4:
                    r.hist[k.tid] = [(tt, p, k.cls, float(np.linalg.norm(w))) for tt, p, w in zip(k.t[m], k.xy[m], k.wh[m])]
                    seen.add(k.tid)
        out.append(r._danger(t, seen))
    return ts, np.array(out)


def live(video: str):
    """Run the real causal Part B estimator over the video and record its danger values."""
    import cv2
    from src.traffic.perception import video_info
    info = video_info(video)
    r = CausalRisk()
    r.reset({"video_id": Path(video).name, "fps": info["fps"], "width": info["width"],
             "height": info["height"], "n_frames": info["n_frames"]})
    dangers, ts = [], []
    orig = r._raw_risk
    def spy(t, seen):
        d = r._danger(t, seen); dangers.append(d); ts.append(t)
        return r.calib(d)
    r._raw_risk = spy
    cap = cv2.VideoCapture(video); i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r.step(f, i / info["fps"]); i += 1
    cap.release()
    return np.array(ts), np.array(dangers)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--videos", required=True)
    ap.add_argument("--live", action="store_true", help="run the real Part B detector/tracker (slow on CPU)")
    a = ap.parse_args()
    allv = []
    for v in sorted(Path(a.videos).glob("*.mp4")):
        ts, d = live(str(v)) if a.live else replay(analyse(str(v)))
        (SCENE_DIR.parent / ".cache").mkdir(exist_ok=True)
        np.save(SCENE_DIR.parent / ".cache" / f"danger_{v.stem}_{'live' if a.live else 'replay'}.npy", np.stack([ts, d]))
        allv.append(d)
        print(v.name, "p50 %.1f p99 %.1f p99.9 %.1f max %.1f" % (np.median(d), *np.percentile(d, [99, 99.9]), d.max()))
    d = np.concatenate(allv)
    q50, q99, q999, mx = np.median(d), *np.percentile(d, [99, 99.9]), d.max()
    x = [0.0, max(q50, 1e-3), q99, q999, mx, 1.5 * mx, 3 * mx]
    x = list(np.maximum.accumulate(np.array(x) + np.arange(len(x)) * 1e-3))
    cal = {"x": [round(float(v), 3) for v in x], "y": [0.0, 0.02, 0.15, 0.30, 0.45, 0.60, 0.90],
           "n_samples": int(len(d))}
    (SCENE_DIR / "risk_calibration.json").write_text(json.dumps(cal, indent=1))
    print("saved", cal)
