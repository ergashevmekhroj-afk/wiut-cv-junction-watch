"""Exploratory data analysis of the sample videos -> site/data/eda.json + images.

    python tools/eda.py --videos samples --out site/data

Everything here is computed from the videos and our cached detections/tracks;
the website reads the JSON and images.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.traffic.flow import CELL, FlowPrior  # noqa: E402
from src.traffic.perception import C, CLASS_NAMES, PERSON  # noqa: E402
from src.traffic.pipeline import analyse  # noqa: E402
from src.traffic.scene import REF_H, REF_W, SCENE_DIR  # noqa: E402
from src.traffic.signal import GREEN, RED  # noqa: E402

GROUPS = {"car": [2], "bus": [5], "truck": [7], "motorbike/bicycle": [1, 3], "person": [0]}


def brightness_series(video: str, step_sec: float = 5.0) -> list[list[float]]:
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in range(0, n, max(1, int(fps * step_sec))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if ok:
            hsv = cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2HSV)
            out.append([round(i / fps, 1), round(float(hsv[..., 2].mean()), 1)])
    cap.release()
    return out


def counts_over_time(obs: np.ndarray, duration: float, bin_sec: float = 5.0) -> dict:
    bins = np.arange(0, duration + bin_sec, bin_sec)
    res = {"t": [round(float(b), 1) for b in bins[:-1]]}
    frames = np.unique(obs[:, C["frame"]])
    for name, ids in GROUPS.items():
        m = np.isin(obs[:, C["cls"]], ids)
        # mean number of objects visible per processed frame in each bin
        per_frame_t = {}
        for f, t in zip(obs[m, C["frame"]], obs[m, C["t"]]):
            per_frame_t.setdefault(f, [t, 0])[1] += 1
        tf = np.array([v[0] for v in per_frame_t.values()]) if per_frame_t else np.zeros(0)
        cf = np.array([v[1] for v in per_frame_t.values()]) if per_frame_t else np.zeros(0)
        all_t = obs[np.isin(obs[:, C["frame"]], frames), C["t"]]
        vals = []
        for a, b in zip(bins[:-1], bins[1:]):
            nf = len(np.unique(obs[(all_t >= a) & (all_t < b), C["frame"]])) if len(all_t) else 0
            s = cf[(tf >= a) & (tf < b)].sum()
            vals.append(round(float(s / nf), 2) if nf else 0.0)
        res[name] = vals
    return res


def heatmap_png(tracks, out: Path, kind: str) -> None:
    acc = np.zeros((REF_H // 4, REF_W // 4), np.float32)
    for k in tracks:
        if (kind == "vehicle") != k.is_vehicle or (kind == "person" and not k.is_person):
            continue
        mv = k.speed > (3 if kind == "person" else 8)
        for x, y in (k.xy[mv] / 4).astype(int):
            if 0 <= x < acc.shape[1] and 0 <= y < acc.shape[0]:
                acc[y, x] += 1
    acc = cv2.GaussianBlur(acc, (0, 0), 2.5)
    acc = np.log1p(acc)
    acc = (255 * acc / max(acc.max(), 1e-6)).astype(np.uint8)
    col = cv2.applyColorMap(cv2.resize(acc, (REF_W, REF_H)), cv2.COLORMAP_INFERNO)
    ref = cv2.imread(str(SCENE_DIR / "reference.jpg"))
    mask = (cv2.resize(acc, (REF_W, REF_H)) > 8)[..., None]
    img = np.where(mask, cv2.addWeighted(ref, 0.35, col, 0.65, 0), (ref * 0.45).astype(np.uint8))
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def flow_png(out: Path) -> None:
    fp = FlowPrior.load()
    img = (cv2.imread(str(SCENE_DIR / "reference.jpg")) * 0.5).astype(np.uint8)
    for i in range(fp.count.shape[0]):
        for j in range(fp.count.shape[1]):
            if fp.count[i, j] < 30:
                continue
            c = np.array([j * CELL + CELL / 2, i * CELL + CELL / 2])
            d = fp.dir[i, j]
            ang = (np.degrees(np.arctan2(d[1], d[0])) % 360) / 2
            col = cv2.cvtColor(np.uint8([[[ang, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
            coh = fp.coherence[i, j]
            L = 12 + 10 * coh
            cv2.arrowedLine(img, tuple((c - d * L / 2).astype(int)), tuple((c + d * L / 2).astype(int)),
                            col if coh > 0.8 else (140, 140, 140), 2 if coh > 0.8 else 1, cv2.LINE_AA, tipLength=0.35)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def trajectories_png(tracks, out: Path) -> None:
    img = (cv2.imread(str(SCENE_DIR / "reference.jpg")) * 0.55).astype(np.uint8)
    for k in tracks:
        if len(k.t) < 10 or np.linalg.norm(k.xy[-1] - k.xy[0]) < 40:
            continue
        d = k.xy[-1] - k.xy[0]
        ang = (np.degrees(np.arctan2(d[1], d[0])) % 360) / 2
        col = (235, 235, 235) if k.is_person else cv2.cvtColor(np.uint8([[[ang, 230, 255]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
        cv2.polylines(img, [k.xy.astype(np.int32)], False, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def signal_cycles(t: np.ndarray, ph: np.ndarray) -> dict:
    ch = np.flatnonzero(np.diff(ph)) + 1
    b = [0, *ch, len(ph)]
    runs = [(int(ph[b[i]]), float(t[b[i]]), float(t[min(b[i + 1], len(t) - 1)])) for i in range(len(b) - 1)]
    inner = runs[1:-1]   # first and last runs are cut by the video boundaries
    g = [e - s for p, s, e in inner if p == GREEN]
    r = [e - s for p, s, e in inner if p == RED]
    return {"runs": [[p, round(s, 1), round(e, 1)] for p, s, e in runs],
            "green_mean": round(float(np.mean(g)), 1) if g else None,
            "red_mean": round(float(np.mean(r)), 1) if r else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="site/data")
    ap.add_argument("--pickles", help="use saved analyse() results (a01.pkl ...) instead of re-running")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    report = {"videos": {}}
    all_tracks = []
    for v in sorted(Path(a.videos).glob("*.mp4")):
        if a.pickles:
            import pickle
            an = pickle.load(open(Path(a.pickles) / f"a{v.stem[-2:]}.pkl", "rb"))
        else:
            an = analyse(str(v))
        info, tracks, obs = an["info"], an["tracks"], an["obs"]
        all_tracks += tracks
        orig = json.loads((Path(a.videos) / "original_info.json").read_text()).get(v.name) \
            if (Path(a.videos) / "original_info.json").exists() else None
        veh = [k for k in tracks if k.is_vehicle and len(k.t) >= 10]
        ped = [k for k in tracks if k.is_person and len(k.t) >= 10]
        sp_v = np.concatenate([k.speed[k.speed > 8] for k in veh]) if veh else np.zeros(0)
        sp_p = np.concatenate([k.speed[k.speed > 3] for k in ped]) if ped else np.zeros(0)
        cls_counts = {}
        for name, ids in GROUPS.items():
            cls_counts[name] = int(sum(1 for k in tracks if k.cls in ids and len(k.t) >= 10))
        entry = {
            "duration": round(info["duration"], 2), "fps": round(info["fps"], 3),
            "proxy_resolution": [info["width"], info["height"]],
            "original": ({"resolution": [orig["streams"][0]["width"], orig["streams"][0]["height"]],
                          "size_gb": round(int(orig["format"]["size"]) / 1e9, 2),
                          "codec": orig["streams"][0]["codec_name"]} if orig else None),
            "registration_inliers": an["n_inliers"],
            "homography": np.round(an["H"], 4).tolist(),
            "brightness": brightness_series(str(v)),
            "counts": counts_over_time(obs, info["duration"]),
            "tracks_by_class": cls_counts,
            "vehicle_speed_hist": np.histogram(sp_v, bins=np.arange(0, 260, 10))[0].tolist(),
            "person_speed_hist": np.histogram(sp_p, bins=np.arange(0, 65, 2.5))[0].tolist(),
            "signal": signal_cycles(an["phase_t"], an["phase"]),
            "detections": int(len(obs)),
        }
        report["videos"][v.name] = entry
        heatmap_png(tracks, out / "img" / f"heat_vehicle_{v.stem}.jpg", "vehicle")
        heatmap_png(tracks, out / "img" / f"heat_person_{v.stem}.jpg", "person")
        trajectories_png(tracks, out / "img" / f"traj_{v.stem}.jpg")
        print(v.name, "done")
    heatmap_png(all_tracks, out / "img" / "heat_vehicle_all.jpg", "vehicle")
    heatmap_png(all_tracks, out / "img" / "heat_person_all.jpg", "person")
    trajectories_png(all_tracks, out / "img" / "traj_all.jpg")
    flow_png(out / "img" / "flow_prior.jpg")
    report["speed_bins"] = {"vehicle": list(range(0, 250, 10)), "person": list(np.arange(0, 62.5, 2.5).round(1))}
    (out / "eda.json").write_text(json.dumps(report))
    print("wrote", out / "eda.json")


if __name__ == "__main__":
    main()
