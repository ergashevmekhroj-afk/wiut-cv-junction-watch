"""Render an annotated copy of a video: boxes + track tails, scene layout, signal
phase, active events and the risk curve.

    python tools/render.py samples/sample_04.mp4 --pred predictions_samples.json --out renders/sample_04.mp4

Writes browser-friendly H.264 (via ffmpeg) at 1280 px width.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.traffic.perception import C, CLASS_NAMES, PERSON  # noqa: E402
from src.traffic.scene import warp_points  # noqa: E402
from src.traffic.signal import GREEN, RED  # noqa: E402

EVENT_COLORS = {  # BGR
    "accident": (40, 40, 230), "near_miss": (0, 140, 255), "red_light": (60, 60, 255),
    "wrong_way": (200, 0, 200), "illegal_u_turn": (255, 0, 150), "stopped_vehicle": (0, 200, 255),
    "jaywalking": (255, 200, 0), "failure_to_yield": (0, 230, 230), "illegal_turn": (180, 100, 255),
    "solid_line_crossing": (255, 150, 150), "stop_line": (80, 80, 255), "congestion": (0, 100, 180),
    "road_obstacle": (120, 200, 120), "fire_smoke": (50, 50, 150),
}
OUT_W = 1280


def render(video: str, out: str, events: list, risk: list, analysis: dict, tail_sec: float = 2.0) -> None:
    info = analysis["info"]
    H = analysis["H"]
    Hinv = np.linalg.inv(H)
    obs = analysis["obs"]
    scale = OUT_W / info["width"]
    out_h = int(round(info["height"] * scale)) // 2 * 2
    fps = info["fps"]
    by_frame: dict[int, np.ndarray] = {}
    for f in np.unique(obs[:, C["frame"]]).astype(int):
        by_frame[f] = obs[obs[:, C["frame"]] == f]
    frames_sorted = np.array(sorted(by_frame))
    # scene overlay in output pixels
    polys = {k: (warp_points(Hinv, p) * scale).astype(np.int32) for k, p in analysis["scene"].polygons.items()}
    stop = (warp_points(Hinv, analysis["scene"].lines["stop_main"]) * scale).astype(np.int32)
    risk_t = np.array([r[0] for r in risk]) if risk else np.zeros(0)
    risk_v = np.array([r[1] for r in risk]) if risk else np.zeros(0)
    tails: dict[int, list] = {}

    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{OUT_W}x{out_h}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264",
                             "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                             "-movflags", "+faststart", out], stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(video)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        img = cv2.resize(frame, (OUT_W, out_h), interpolation=cv2.INTER_AREA)
        lay = img.copy()
        for name in ("cross_main", "cross_upper", "cross_side"):
            cv2.polylines(lay, [polys[name]], True, (255, 255, 255), 1, cv2.LINE_AA)
        ph = analysis["phase"][min(len(analysis["phase"]) - 1, int(np.searchsorted(analysis["phase_t"], t)))] \
            if len(analysis["phase"]) else -1
        cv2.line(lay, tuple(stop[0]), tuple(stop[1]), (0, 0, 255) if ph == RED else (0, 220, 0), 2, cv2.LINE_AA)
        img = cv2.addWeighted(lay, 0.6, img, 0.4, 0)
        # detections at the nearest processed frame
        j = int(np.searchsorted(frames_sorted, idx, side="right")) - 1
        if j >= 0:
            for r in by_frame[int(frames_sorted[j])]:
                x1, y1, x2, y2 = (r[[C["x1"], C["y1"], C["x2"], C["y2"]]] * scale).astype(int)
                tid, cls = int(r[C["tid"]]), int(r[C["cls"]])
                col = (255, 200, 0) if cls == PERSON else (80, 255, 80)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
                cv2.putText(img, f"{tid}", (x1, max(10, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
                tl = tails.setdefault(tid, [])
                if not tl or tl[-1][0] != frames_sorted[j]:
                    tl.append((frames_sorted[j], ((x1 + x2) // 2, y2)))
        for tid in list(tails):
            tl = [p for p in tails[tid] if (idx - p[0]) / fps < tail_sec]
            tails[tid] = tl
            if len(tl) > 1:
                cv2.polylines(img, [np.array([p[1] for p in tl], np.int32)], False, (200, 200, 200), 1, cv2.LINE_AA)
        # HUD: signal + active events
        cv2.rectangle(img, (0, 0), (OUT_W, 26), (20, 20, 20), -1)
        cv2.putText(img, f"t = {t:6.1f}s", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        sig = {RED: ("MAIN: RED", (0, 0, 255)), GREEN: ("MAIN: GREEN", (0, 220, 0))}.get(ph, ("MAIN: ?", (180, 180, 180)))
        cv2.putText(img, sig[0], (130, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, sig[1], 2, cv2.LINE_AA)
        x = 290
        for s, e, lab in events:
            if s <= t <= e:
                col = EVENT_COLORS.get(lab, (255, 255, 255))
                (w, _), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img, (x - 4, 3), (x + w + 4, 23), col, -1)
                cv2.putText(img, lab, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                x += w + 14
        # risk sparkline (last 10 s)
        if len(risk_t):
            m = (risk_t > t - 10) & (risk_t <= t)
            if m.sum() > 1:
                x0, y0, w, h = OUT_W - 210, out_h - 60, 200, 50
                cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (20, 20, 20), -1)
                cv2.line(img, (x0, y0 + h // 2), (x0 + w, y0 + h // 2), (90, 90, 90), 1)
                pts = np.stack([x0 + (risk_t[m] - (t - 10)) / 10 * w, y0 + h - risk_v[m] * h], 1).astype(np.int32)
                cv2.polylines(img, [pts], False, (0, 170, 255), 1, cv2.LINE_AA)
                cv2.putText(img, f"risk {risk_v[m][-1]:.2f}", (x0 + 4, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (255, 255, 255), 1, cv2.LINE_AA)
        proc.stdin.write(img.tobytes())
        idx += 1
    cap.release()
    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--pred", help="predictions json (events + risk); default: run the pipeline")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from src.traffic.pipeline import analyse, events_from_analysis
    an = analyse(a.video)
    name = Path(a.video).name
    if a.pred:
        v = json.loads(Path(a.pred).read_text())["videos"][name]
        events, risk = v["events"], v.get("risk", [])
    else:
        events, risk = events_from_analysis(an)[0], []
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    render(a.video, a.out, events, risk, an)
    print("wrote", a.out)
