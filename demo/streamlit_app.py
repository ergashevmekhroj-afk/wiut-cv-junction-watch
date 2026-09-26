"""Live demo (Streamlit Community Cloud): upload a clip from the junction camera, get the events back.

Runs the same Part A pipeline as solution.py, on CPU: detector at 640 px and 5 fps
instead of 960 px and 10 fps, so a 2-minute clip finishes in a few minutes. The
risk curve is the Part B danger function replayed causally over the tracks.

    streamlit run demo/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TRAFFIC_CACHE", "off")

MAX_SEC, MAX_MB = 125.0, 200

st.set_page_config(page_title="Junction Watch · live demo", page_icon="🚦", layout="wide")


@st.cache_resource
def model():
    from src.traffic.perception import PerceptionConfig, load_model
    return load_model(PerceptionConfig())


def risk_curve(an: dict, fps: float = 5.0) -> list[list[float]]:
    from src.traffic.risk import CausalRisk
    r, out, hist = CausalRisk(), [], []
    for t in np.arange(0, an["info"]["duration"], 1 / fps):
        r.hist, seen = {}, set()
        for k in an["tracks"]:
            if k.t[0] <= t <= k.t[-1]:
                m = (k.t <= t) & (k.t > t - 1.6)
                if m.sum() >= 4:
                    r.hist[k.tid] = [(tt, p, k.cls, float(np.linalg.norm(w))) for tt, p, w in zip(k.t[m], k.xy[m], k.wh[m])]
                    seen.add(k.tid)
        hist.append(r.calib(r._danger(t, seen)))
        out.append([round(float(t), 2), round(float(np.mean(hist[-3:])), 4)])
    return out


def timeline_fig(events, risk, phase_t, phase, duration):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = sorted({e[2] for e in events})
    lanes = ["signal"] + labels
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 1.6 + 0.35 * len(lanes) + 1.6), sharex=True,
                                 gridspec_kw={"height_ratios": [len(lanes), 2]})
    if len(phase):
        ch = np.flatnonzero(np.diff(phase)) + 1
        b = [0, *ch, len(phase)]
        for i in range(len(b) - 1):
            p = int(phase[b[i]])
            if p >= 0:
                s, e = phase_t[b[i]], phase_t[min(b[i + 1], len(phase_t) - 1)]
                a1.barh(0, e - s, left=s, height=0.6, color="#D23B35" if p == 1 else "#2E8B57", alpha=0.8)
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    for e in events:
        y = lanes.index(e[2])
        a1.barh(y, e[1] - e[0], left=e[0], height=0.6, color=pal[(y - 1) % len(pal)])
    a1.set_yticks(range(len(lanes)), lanes, fontsize=9)
    a1.invert_yaxis()
    a1.set_xlim(0, duration)
    a1.grid(axis="x", color="#ddd", lw=0.6)
    if risk:
        rt, rv = zip(*risk)
        a2.plot(rt, rv, color="#eb6834", lw=1.6)
    a2.axhline(0.5, color="#888", ls="--", lw=1)
    a2.set_ylim(0, 1)
    a2.set_ylabel("risk")
    a2.set_xlabel("seconds")
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


st.title("Junction Watch · live demo")
st.markdown("Upload an **.mp4 from this camera, up to 2 minutes and 200 MB**. The same pipeline as our "
            "submission runs on this server's CPU (detector at 640 px, 5 fps). A 2-minute clip takes about "
            "3–5 minutes. WIUT Hackathon 2026.")

col1, col2 = st.columns([2, 1])
with col1:
    up = st.file_uploader("Your video (.mp4)", type=["mp4"])
with col2:
    use_example = st.button("Use the 30 s example from sample 04")

src = None
if up is not None:
    if up.size / 1e6 > MAX_MB:
        st.error(f"The file is {up.size / 1e6:.0f} MB; the demo accepts up to {MAX_MB} MB.")
    else:
        src = Path(tempfile.mkdtemp()) / "upload.mp4"
        src.write_bytes(up.getbuffer())
elif use_example:
    src = Path(__file__).with_name("demo_clip.mp4")

if src is not None:
    from src.traffic.perception import PerceptionConfig, video_info
    from src.traffic.pipeline import analyse, events_from_analysis
    import tools.render as R

    info = video_info(str(src))
    if info["n_frames"] <= 0 or info["fps"] <= 0:
        st.error("Could not read this video. Please upload an H.264 .mp4.")
        st.stop()
    if info["duration"] > MAX_SEC:
        st.error(f"The clip is {info['duration']:.0f} s long; the demo accepts up to 2 minutes. Trim it and try again.")
        st.stop()
    bar = st.progress(0.0, text="Aligning the video to the scene map")
    an = analyse(str(src), cfg=PerceptionConfig(imgsz=640, target_fps=5.0, device="cpu"), model=model(),
                 progress=lambda p: bar.progress(min(0.7, 0.05 + 0.65 * p), text="Detecting and tracking road users"))
    bar.progress(0.72, text="Applying the event rules")
    events, _ = events_from_analysis(an)
    bar.progress(0.76, text="Computing the risk curve")
    risk = risk_curve(an)
    bar.progress(0.8, text="Rendering the annotated video")
    out = Path(tempfile.mkdtemp()) / "annotated.mp4"
    R.OUT_W = 960
    R.render(str(src), str(out), events, risk, an)
    bar.progress(1.0, text="Done")

    n_v = sum(1 for k in an["tracks"] if k.is_vehicle and len(k.t) >= 5)
    n_p = sum(1 for k in an["tracks"] if k.is_person and len(k.t) >= 5)
    st.success(f"{len(events)} events in {info['duration']:.0f} s · {n_v} vehicles and {n_p} people tracked · "
               f"registration: {an['n_inliers']} matched features"
               + (" (low: is this the same camera?)" if an["n_inliers"] < 30 else ""))
    c1, c2 = st.columns([3, 2])
    with c1:
        st.video(str(out))
    with c2:
        st.dataframe([{"class": e[2], "start (s)": e[0], "end (s)": e[1], "length (s)": round(e[1] - e[0], 2)} for e in events],
                     use_container_width=True, hide_index=True)
        st.download_button("Download events + risk (JSON)", json.dumps({"events": events, "risk": risk}, indent=1),
                           file_name="events.json", mime="application/json")
    st.pyplot(timeline_fig(events, risk, an["phase_t"], an["phase"], info["duration"]))
