"""Run detection + tracking on a folder of videos and cache the observation tables.

    python tools/cache_tracks.py --videos samples
"""
import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.traffic.perception import PerceptionConfig, load_model, run_perception

ap = argparse.ArgumentParser()
ap.add_argument("--videos", required=True)
ap.add_argument("--order", default="size", choices=["size", "name"])
a = ap.parse_args()
vids = sorted(Path(a.videos).glob("*.mp4"), key=(lambda p: p.stat().st_size) if a.order == "size" else None)
cfg = PerceptionConfig(); model = load_model(cfg)
for v in vids:
    t = time.time()
    obs, info = run_perception(str(v), cfg, model, progress=lambda p: print(f"  {v.name} {p:5.1%}", flush=True))
    print(f"{v.name}: {len(obs)} observations, {time.time()-t:.0f}s", flush=True)
