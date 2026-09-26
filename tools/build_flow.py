"""Learn the legal driving direction per road cell from the sample videos' vehicle tracks.

    python tools/build_flow.py --videos samples      # -> scene/flow_prior.npz
Requires the detector cache (tools/cache_tracks.py) or runs the detector.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.traffic.flow import FlowPrior
from src.traffic.pipeline import analyse

ap = argparse.ArgumentParser(); ap.add_argument("--videos", required=True); a = ap.parse_args()
fp = FlowPrior.empty()
for v in sorted(Path(a.videos).glob("*.mp4")):
    res = analyse(str(v))
    for k in res["tracks"]:
        if k.is_vehicle:
            fp.add(k.xy, k.v)
    print(v.name, len(res["tracks"]), "tracks")
fp.save(); print("saved scene/flow_prior.npz")
