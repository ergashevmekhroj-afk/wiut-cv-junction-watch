"""Collect per-sample results for the website -> site/data/results.json.

    python tools/site_data.py --analyses DIR_WITH_PICKLES --danger DIR_WITH_danger_*.npy --out site/data
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.traffic.pipeline import events_from_analysis
from src.traffic.risk import CausalRisk
from src.traffic.scene import Scene

ap = argparse.ArgumentParser()
ap.add_argument("--analyses", required=True); ap.add_argument("--danger", required=True)
ap.add_argument("--pred", help="official predictions_samples.json (events + risk) to use instead")
ap.add_argument("--out", required=True)
a = ap.parse_args()
pred = json.loads(Path(a.pred).read_text())["videos"] if a.pred else None
res = {}
cal = CausalRisk()
for pk in sorted(Path(a.analyses).glob("a0*.pkl")):
    v = f"sample_{pk.stem[1:]}.mp4"
    an = pickle.load(open(pk, "rb")); an["scene"] = Scene.load()
    ev, dbg = events_from_analysis(an)
    if pred and v in pred:
        ev = pred[v]["events"]
        risk = pred[v]["risk"][::6]
    else:
        d = np.load(Path(a.danger) / f"danger_{Path(v).stem}_live.npy")
        risk = [[round(float(t), 2), round(cal.calib(float(x)), 4)] for t, x in zip(d[0], d[1])]
    ph = an["phase"]; t = an["phase_t"]
    ch = np.flatnonzero(np.diff(ph)) + 1; b = [0, *ch, len(ph)]
    runs = [[int(ph[b[i]]), round(float(t[b[i]]), 2), round(float(t[min(b[i + 1], len(t) - 1)]), 2)] for i in range(len(b) - 1)]
    res[v] = {"duration": round(an["info"]["duration"], 2), "events": ev, "risk": risk, "signal": runs,
              "legal_u_turns": [[x["t"], x["end"]] for x in dbg.get("legal_u_turn", [])],
              "n_tracks": {"vehicles": sum(1 for k in an["tracks"] if k.is_vehicle and len(k.t) >= 10),
                           "people": sum(1 for k in an["tracks"] if k.is_person and len(k.t) >= 10)}}
    print(v, len(ev), "events")
Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out) / "results.json").write_text(json.dumps(res))
