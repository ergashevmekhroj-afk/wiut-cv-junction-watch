"""Rules involving pedestrians: jaywalking, failure_to_yield."""
from __future__ import annotations

import numpy as np

from ..scene import polygon_depth
from ..segments import runs
from .context import Context

CROSSINGS = ("cross_main", "cross_upper", "cross_side")
JAY_MIN_SEC = 1.0              # people cut across the junction gap in 1-2 s
JAY_CROSSING_CLEARANCE = 30.0   # ref px away from any crossing's stripes
JAY_KERB_DEPTH = 18.0           # ref px inside the carriageway (not standing at the kerb)
RIDER_SPEED = 80.0              # ref px/s median: a 'person' this fast is riding (walkers near the camera reach ~55)
SAFE_ZONES = ("median", "refuge", "island_a", "island_b", "island_c")
YIELD_RADIUS = 90.0      # ref px: pedestrian this close to the vehicle's path on the crossing
YIELD_MIN_SPEED = 20.0    # driving through, not creeping in a queue


def _on_crossing(ctx: Context, xy: np.ndarray, pad: float = 12.0) -> np.ndarray:
    m = np.zeros(len(xy), bool)
    for c in CROSSINGS:
        m |= ctx.scene.inside(c, xy)
        if pad:
            # tolerate walking just beside the stripes
            for dx, dy in ((pad, 0), (-pad, 0), (0, pad), (0, -pad)):
                m |= ctx.scene.inside(c, xy + np.array([dx, dy]))
    return m


def jaywalking(ctx: Context) -> list[tuple[float, float]]:
    """Pedestrian clearly on the carriageway: deeper than the kerb, away from every crossing."""
    out = []
    P = ctx.scene.polygons
    for p in ctx.people():
        moving = p.speed[p.speed > 5.0]
        if p.rider_hits >= 1 or (len(moving) >= 5 and np.median(moving) > RIDER_SPEED):
            continue                               # rider in traffic, not a pedestrian (median: robust to box jitter)
        depth = polygon_depth(P["road"], p.xy)
        clear = np.min([-polygon_depth(P[c], p.xy) for c in CROSSINGS], axis=0)
        safe = np.max([polygon_depth(P[z], p.xy) for z in SAFE_ZONES], axis=0)
        on_road = (depth > JAY_KERB_DEPTH) & (clear > JAY_CROSSING_CLEARANCE) & (safe < -5.0)
        for s, e in runs(p.t, on_road, merge_gap=1.0, min_len=JAY_MIN_SEC):
            # a person inside a vehicle box (driver/passenger getting out) is not a pedestrian
            if _inside_vehicle(ctx, p, s, e):
                continue
            out.append((s, e))
            ctx.debug.setdefault("jaywalking", []).append({"tid": p.tid, "t": round(s, 2), "len": round(e - s, 1)})
    return out


def _inside_vehicle(ctx: Context, p, s: float, e: float) -> bool:
    j = p.at((s + e) / 2)
    bx = p.box[j]
    for v in ctx.vehicles():
        if v.t[0] <= p.t[j] <= v.t[-1]:
            vb = v.box[v.at(p.t[j])]
            ix = max(0, min(bx[2], vb[2]) - max(bx[0], vb[0])) * max(0, min(bx[3], vb[3]) - max(bx[1], vb[1]))
            if ix > 0.6 * (bx[2] - bx[0]) * (bx[3] - bx[1]):
                return True
    return False


def failure_to_yield(ctx: Context) -> list[tuple[float, float]]:
    """Vehicle drives through a crossing while a pedestrian is on it near its path."""
    out = []
    people = ctx.people()
    for k in ctx.vehicles():
        # front and rear of the vehicle along its heading: the event runs from the
        # front entering the crossing to the rear leaving it (annotation convention)
        u = k.v / np.maximum(k.speed[:, None], 1e-6)
        half = 0.4 * np.linalg.norm(k.wh, axis=1)[:, None]
        front, rear = k.xy + u * half, k.xy - u * half
        for c in CROSSINGS:
            occ = ctx.scene.inside(c, k.xy) | ctx.scene.inside(c, front) | ctx.scene.inside(c, rear)
            inside = occ & (k.speed > YIELD_MIN_SPEED)
            for s, e in runs(k.t, inside, merge_gap=0.5, min_len=0.3):
                hit = False
                for p in people:
                    if p.t[0] > e or p.t[-1] < s:
                        continue
                    sel = (p.t >= s - 0.5) & (p.t <= e)
                    if not sel.any():
                        continue
                    pxy = p.xy[sel]
                    on = ctx.scene.inside(c, pxy)
                    if not on.any():
                        continue
                    # closest approach between pedestrian and vehicle while both on the crossing
                    vt = k.t[(k.t >= s) & (k.t <= e)]
                    vxy = k.xy[(k.t >= s) & (k.t <= e)]
                    d = np.min(np.linalg.norm(vxy[:, None] - pxy[on][None], axis=2))
                    if d < YIELD_RADIUS:
                        hit = True
                        break
                if hit:
                    out.append((s, e))
                    ctx.debug.setdefault("failure_to_yield", []).append({"tid": k.tid, "t": round(s, 2), "crossing": c})
    return out
