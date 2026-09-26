"""Rules on single-vehicle motion: stopped_vehicle, wrong_way, illegal_u_turn, congestion."""
from __future__ import annotations

import numpy as np

from ..signal import RED
from ..scene import polygon_depth
from ..segments import runs
from ..tracks import predecessor
from .context import Context

STOP_SPEED = 6.0         # ref px/s
STOPPED_MIN_SEC = 10.0   # task definition
JUNCTION_STOP_SEC = 20.0 # turning vehicles wait for gaps up to ~15 s
WRONG_COS = -0.5         # moving within 60 deg of the reverse of the legal direction
WRONG_MIN_SPEED = 25.0
WRONG_MIN_SEC = 2.5
WRONG_MIN_DIST = 80.0    # ref px travelled against the flow
FLOW_MIN_COHERENCE = 0.8
FLOW_MIN_COUNT = 30


SIGNAL_AREA = ("main_approach", "stop_zone", "cross_main", "junction_box")
CROSSINGS = ("cross_main", "cross_upper", "cross_side")


def _waiting(ctx: Context, k, i0: int, i1: int) -> bool:
    """The stop is explained by traffic control or other traffic (not a 'stopped vehicle').

    * at the signal: inside the approach / stop area / junction while the main phase is red;
    * in a queue: another stationary vehicle close ahead in the local flow direction;
    * at a crossing: pedestrians on the crossing right in front of it.
    """
    sel = np.arange(i0, i1 + 1, max(1, (i1 - i0) // 20))
    for i in sel:
        t, p = float(k.t[i]), k.xy[i]
        if ctx.phase_at(t) == RED and any(ctx.scene.inside(r, p[None])[0] for r in SIGNAL_AREA):
            return True
    mid = (i0 + i1) // 2
    t, p = float(k.t[mid]), k.xy[mid]
    ahead = None
    if ctx.flow is not None:
        d, coh, _ = ctx.flow.lookup(p[None])
        ahead = d[0] if coh[0] > 0.5 else None
    for o in ctx.tracks:
        if o is k or not (o.t[0] <= t <= o.t[-1]):
            continue
        j = o.at(t)
        rel = o.xy[j] - p
        dist = np.linalg.norm(rel)
        if o.is_vehicle and dist < 80 and o.speed[j] < STOP_SPEED and (ahead is None or rel @ ahead > 0):
            return True
        if o.is_person and dist < 110 and any(ctx.scene.inside(c, o.xy[j:j + 1])[0] for c in CROSSINGS):
            return True
    return False


def stopped_vehicle(ctx: Context) -> list[tuple[float, float]]:
    out = []
    for k in ctx.vehicles():
        on_road = ctx.scene.inside("road", k.xy)
        still = (k.speed < STOP_SPEED) & on_road
        for s, e in runs(k.t, still, merge_gap=1.5):
            if e - s < STOPPED_MIN_SEC:
                continue
            i0, i1 = k.at(s), k.at(e)
            if _waiting(ctx, k, i0, i1):
                continue
            xy = k.xy[i0:i1 + 1]
            P = ctx.scene.polygons
            near_crossing = max(float(polygon_depth(P[c], xy).max()) for c in CROSSINGS) > -70.0
            if near_crossing:
                continue                           # waiting to pass a crossing
            if ctx.scene.inside("junction_box", xy).mean() > 0.5 and e - s < JUNCTION_STOP_SEC:
                continue                           # waiting for a gap to turn
            out.append((s, e))
            ctx.debug.setdefault("stopped_vehicle", []).append({"tid": k.tid, "t": round(s, 2), "len": round(e - s, 1)})
    return out


def wrong_way(ctx: Context) -> list[tuple[float, float]]:
    if ctx.flow is None:
        return []
    out = []
    for k in ctx.vehicles():
        d, coh, cnt = ctx.flow.lookup(k.xy)
        sp = k.speed
        u = k.v / np.maximum(sp[:, None], 1e-6)
        cos = (u * d).sum(1)
        edge = k.edge if k.edge is not None else np.zeros(len(k.t), bool)
        # only on straight carriageway links: inside the junction, turning paths legitimately cross
        in_junction = ctx.scene.inside("junction_box", k.xy) | ctx.scene.inside("side_area", k.xy)
        bad = ~in_junction & (sp > WRONG_MIN_SPEED) & (coh > FLOW_MIN_COHERENCE) & (cnt > FLOW_MIN_COUNT) & (cos < WRONG_COS) & ~edge
        if bad.any() and _is_u_turn(ctx, k):
            continue                               # the loop of a U-turn briefly points against the flow
        for s, e in runs(k.t, bad, merge_gap=1.0, min_len=WRONG_MIN_SEC):
            seg = k.xy[(k.t >= s) & (k.t <= e)]
            if len(seg) < 2 or np.linalg.norm(seg[-1] - seg[0]) < WRONG_MIN_DIST:
                continue
            out.append((s, e))
            ctx.debug.setdefault("wrong_way", []).append({"tid": k.tid, "t": round(s, 2), "len": round(e - s, 1)})
    return out


def headings(k, window: float = 1.0, min_disp: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Headings from displacement over `window` seconds (robust to jitter while stopped)."""
    ts, hs = [], []
    j = 0
    edge = k.edge if k.edge is not None else np.zeros(len(k.t), bool)
    for i in range(len(k.t)):
        if edge[i]:
            continue
        while j < len(k.t) and k.t[j] - k.t[i] < window:
            j += 1
        if j >= len(k.t):
            break
        d = k.xy[j] - k.xy[i]
        if np.linalg.norm(d) >= min_disp and not edge[j]:
            ts.append(k.t[i]); hs.append(np.arctan2(d[1], d[0]))
    return np.asarray(ts), np.unwrap(np.asarray(hs)) if hs else np.zeros(0)


def _is_u_turn(ctx: Context, k) -> bool:
    ts, h = headings(k)
    return len(h) >= 6 and abs(np.median(h[-3:]) - np.median(h[:3])) > np.radians(120)


def _illegal_where(ctx: Context, k, ts: np.ndarray, h: np.ndarray, h0: float) -> str | None:
    """Why this U-turn breaks the Uzbek traffic rules, or None if it is legal.

    Rules of the Road of Uzbekistan, clause 62: a U-turn is prohibited on
    pedestrian crossings (also tunnels, bridges, level crossings, poor
    visibility). At an intersection a U-turn from the left-most lane is
    otherwise allowed unless a sign/marking forbids it; the only sign at this
    junction's median nose is "keep right" (4.2.1), which does not. So we flag:
      * the reversal (heading ~90 deg from the start) happens with the vehicle
        on a pedestrian crossing (front, centre or rear inside it);
      * the path cuts across the solid median before the median ends.
    """
    i = int(np.argmax(np.abs(h - h0) > np.radians(90)))
    j = k.at(ts[i])
    u = k.v[j] / max(np.linalg.norm(k.v[j]), 1e-6)
    half = 0.4 * np.linalg.norm(k.wh[j])
    pts = np.stack([k.xy[j], k.xy[j] + u * half, k.xy[j] - u * half])
    for c in CROSSINGS:
        if ctx.scene.inside(c, pts).any():
            return f"on {c}"
    if ctx.scene.inside("median", k.xy).sum() >= 3:
        return "across the solid median"
    return None


def illegal_u_turn(ctx: Context) -> list[tuple[float, float]]:
    """A U-turn (heading reverses > 150 deg) made where Uzbek rules prohibit it.

    Legal U-turns in the junction (the common manoeuvre around the median nose
    in the samples) are recorded in ctx.debug["legal_u_turn"] for the EDA but
    not reported as events.
    """
    out = []
    for k in ctx.vehicles():
        ts, h = headings(k)
        if len(h) < 6:
            continue
        h0, h1 = np.median(h[:3]), np.median(h[-3:])
        turn = h1 - h0
        if abs(turn) < np.radians(150):
            continue
        # path must actually travel (not a jittery parked car) and turn inside the junction
        path = np.linalg.norm(np.diff(k.xy, axis=0), axis=1).sum()
        if path < 250:
            continue
        # must START on the main carriageway (side-road left turns look like U-turns in the image)
        start_xy = k.xy[k.at(ts[1])][None]
        main_dir = np.asarray(ctx.scene.meta["main_dir"])
        if ctx.scene.inside("side_area", start_xy)[0] or np.cos(h0 - np.arctan2(main_dir[1], main_dir[0])) < 0.8:
            continue
        mid = k.xy[k.at(ts[np.argmax(np.abs(h - h0) > abs(turn) / 2)])]
        if not (ctx.scene.inside("junction_box", mid[None])[0] or ctx.scene.inside("cross_upper", mid[None])[0]
                or ctx.scene.inside("refuge", mid[None])[0]):
            continue
        i_done = int(np.argmax(np.abs(h - h0) > abs(turn) - np.radians(30)))
        # a real U-turn drives away afterwards (not a hook at the end of a lost track)
        after = k.xy[k.t >= ts[i_done]]
        upper = np.asarray(ctx.scene.meta["upper_dir"], float)
        if len(after) < 2:
            continue
        disp = after[-1] - after[0]
        if np.linalg.norm(disp) < 100 or disp @ upper / np.linalg.norm(disp) / np.linalg.norm(upper) < 0.7:
            continue                               # must drive away up the upper carriageway
        s = float(ts[np.argmax(np.abs(h - h0) > np.radians(30))])
        e = float(ts[i_done]) + 1.0
        # the turn may have started on an earlier fragment of the same vehicle
        prev = predecessor(ctx.tracks, k)
        if prev is not None and np.linalg.norm(prev.xy[-1] - k.xy[0]) < 60:
            ps, ph = headings(prev)
            if len(ph) >= 3:
                dev = np.abs(ph - np.median(ph[:3]))
                if (dev > np.radians(30)).any():
                    s = float(ps[np.argmax(dev > np.radians(30))])
        if e <= s:
            continue
        why = _illegal_where(ctx, k, ts, h, h0)
        if why is None:
            ctx.debug.setdefault("legal_u_turn", []).append({"tid": k.tid, "t": round(s, 2), "end": round(e, 2)})
            continue
        out.append((s, e))
        ctx.debug.setdefault("illegal_u_turn", []).append({"tid": k.tid, "t": round(s, 2), "why": why,
                                                           "turn_deg": round(float(np.degrees(turn)))})
    return out


def congestion(ctx: Context, region: str = "main_approach", min_vehicles: int = 8,
               crawl: float = 12.0, min_sec: float = 20.0, step: float = 1.0) -> list[tuple[float, float]]:
    """Traffic at a standstill / crawling across the approach, NOT explained by a red light.

    Per second: vehicles inside `region`; congested if there are many and their
    median speed is a crawl. Runs are kept only if they persist through a green
    phase (a normal red-light queue clears within one green).
    """
    ts = np.arange(0, ctx.duration, step)
    flag = np.zeros(len(ts), bool)
    for n, t in enumerate(ts):
        speeds = []
        for k in ctx.vehicles():
            if k.t[0] <= t <= k.t[-1]:
                j = k.at(t)
                if abs(k.t[j] - t) < 0.5 and ctx.scene.inside(region, k.xy[j:j + 1])[0]:
                    speeds.append(k.speed[j])
        flag[n] = len(speeds) >= min_vehicles and np.median(speeds) < crawl
    out = []
    for s, e in runs(ts, flag, merge_gap=5.0, min_len=min_sec):
        green = sum(1 for g in ctx.green_onsets() if s + 5 < g < e - 10)
        if green or len(ctx.phase) == 0:
            out.append((s, e))
    return out
