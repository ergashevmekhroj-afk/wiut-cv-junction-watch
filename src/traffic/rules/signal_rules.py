"""red_light and stop_line: rules that combine the signal phase with the stop line."""
from __future__ import annotations

import numpy as np

from ..signal import RED
from .context import Context

# Calibrated on the samples (EDA): after the pedestrian lamp turns red, legal
# clearing traffic keeps crossing for up to 3.9 s; the vehicle green starts up
# to ~1 s before the lamp turns green (drivers move off 0.3-1 s early).
AMBER_SEC = 4.5          # crossings this soon after the change are amber/clearance
EARLY_GREEN_SEC = 1.5    # crossings this close before the lamp's green are the vehicle green
MIN_CROSS_SPEED = 20.0   # ref px/s: must be driving across, not creeping on the line
STOP_SPEED = 6.0         # ref px/s: "stopped"
MAX_EVENT_SEC = 10.0


def _stopline_crossings(ctx: Context, track) -> list[int]:
    """Sample indices where the track crosses the main stop line in the driving direction."""
    line = ctx.scene.lines["stop_main"]
    sd = ctx.scene.side("stop_main", track.xy)
    x_ok = (track.xy[:, 0] >= line[:, 0].min() - 10) & (track.xy[:, 0] <= line[:, 0].max() + 10)
    return [i for i in np.flatnonzero((sd[:-1] < 0) & (sd[1:] >= 0)) if x_ok[i] or x_ok[i + 1]]


def _all_crossings(ctx: Context) -> list[tuple[float, int]]:
    out = []
    for k in ctx.vehicles():
        for i in _stopline_crossings(ctx, k):
            if k.speed[i:i + 2].max() >= MIN_CROSS_SPEED:
                out.append((float(k.t[i + 1]), k.tid))
    return out


def _queue_waiting(ctx: Context, t: float, exclude: int) -> bool:
    """Some other vehicle is stationary just behind the stop line at time t (evidence of red)."""
    for o in ctx.vehicles():
        if o.tid == exclude or not (o.t[0] <= t <= o.t[-1]):
            continue
        j = o.at(t)
        if abs(o.t[j] - t) > 0.5 or o.speed[j] > STOP_SPEED:
            continue
        sd = ctx.scene.side("stop_main", o.xy[j:j + 1])[0]
        if ctx.scene.inside("main_approach", o.xy[j:j + 1])[0] and sd < 0:
            return True
    return False


def red_light(ctx: Context) -> list[tuple[float, float]]:
    """A vehicle crosses the stop line on red while others wait.

    The lamp gives the phase, but the vehicle green can lead the pedestrian lamp
    by several seconds (sample_02), so a crossing only counts when it is not part
    of a queue discharge (< 2 other crossings within [-2, +4] s) and some other
    vehicle is still waiting behind the line.
    """
    out = []
    if len(ctx.phase) == 0:
        return out
    crossings = _all_crossings(ctx)
    for k in ctx.vehicles():
        for i in _stopline_crossings(ctx, k):
            t0 = float(k.t[i + 1])
            others = sum(1 for t, tid in crossings if tid != k.tid and t0 - 2.0 <= t <= t0 + 4.0)
            if others >= 2 or not _queue_waiting(ctx, t0, k.tid):
                continue
            if ctx.time_since_red(t0) < AMBER_SEC or k.speed[i:i + 2].max() < MIN_CROSS_SPEED:
                continue
            if ctx.next_green(t0) - t0 < EARLY_GREEN_SEC:
                continue
            after = np.arange(i + 1, len(k.t))
            # must drive on into the junction (stopping on the crossing is a stop_line violation)
            soon = after[k.t[after] <= t0 + 5.0]
            into = ctx.scene.inside("junction_box", k.xy[soon])
            if not into.any() or (k.speed[soon] < STOP_SPEED).any():
                continue
            # end: vehicle leaves the junction box / frame, capped
            inside = ctx.scene.inside("junction_box", k.xy[after]) | ctx.scene.inside("cross_main", k.xy[after])
            last = after[inside].max() if inside.any() else i + 1
            t1 = min(float(k.t[last]), t0 + MAX_EVENT_SEC)
            out.append((t0, max(t1, t0 + 1.0)))
            ctx.debug.setdefault("red_light", []).append({"tid": k.tid, "t": round(t0, 2)})
    return out


def stop_line(ctx: Context) -> list[tuple[float, float]]:
    """Vehicle stops past the stop line on red without entering the intersection.

    Start: the vehicle stops; end: the signal turns green (or the vehicle leaves).
    """
    out = []
    if len(ctx.phase) == 0:
        return out
    for k in ctx.vehicles():
        sd = ctx.scene.side("stop_main", k.xy)
        past = (sd > 0) & (ctx.scene.inside("cross_main", k.xy) | ctx.scene.inside("stop_zone", k.xy))
        stopped = k.speed < STOP_SPEED
        red = np.array([ctx.phase_at(t) == RED for t in k.t])
        m = past & stopped & red
        if m.sum() < 3:
            continue
        i0 = int(np.flatnonzero(m)[0])
        if float(k.t[m].max() - k.t[i0]) < 2.0:
            continue
        t0 = float(k.t[i0])
        t1 = min(ctx.next_green(t0), float(k.t[-1]))
        if t1 > t0 + 1.0:
            out.append((t0, t1))
            ctx.debug.setdefault("stop_line", []).append({"tid": k.tid, "t": round(t0, 2)})
    return out
