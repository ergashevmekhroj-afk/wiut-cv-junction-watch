"""Rules on pairs of road users: accident, near_miss.

Both start from the same pairwise signal: two road users closing on each other
fast (low time-to-collision). Then
  * accident  = their footprints touch while one is still moving fast, and
                afterwards the involved objects are stationary (they stopped
                because of the contact, not because of a queue);
  * near_miss = an evasive action (hard braking or a sharp swerve) while the
                time-to-collision with another road user is short, and no contact.
Thresholds are deliberately strict: predicting a class that never occurs adds a
zero-scoring class to Score A.
"""
from __future__ import annotations

import numpy as np

from ..tracks import id_switch_near, raw_speed
from .context import Context

TTC_NEAR = 1.0           # s
NM_MIN_SPEED = 60.0      # ref px/s before braking (~20 km/h)
NM_DROP = 0.35           # speed after / before
BRAKE_DECEL = 120.0      # ref px/s^2  (sudden stop from ~60 px/s in 0.5 s)
SWERVE_DEG_S = 60.0      # heading rate
CONTACT_PAD = 0.15       # footprint overlap tolerance (fraction of size)
ACC_DECEL = 90.0         # ref px/s^2: impact-like deceleration (a normal stop is ~20-40)


def footprint(k, i) -> np.ndarray:
    """Approximate ground footprint in reference px: centred on xy, sized from the box."""
    w, h = k.wh[i]
    fw, fh = w * 0.8, h * 0.5
    x, y = k.xy[i]
    return np.array([x - fw / 2, y - fh / 2, x + fw / 2, y + fh / 2])


def gap(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two axis-aligned rectangles (0 if they overlap)."""
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return float(np.hypot(dx, dy))


def ttc(p1, v1, p2, v2, r: float) -> float:
    """Time until two discs (radius sum r) moving at constant velocity touch; inf if never."""
    dp, dv = p2 - p1, v2 - v1
    a = dv @ dv
    b = 2 * dp @ dv
    c = dp @ dp - r * r
    if c <= 0:
        return 0.0
    if a < 1e-6:
        return np.inf
    disc = b * b - 4 * a * c
    if disc < 0:
        return np.inf
    t = (-b - np.sqrt(disc)) / (2 * a)
    return float(t) if t >= 0 else np.inf


def _accel(k) -> np.ndarray:
    """Signed longitudinal acceleration (negative = braking)."""
    sp = k.speed
    if len(sp) < 3:
        return np.zeros_like(sp)
    return np.gradient(sp, k.t)


def _pairs_at(ctx: Context, t: float, max_dist: float = 150.0):
    """(track, index) of every road user present at time t, and close pairs among them."""
    present = []
    for k in ctx.tracks:
        if k.t[0] <= t <= k.t[-1]:
            j = k.at(t)
            if abs(k.t[j] - t) < 0.2:
                present.append((k, j))
    pairs = []
    for a in range(len(present)):
        for b in range(a + 1, len(present)):
            (ka, ia), (kb, ib) = present[a], present[b]
            if not (ka.is_vehicle or kb.is_vehicle):
                continue
            if np.linalg.norm(ka.xy[ia] - kb.xy[ib]) < max_dist:
                pairs.append((ka, ia, kb, ib))
    return pairs


def _overlap_frac(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def _separated(ctx: Context, p: np.ndarray, q: np.ndarray) -> bool:
    """The segment p-q crosses the median/refuge: the two are on different carriageways."""
    pts = p[None] + np.linspace(0, 1, 9)[:, None] * (q - p)[None]
    return bool(ctx.scene.inside("median", pts).any() or ctx.scene.inside("refuge", pts).any())


def accident(ctx: Context, step: float = 0.2) -> list[tuple[float, float]]:
    """Contact between road users followed by an abrupt stop.

    * footprints overlap (>= 10% of the smaller one), not across the median;
    * the faster party was moving (>= 40 px/s over the previous second) and lost
      >= 70% of its speed within ~0.6 s of the contact (normal stops take 2-3 s);
    * afterwards both stay stationary for >= 3 s.
    End: all involved objects stop moving (start of the stationary phase is
    reached) or leave the frame; we report until either moves again (capped).
    """
    out = []
    seen = set()
    for t in np.arange(0, ctx.duration, step):
        for ka, ia, kb, ib in _pairs_at(ctx, t, 120.0):
            key = (ka.tid, kb.tid)
            if key in seen or not (ka.is_vehicle and kb.is_vehicle or ka.is_person or kb.is_person):
                continue
            if _overlap_frac(footprint(ka, ia), footprint(kb, ib)) < 0.10:
                continue
            if _separated(ctx, ka.xy[ia], kb.xy[ib]) or _occluded(ka, kb, t) or _occluded(kb, ka, t):
                continue
            if id_switch_near(ka, t) or id_switch_near(kb, t):
                continue
            va = _median_speed(ka, t - 0.6, t - 0.1)
            vb = _median_speed(kb, t - 0.6, t - 0.1)
            mover = ka if va >= vb else kb
            v0 = max(va, vb)
            v1 = _median_speed(mover, t + 0.2, t + 0.6)
            # abrupt: >= 65% of the speed lost and >= ACC_DECEL px/s^2 across the contact
            if v0 < 40.0 or v1 > 0.35 * v0 or (v0 - v1) / 0.75 < ACC_DECEL:
                continue
            ok, end = True, t
            for k in (ka, kb):
                if k.t[-1] < t + 4.0 or _median_speed(k, t + 1.5, t + 4.0) > 8.0:
                    ok = False
                    break
                moving = np.flatnonzero((k.t > t + 4.0) & (k.speed > 8.0))
                end = max(end, float(k.t[moving[0]]) if len(moving) else float(k.t[-1]))
            if not ok:
                continue
            seen.add(key)
            out.append((t, min(end, t + 60.0)))
            ctx.debug.setdefault("accident", []).append({"tids": key, "t": round(float(t), 2), "v0": round(v0)})
    return out


def _box_iou_min(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix / max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def _occluded(k, o, t: float) -> bool:
    """k's box is mostly covered by o's box around t (its 'braking' is a tracking artefact)."""
    return any(_box_iou_min(k.box[k.at(t + dt)], o.box[o.at(t + dt)]) > 0.4 for dt in (0.0, 0.5, 1.0))


def _median_speed(k, t0: float, t1: float) -> float:
    m = (k.t >= t0) & (k.t <= t1)
    return float(np.median(k.speed[m])) if m.any() else 0.0


def near_miss(ctx: Context) -> list[tuple[float, float]]:
    """Sharp braking of a vehicle toward a road user crossing its path, without contact.

    * braking: median speed over the previous second >= NM_MIN_SPEED and <= half of
      that within the next second;
    * conflict: just before braking, another road user is on a crossing course
      (heading differs >= 30 deg, or it is a pedestrian/cyclist), they are not
      already touching, and the time-to-collision is short;
    * no contact within 2 s (contact would be an accident).
    """
    from .motion_rules import _is_u_turn
    out = []
    for k in ctx.vehicles():
        if len(k.t) < 10 or _is_u_turn(ctx, k):
            continue                               # U-turners slow down mid-turn as a matter of course
        for i in range(len(k.t)):
            t = float(k.t[i])
            if t - k.t[0] < 1.0 or k.t[-1] - t < 1.0:
                continue
            if any(s - 1.0 <= t <= e + 1.0 for s, e in out):
                continue
            v0 = _median_speed(k, t - 1.0, t - 0.1)
            if v0 < NM_MIN_SPEED:
                continue
            v1 = _median_speed(k, t + 0.3, t + 1.0)
            if v1 > NM_DROP * v0:
                continue
            win = k.t[(k.t >= t - 1.0) & (k.t <= t + 1.0)]
            if len(win) < 2 or np.diff(win).max() > 0.35 or id_switch_near(k, t):
                continue                           # detection gap / ID switch, not braking
            if raw_speed(k, t + 0.3, t + 1.0) > 0.5 * raw_speed(k, t - 1.0, t - 0.1):
                continue                           # braking not visible in the raw boxes
            j0 = k.at(t - 0.4)
            best, partner = np.inf, None
            for ka, ia, kb, ib in _pairs_at(ctx, float(k.t[j0]), 150.0):
                if k not in (ka, kb):
                    continue
                o, io = (kb, ib) if ka is k else (ka, ia)
                if gap(footprint(k, j0), footprint(o, io)) == 0 or _occluded(k, o, t):
                    continue                       # touching / hidden behind the other: queue or occlusion
                crossing = not o.is_vehicle
                if o.speed[io] > 10 and k.speed[j0] > 10:
                    cosang = (o.v[io] @ k.v[j0]) / (o.speed[io] * k.speed[j0])
                    crossing = crossing or cosang < np.cos(np.radians(30))
                if not crossing:
                    continue
                r = 0.35 * (np.linalg.norm(k.wh[j0]) + np.linalg.norm(o.wh[io]))
                tc = ttc(k.xy[j0], k.v[j0], o.xy[io], o.v[io], r)
                if 0.0 < tc < best:
                    best, partner = tc, o
            if best > TTC_NEAR or partner is None:
                continue
            contact = any(gap(footprint(k, k.at(t + dt)), footprint(partner, partner.at(t + dt))) == 0
                          for dt in np.arange(0, 2.0, 0.2))
            if contact:
                continue
            e = t + 3.0
            for dt in np.arange(0.2, 4.0, 0.2):
                if np.linalg.norm(k.xy[k.at(t + dt)] - partner.xy[partner.at(t + dt)]) > 80.0:
                    e = t + dt
                    break
            s = max(0.0, t - 0.5)
            out.append((s, e))
            ctx.debug.setdefault("near_miss", []).append({"tid": k.tid, "other": partner.tid, "t": round(s, 2),
                                                          "ttc": round(best, 2), "v0": round(v0), "v1": round(v1)})
    return out
