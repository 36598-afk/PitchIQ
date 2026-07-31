"""
path_scoring.py — Scores every detection in a pitch, decides which ones
form the real flight path, and identifies where that path ENDS (impact).

The core idea: a real pitch traces a smooth, continuous, one-directional
arc. False positives don't. So we list every detection first, then judge
each one against the shape the trajectory has established so far.

Rules applied (in rough order of how much they matter):

  1. DIRECTION CONSISTENCY (the strongest signal)
     The angle between consecutive steps shouldn't swing wildly. A ball
     doesn't zigzag. Big angle changes = almost certainly a different object.

  2. STEP SIZE SANITY
     Frame-to-frame movement should stay in a believable range relative to
     the rest of the flight. A step several times larger than the typical
     step means the "ball" teleported — that's a jump to another object.

  3. NO BIG JUMPS ACROSS THE FRAME
     Any single step covering a large fraction of the frame is rejected
     outright regardless of anything else.

  4. NO RISING LATE IN FLIGHT
     A pitch rises briefly (if at all) then falls. Once we're several
     detections in and past the apex, the ball moving UP again is
     physically wrong — flag it.

  5. EARLY DETECTIONS ARE GUILTY UNTIL PROVEN INNOCENT
     The first few points have no established shape to check against, so
     they're only kept if they're consistent with the points that follow.

The output is the accepted path (in order) plus every rejected detection
with the specific reason it was rejected — so nothing is hidden.

IMPORTANT: the impact estimate is simply the LAST POINT OF THE ACCEPTED
PATH. It is not extrapolated, averaged, or computed any other way — it is
literally the final surviving detection of the validated flight.
"""

import numpy as np
import math


def _dist(a, b):
    return math.hypot(b[1] - a[1], b[2] - a[2])


def _angle(a, b):
    """Heading in degrees of the step from a to b."""
    return math.degrees(math.atan2(b[2] - a[2], b[1] - a[1]))


def _angle_diff(a1, a2):
    """Smallest absolute difference between two headings, 0-180."""
    d = abs(a1 - a2) % 360
    return d if d <= 180 else 360 - d


def _fit_rmse(run):
    """How well the run fits a smooth curve (quadratic in both x and y)."""
    if len(run) < 4:
        return float("inf")
    f = [d[0] for d in run]
    xs = [d[1] for d in run]
    ys = [d[2] for d in run]
    deg = 2 if len(run) >= 5 else 1
    try:
        px = np.polyfit(f, xs, deg)
        py = np.polyfit(f, ys, deg)
    except Exception:
        return float("inf")
    ex = np.array(xs) - np.polyval(px, f)
    ey = np.array(ys) - np.polyval(py, f)
    return float(np.sqrt(np.mean(ex ** 2 + ey ** 2)))


def analyze_pitch(dets, frame_w, frame_h,
                  max_gap=5,
                  max_angle_change=70.0,
                  step_ratio_limit=3.5,
                  jump_fraction=0.35,
                  apex_grace=10,
                  min_avg_motion=3.0):
    """
    dets: list of (frame, x, y, conf), any order — will be sorted.
    Returns a dict:
        path      : accepted detections in order (the real flight)
        rejected  : list of (detection, reason)
        impact    : the LAST point of path, or None
        rows      : per-detection report rows (every detection, with status)
        summary   : dict of aggregate info
    """
    dets = sorted(dets, key=lambda d: d[0])
    if not dets:
        return {"path": [], "rejected": [], "impact": None, "rows": [],
                "summary": {"n_detections": 0, "n_path": 0, "n_rejected": 0,
                            "rmse": None, "note": "no detections"}}

    max_jump = jump_fraction * max(frame_w, frame_h)

    # ── Stage 1: split into contiguous runs (a gap means the ball vanished,
    #    which usually separates real flight from stray detections) ─────────
    runs = []
    cur = [dets[0]]
    for d in dets[1:]:
        if d[0] - cur[-1][0] <= max_gap:
            cur.append(d)
        else:
            runs.append(cur)
            cur = [d]
    runs.append(cur)

    # ── Stage 2: pick the run that best looks like real flight ─────────────
    def avg_motion(run):
        if len(run) < 2:
            return 0.0
        return sum(_dist(run[i - 1], run[i]) for i in range(1, len(run))) / (len(run) - 1)

    moving = [r for r in runs if len(r) < 2 or avg_motion(r) >= min_avg_motion]
    candidates = moving if moving else runs

    scored = [(r, _fit_rmse(r)) for r in candidates]
    usable = [(r, rm) for r, rm in scored if rm != float("inf")]
    if usable:
        # prefer the LONGEST run that still fits a believable curve — a short
        # fragment always fits better trivially, so length has to dominate
        ceiling = 120.0
        ok = [(r, rm) for r, rm in usable if rm <= ceiling]
        base_run, base_rmse = (max(ok, key=lambda t: len(t[0]))
                               if ok else min(usable, key=lambda t: t[1]))
    else:
        base_run, base_rmse = max(candidates, key=len), None

    base_frames = set(d[0] for d in base_run)

    # ── Stage 3: walk the chosen run point by point and validate the SHAPE ─
    path = []
    rejected = []
    reasons = {}

    prev_heading = None
    steps_so_far = []

    for i, d in enumerate(base_run):
        if not path:
            path.append(d)
            continue

        last = path[-1]
        step = _dist(last, d)
        heading = _angle(last, d)

        reason = None

        # Rule 3: massive jump across the frame
        if step > max_jump:
            reason = f"jump of {step:.0f}px (> {max_jump:.0f}px limit)"

        # Rule 2: step wildly out of scale with the rest of the flight
        if reason is None and len(steps_so_far) >= 3:
            med = float(np.median(steps_so_far))
            if med > 1 and step > med * step_ratio_limit:
                reason = f"step {step:.0f}px vs typical {med:.0f}px"

        # Rule 1: direction changed too sharply
        if reason is None and prev_heading is not None:
            turn = _angle_diff(prev_heading, heading)
            if turn > max_angle_change:
                reason = f"direction change of {turn:.0f} deg"

        # Rule 4: rising again late in the flight
        if reason is None and len(path) >= apex_grace:
            dy = d[2] - last[2]
            if dy < -8:  # y decreasing = moving up on screen
                reason = f"moving upward ({dy}px) after {len(path)} points"

        if reason:
            rejected.append(d)
            reasons[d[0]] = reason
            # don't update heading/steps from a rejected point
            continue

        path.append(d)
        steps_so_far.append(step)
        prev_heading = heading

    # ── Stage 4: re-check the earliest points against the established shape ─
    # (they were accepted before any shape existed, so verify them now)
    if len(path) >= 8:
        core = path[3:]
        f = [d[0] for d in core]
        xs = [d[1] for d in core]
        ys = [d[2] for d in core]
        try:
            px = np.polyfit(f, xs, 2)
            py = np.polyfit(f, ys, 2)
            tol = max(130.0, 5.0 * _fit_rmse(core))
            keep_head = []
            for d in path[:3]:
                ex = abs(d[1] - np.polyval(px, d[0]))
                ey = abs(d[2] - np.polyval(py, d[0]))
                if math.hypot(ex, ey) > tol:
                    rejected.append(d)
                    reasons[d[0]] = (f"early point off the established curve "
                                     f"by {math.hypot(ex, ey):.0f}px")
                else:
                    keep_head.append(d)
            path = keep_head + core
        except Exception:
            pass

    # anything outside the chosen run was never in contention
    for d in dets:
        if d[0] not in base_frames:
            rejected.append(d)
            reasons.setdefault(d[0], "outside the main flight segment")

    path.sort(key=lambda d: d[0])
    rejected.sort(key=lambda d: d[0])

    impact = path[-1] if path else None

    # ── per-detection report ───────────────────────────────────────────────
    path_frames = set(d[0] for d in path)
    rows = []
    prev_pt = None
    for d in dets:
        in_path = d[0] in path_frames
        if in_path and prev_pt is not None:
            step = round(_dist(prev_pt, d), 1)
            heading = round(_angle(prev_pt, d), 1)
        else:
            step, heading = "", ""
        rows.append({
            "frame": d[0], "x": d[1], "y": d[2], "conf": round(d[3], 3),
            "step_px": step, "heading_deg": heading,
            "status": "PATH" if in_path else "rejected",
            "is_impact": "IMPACT" if (impact and d[0] == impact[0]) else "",
            "reason": "" if in_path else reasons.get(d[0], ""),
        })
        if in_path:
            prev_pt = d

    summary = {
        "n_detections": len(dets),
        "n_path": len(path),
        "n_rejected": len(rejected),
        "rmse": round(_fit_rmse(path), 1) if len(path) >= 4 else None,
        "start_frame": path[0][0] if path else None,
        "impact_frame": impact[0] if impact else None,
        "impact_x": impact[1] if impact else None,
        "impact_y": impact[2] if impact else None,
    }

    return {"path": path, "rejected": rejected, "impact": impact,
            "rows": rows, "summary": summary}
