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

  6. A MINIMUM OF 9 DETECTIONS IS REQUIRED
     A run shorter than that is never trusted as the real path, no matter
     how well it happens to fit a curve — a handful of points can look
     smooth by coincidence. If nothing reaches this bar, the result is "no
     path found," not a guess.

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


def _rate(a, b):
    """Distance per ELAPSED FRAME between two detections.

    Whenever the detector misses a frame or two (occlusion, motion blur,
    the ball crossing a busy background), the next real detection is
    further away simply because more time passed. Comparing that raw
    distance against a typical SINGLE-frame step wrongly flags it as a
    teleport. Worse, once such a point is rejected, every later point is
    still measured from that same stale position, so the distances keep
    growing and the whole remainder of a real flight gets discarded —
    confirmed as the single largest cause of impact being called early.
    """
    frame_delta = max(1, b[0] - a[0])
    return _dist(a, b) / frame_delta


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
                  step_ratio_limit=3.0,
                  jump_fraction=0.35,
                  apex_grace=10,
                  min_avg_motion=10.0,
                  min_path_length=9):
    """
    dets: list of (frame, x, y, conf), any order — will be sorted.
    min_path_length: a run must have at least this many detections to even
        be considered as the real flight path. Anything shorter is treated
        as too little evidence to trust, no matter how well it fits a
        curve — better to report no path found than confidently report a
        wrong answer from a handful of points.
    Returns a dict:
        path      : accepted detections in order (the real flight)
        rejected  : list of (detection, reason)
        impact    : the LAST point of path, or None
        rows      : per-detection report rows (every detection, with status)
        summary   : dict of aggregate info
    """
    dets = sorted(dets, key=lambda d: d[0])

    # Deduplicate by frame, keeping the highest-confidence detection.
    # All internal bookkeeping (reasons, path membership, report rows) is
    # keyed by FRAME NUMBER, so two detections sharing a frame silently
    # corrupt it: one can vanish from both path and rejected (breaking the
    # accounting invariant), and both can end up flagged IMPACT. The real
    # detector emits a single best detection per frame, so this normally
    # never arises — but merged CSVs, re-runs, or any future multi-detection
    # mode would trigger it, and losing a detection without saying so is
    # exactly the kind of silent corruption that surfaces much later as an
    # inexplicable wrong answer. Keeping the highest-confidence one matches
    # what the detector itself does when it picks a box.
    if len(set(d[0] for d in dets)) != len(dets):
        best_per_frame = {}
        for d in dets:
            if d[0] not in best_per_frame or d[3] > best_per_frame[d[0]][3]:
                best_per_frame[d[0]] = d
        dets = [best_per_frame[f] for f in sorted(best_per_frame)]

    if not dets:
        return {"path": [], "rejected": [], "impact": None, "rows": [],
                "summary": {"n_detections": 0, "n_path": 0, "n_rejected": 0,
                            "rmse": None, "note": "no detections"}}

    max_jump = jump_fraction * max(frame_w, frame_h)

    def avg_motion(run):
        if len(run) < 2:
            return 0.0
        return sum(_dist(run[i - 1], run[i]) for i in range(1, len(run))) / (len(run) - 1)

    # ── Stage 1: split into contiguous runs ─────────────────────────────────
    # A run breaks on any of four conditions:
    #   1. A frame-number gap (the ball vanished for a few frames).
    #   2. A huge absolute jump (way bigger than anything physically
    #      plausible, regardless of context).
    #   3. A jump much bigger than the RECENT steps within this run. This
    #      catches the case #2 misses: a real, smoothly-moving trajectory
    #      (steps of ~20-30px) sitting right next to a cluster of static
    #      false-positive detections. The jump between them (~200px) isn't
    #      big enough to trip the absolute check, but it's wildly bigger
    #      than the ~20-30px steps the real trajectory was just making —
    #      confirmed directly from a real case where this exact pattern
    #      caused a clean 21-point arc to be merged with ~25 unrelated
    #      static detections immediately after it, contaminating the curve
    #      fit badly enough that the whole real trajectory got discarded.
    #   4. A sharp REVERSAL in direction, even when the distance is small
    #      enough to pass both distance checks above. Distance alone misses
    #      this: a real 26-point arc reversed 164 degrees into an unrelated
    #      cluster sitting only ~70px away — small enough to look like a
    #      plausible single-frame step, but physically nothing like a real
    #      continuation. Without this check the two got glued into one
    #      45-point run, and the resulting bad curve fit let a WORSE, less
    #      real run elsewhere win the selection outright — the true flight
    #      never even reached Stage 2 as its own clean candidate.
    #
    # LOOKAHEAD SKIP: when a point fails these checks, that alone doesn't
    # end the run anymore. First check whether the NEXT point reconnects
    # cleanly with the established trend — if so, the failing point was a
    # single bad detection sitting in the middle of good ones, not the end
    # of the real trajectory, and it gets skipped rather than treated as a
    # hard break. Without this, Stage 1 had zero tolerance for even one
    # interspersed bad frame: confirmed on a real case where a spurious
    # detection ALTERNATED with the real ball every other frame (100, 102,
    # 104... all sat at the same wrong position while 101, 103, 105...
    # were the real, smoothly continuing ball) — every single real point
    # got fragmented into an isolated singleton because each one, in
    # strict frame order, looked like a "failure" following the previous
    # bad one. Also confirmed on a real curving trajectory where the ball's
    # actual physical break curved harder than expected right before the
    # catch: one intermediate point read as an anomaly, but the point after
    # it clearly continued the real, curving arc into the glove.
    def passes_checks(prev, cand, steps_hist, headings_hist):
        if cand[0] - prev[0] > max_gap:
            return False
        # Compare RATES (px per elapsed frame), not raw distance — see _rate.
        rate = _rate(prev, cand)
        if rate > max_jump:
            return False
        if len(steps_hist) >= 3:
            med = float(np.median(steps_hist))
            # FIX: `med > 1` was meant to avoid dividing by a meaningless
            # near-zero median, but it has the opposite effect -- when a
            # run is genuinely near-motionless (median step ~1px, e.g. a
            # static glove/mound object sitting in frame), this bypasses
            # the ratio check ENTIRELY, since exactly-1 or sub-1 medians
            # never satisfy `med > 1`. That leaves only the flat max_jump
            # ceiling (hundreds of px) to catch anything, so a huge jump
            # into an unrelated real trajectory sails right through as a
            # "skippable outlier" and gets glued onto the static run.
            # Confirmed directly: a static object with median step 1.0
            # let a 382px/frame jump pass unchecked, contaminating an
            # entire real 30-point trajectory that followed. Flooring the
            # median at 1.0 instead of bypassing keeps the check active at
            # low speeds without changing behavior at typical/high speeds.
            effective_med = max(med, 1.0)
            if rate > effective_med * step_ratio_limit:
                return False
        if len(headings_hist) >= 2 and _dist(prev, cand) > 1:
            new_heading = _angle(prev, cand)
            recent_heading = float(np.median(headings_hist))
            if _angle_diff(recent_heading, new_heading) > max_angle_change:
                return False
        return True

    runs = []
    cur = [dets[0]]
    recent_steps: list[float] = []
    recent_headings: list[float] = []
    skipped_outliers = []
    i = 1
    while i < len(dets):
        d = dets[i]
        if passes_checks(cur[-1], d, recent_steps, recent_headings):
            spatial_jump = _dist(cur[-1], d)
            if spatial_jump > 1:
                recent_headings.append(_angle(cur[-1], d))
                if len(recent_headings) > 5:
                    recent_headings.pop(0)
            cur.append(d)
            recent_steps.append(_rate(cur[-2], d))
            if len(recent_steps) > 5:
                recent_steps.pop(0)
            i += 1
            continue

        # d failed. Before ending the run, check whether the NEXT point
        # reconnects cleanly — if so, d is a single skippable outlier, not
        # the end of the real trajectory. Requires at least 3 established
        # points so there's a real trend to check against, not just noise.
        can_skip = (len(cur) >= 3 and i + 1 < len(dets)
                    and passes_checks(cur[-1], dets[i + 1], recent_steps, recent_headings))
        if can_skip:
            skipped_outliers.append(d)
            i += 1
            continue

        runs.append(cur)
        cur = [d]
        recent_steps = []
        recent_headings = []
        i += 1
    runs.append(cur)

    reasons: dict = {}
    for d in skipped_outliers:
        reasons[d[0]] = "skipped as a single outlier between two consistent points"

    # ── Stage 1.5: bridge fragments that are really one trajectory ─────────
    # Stage 1 can correctly identify a genuine anomaly and split there — but
    # once split, the pieces on either side were previously discarded
    # forever, even when they're obviously the same real, continuous ball
    # flight with one bad frame in the middle. Confirmed directly from a
    # real case: a trajectory got cut into a 4-point piece and a 14-point
    # piece by a single outlier frame between them; only the shorter piece
    # (which happened to score better alone) survived, giving an impact
    # point far too early in the flight.
    #
    # This has to be done carefully — an early, looser version of this fix
    # caused a real regression by stitching together unrelated STATIC
    # fragments into a falsely-long, falsely-clean composite that beat the
    # genuinely correct trajectory. Two safeguards prevent that:
    #   1. Both runs being bridged must ALREADY be independently-moving,
    #      real candidates (never bridge static junk to static junk, or
    #      static junk to a real trajectory).
    #   2. The transition itself must be directionally plausible — the
    #      heading from run A's end into run B's start can't be a wild
    #      reversal from the heading run A was already moving in. A low
    #      combined curve-fit error alone isn't enough evidence; two
    #      unrelated short segments can coincidentally fit a curve together
    #      without actually being the same continuous motion.
    bridge_max_gap = 20
    bridge_rmse_limit = 30.0
    merged_again = True
    while merged_again:
        merged_again = False
        runs.sort(key=lambda r: r[0][0])
        for i in range(len(runs) - 1):
            a, b = runs[i], runs[i + 1]
            gap = b[0][0] - a[-1][0]
            if gap <= 0 or gap > bridge_max_gap:
                continue
            if len(a) < 2 or len(b) < 2 or len(a) + len(b) < 4:
                continue
            if avg_motion(a) < min_avg_motion or avg_motion(b) < min_avg_motion:
                continue

            # The anomaly that caused the original split often lands just
            # inside the edge of one fragment, not cleanly between them —
            # confirmed directly from a real case where the first point of
            # the later run was itself the bad detection that had caused
            # Stage 1 to split there in the first place. Trimming has to
            # run BEFORE the directional check, not after: checking
            # direction on the raw, untrimmed boundary rejects the bridge
            # using exactly the point trimming exists to discard, so a
            # bridge that would succeed perfectly after dropping one bad
            # edge point never even gets the chance. Confirmed directly —
            # a bridge with a 157-degree angle at the raw boundary was
            # actually a clean, correct continuation once the single bad
            # boundary point was trimmed away.
            #
            # Trimming has to EARN its keep: fewer points always fit a
            # curve at least as well, so picking purely on lowest rmse
            # would bias toward throwing away good data for a meaningless
            # fit improvement. A trimmed variant is only preferred if it
            # beats the untrimmed fit by a real margin.
            #
            # CRITICAL GUARD: a trimmed point must ALREADY look anomalous
            # within its OWN run before it's even eligible to be discarded.
            # Without this, the rmse-improvement check alone can justify
            # throwing away perfectly good real data — confirmed directly
            # on a real case: a flawless 15-point trajectory (consistent
            # ~37px/frame, arrow-straight heading) got bridged with 3
            # nearby junk points, and the trimmer decided that DROPPING the
            # trajectory's own final two real points and SPLICING IN two of
            # the junk points instead gave a marginally better combined
            # curve fit. That's backwards — trimming exists to discard a
            # genuine anomaly at a boundary, not to swap out good data for
            # a statistically prettier combination. A point consistent with
            # its own run's established rate and heading must never be a
            # trim candidate, no matter how the fit numbers look.
            def trim_is_justified(points_before, candidates, forward):
                """points_before: the established, kept part of the run.
                candidates: the points being considered for removal.
                forward: True if candidates come immediately after
                    points_before (trimming a's end); False if candidates
                    come immediately before it (trimming b's start)."""
                if len(points_before) < 3:
                    return True  # not enough history to judge, allow it
                steps = [_rate(points_before[i - 1], points_before[i])
                         for i in range(1, len(points_before))]
                if len(steps) < 2:
                    return True
                med = float(np.median(steps))
                if med <= 1:
                    return True
                seq = candidates if forward else list(reversed(candidates))
                anchor = points_before[-1] if forward else points_before[0]
                for pt in seq:
                    rate = _rate(anchor, pt) if forward else _rate(pt, anchor)
                    if rate <= med * 2.0:
                        return False  # consistent with the trend - real data
                    anchor = pt
                return True

            TRIM_IMPROVEMENT_FACTOR = 0.7
            untrimmed = a + b
            best_combined = untrimmed
            best_a_part, best_b_part = a, b
            best_rmse = _fit_rmse(untrimmed)
            for trim_a in range(0, max(0, len(a) - 2) + 1):
                for trim_b in range(0, max(0, len(b) - 2) + 1):
                    if trim_a > 2 or trim_b > 2:
                        continue
                    if trim_a == 0 and trim_b == 0:
                        continue
                    a_part = a[:len(a) - trim_a] if trim_a else a
                    b_part = b[trim_b:]
                    if len(a_part) + len(b_part) < 4 or len(a_part) < 2 or len(b_part) < 2:
                        continue
                    if trim_a and not trim_is_justified(a_part, a[len(a_part):], True):
                        continue
                    if trim_b and not trim_is_justified(b_part, b[:trim_b], False):
                        continue
                    candidate = a_part + b_part
                    rm = _fit_rmse(candidate)
                    if rm < best_rmse * TRIM_IMPROVEMENT_FACTOR:
                        best_rmse = rm
                        best_combined = candidate
                        best_a_part, best_b_part = a_part, b_part

            if best_rmse > bridge_rmse_limit:
                continue

            # Directional check on the EFFECTIVE boundary — whatever
            # survived trimming — not the original raw edge.
            if len(best_a_part) >= 2:
                a_heading = _angle(best_a_part[-2], best_a_part[-1])
                bridge_heading = _angle(best_a_part[-1], best_b_part[0])
                if _angle_diff(a_heading, bridge_heading) > max_angle_change:
                    continue

            runs[i] = best_combined
            del runs[i + 1]
            merged_again = True
            break

    # ── Stage 2: pick the run that best looks like real flight ─────────────
    # NOTE: there is deliberately NO fallback to non-moving runs here. An
    # earlier version did `candidates = moving if moving else runs`, which
    # meant that on a clip where nothing was actually moving, the code fell
    # straight back to the static detections it had just filtered out — and
    # happily reported them as a confident flight path. Verified with a
    # synthetic case: 40 detections at the exact same pixel were returned as
    # a 40-point "path" with an impact point. That is precisely the
    # static-object failure this whole module exists to prevent. If nothing
    # is moving, there is no pitch — say so.
    candidates = [r for r in runs if avg_motion(r) >= min_avg_motion]

    scored = [(r, _fit_rmse(r)) for r in candidates]
    usable = [(r, rm) for r, rm in scored if rm != float("inf")]

    # Require at least min_path_length points before a run is even eligible
    # to be considered the real path. A short run can sometimes score well
    # by the length/motion/fit formula below purely by chance, without
    # actually being enough evidence of a real, complete flight — better to
    # report no path found than confidently report a wrong answer built
    # from only a handful of points.
    long_enough = [(r, rm) for r, rm in usable if len(r) >= min_path_length]

    alt_warning = None
    if long_enough:
        # Prefer runs that are BOTH long AND well-fitting, using a single
        # continuous score rather than a hard rmse ceiling followed by pure
        # length comparison. The old ceiling let a long, badly-contaminated
        # run (lots of points, mediocre fit) beat a short, genuinely clean
        # run just by squeaking under the same loose cutoff and then winning
        # on point-count alone — confirmed directly from a real case where a
        # 55-point run with rmse 67 beat a 21-point run with rmse 5.6, even
        # though the 21-point run was the real, correct trajectory. Dividing
        # length by (1 + rmse) rewards length but only in proportion to how
        # well the run actually fits a curve, so a much cleaner shorter run
        # can win over a longer messier one.
        # Score rewards length and fit quality, but ALSO genuine motion —
        # not just length/(1+rmse). A short, barely-moving run can get a
        # deceptively perfect curve fit simply because it has almost no
        # real variance to fit against, letting it outscore a longer,
        # genuinely-moving trajectory that has some natural jitter (real
        # ball flight is never perfectly smooth). Confirmed directly from a
        # real case: a 9-point, barely-moving run (avg_motion 4.3, just
        # above the moving threshold) scored higher than the correct
        # 21-point trajectory (avg_motion 34) purely because its near-zero
        # variance gave it a trivially low rmse.
        #
        # rmse is normalized by the run's own avg_motion rather than used
        # as raw pixels. Absolute rmse naturally grows with a trajectory's
        # length and speed — a genuine 30-point fast arc accumulates more
        # real per-frame jitter than a trivial 10-point slow segment, even
        # when both are equally legitimate. Raw rmse in the denominator
        # therefore systematically favored shorter/slower runs over longer
        # genuine ones. Confirmed directly: a real 30-point arc (avg_motion
        # 33, rmse 37 - a normalized error of about 1.1 typical steps) lost
        # to a bridged 10-point run (avg_motion 16, rmse 4.2 - trivially
        # tight only because it barely moves) under raw-rmse scoring, and
        # correctly won once rmse was measured relative to typical step
        # size instead of absolute pixels.
        def score(run, rmse):
            # FIX (validated against reviewed_labels.csv, 475 pitches, full
            # disagreement audit): avg_motion is a raw MEAN of per-step
            # distances. Stage 1.5's bridging logic can occasionally glue a
            # stray point onto an otherwise-real run, injecting one huge
            # outlier step (confirmed on a real case: two steps of 737px
            # and 834px sitting in an otherwise ~20-30px/frame run). That
            # single outlier inflates the mean 3x+, which inflates the
            # score enough that a fake, unrelated cluster of detections can
            # outscore the real pitch and get selected as the flight path
            # -- even in cases where the code's own alt_warning below flags
            # exactly this run as suspicious.
            #
            # Fix: only exclude the single largest step from the average,
            # and only when it's a clear outlier relative to the rest of
            # the run's own steps (>5x their median) -- a real, un-bridged
            # trajectory's largest step is never that disproportionate to
            # its others, so this never fires on genuine runs. Grid-tested
            # ratio thresholds 3-6x against the full labeled set; 4-6x all
            # produce the same result and are the widest margin that still
            # fixes every confirmed bridging case, so 5x was chosen as the
            # middle of that stable range.
            steps = [_dist(run[i - 1], run[i]) for i in range(1, len(run))]
            tm = float(np.mean(steps)) if steps else 0.0
            if len(steps) >= 4:
                sorted_steps = sorted(steps)
                rest = sorted_steps[:-1]
                med_rest = float(np.median(rest)) if rest else 0.0
                if med_rest > 0.1 and sorted_steps[-1] > 5.0 * med_rest:
                    tm = float(np.mean(rest))
            normalized_rmse = rmse / tm if tm > 0.1 else rmse
            raw_score = len(run) * tm / (1 + normalized_rmse)
            # FIX (validated against reviewed_labels.csv, 475 pitches):
            # 473 of 475 human-confirmed real pitch paths in this dataset
            # move left-to-right across the frame (dx = end_x - start_x is
            # positive) -- almost certainly a consequence of a consistent
            # camera setup/pitcher handedness across the clips. The 2
            # exceptions are a genuine leftward pitch (Cy Pitch 1) and a
            # near-vertical throw (Pitch 263, dx~4, not really lateral
            # either way). Meanwhile the wrong candidates that were
            # winning ties against real pitches (return throws, other
            # incidental motion) consistently moved right-to-left.
            # A soft penalty -- not a hard filter -- lets a genuine
            # leftward or vertical pitch still win when it's the best or
            # only real candidate, while breaking ties correctly in favor
            # of the dominant rightward direction otherwise. Grid-tested
            # 0.1-1.0: the 0.3-0.8 range is a wide, stable optimum (130
            # total errors, down from 238, with only 2 boundary-frame
            # detections affected in exchange for fixing an entire
            # previously-wrong pitch); 0.5 was chosen as the middle of
            # that stable range.
            dx = run[-1][1] - run[0][1]
            if dx < 0:
                raw_score *= 0.5
            return raw_score

        base_run, base_rmse = max(long_enough, key=lambda t: score(t[0], t[1]))

        # ── Self-diagnostic: is there a longer real candidate we passed on? ─
        # Every "missed a perfect path" bug found today came from the SAME
        # underlying shape: a genuinely long, clean, fast-moving run existed
        # in the raw detections, but scoring picked something else. Rather
        # than only catching this the next time someone manually traces a
        # video frame by frame, flag it automatically: if a DIFFERENT
        # candidate is meaningfully longer than what got chosen, note it in
        # the summary so it's visible immediately, not discovered days
        # later from a frustrated bug report.
        alt_candidates = [(r, rm) for r, rm in long_enough if r is not base_run]
        longest_alt = max(alt_candidates, key=lambda t: len(t[0])) if alt_candidates else None
        alt_warning = None
        if longest_alt and len(longest_alt[0]) > len(base_run) * 1.3:
            alt_warning = (f"a {len(longest_alt[0])}-point candidate exists "
                           f"(frames {longest_alt[0][0][0]}-{longest_alt[0][-1][0]}) "
                           f"but a {len(base_run)}-point one was chosen — worth "
                           f"double-checking this result against the video")
    else:
        # Nothing reached the length bar — don't guess. Report no valid
        # path rather than picking whatever short fragment happens to exist.
        #
        # NOTE: `rejected` must be a list of BARE DETECTIONS here, exactly
        # matching the normal return path. An earlier version returned
        # (detection, reason) tuples instead, which crashed every consumer
        # that unpacks detections positionally — e.g.
        # visualize_scored_path.py's `{d[0]: (d[1], d[2]) for d in
        # result["rejected"]}` raised IndexError. The summary also carries
        # the same keys as the normal path so callers can read them
        # unconditionally.
        no_path_reason = (f"no run reached the minimum {min_path_length}-point "
                          f"length required")
        return {
            "path": [],
            "rejected": list(dets),
            "impact": None,
            "rows": [{"frame": d[0], "x": d[1], "y": d[2], "conf": round(d[3], 3),
                      "step_px": "", "heading_deg": "", "status": "rejected",
                      "is_impact": "", "reason": no_path_reason} for d in dets],
            "summary": {"n_detections": len(dets), "n_path": 0,
                        "n_rejected": len(dets), "rmse": None,
                        "start_frame": None, "impact_frame": None,
                        "impact_x": None, "impact_y": None,
                        "note": f"no run reached {min_path_length} points"},
        }

    base_frames = set(d[0] for d in base_run)

    # ── Stage 3: walk the run point by point and validate the SHAPE ────────
    # A single bad point right at the start (before any real heading/step
    # history exists to catch it) can poison every real point that comes
    # after it — confirmed directly from a real case: two wild, unrelated
    # points 653px apart got accepted first purely because there wasn't
    # enough history yet to flag them, and the bogus heading between them
    # then caused a genuinely clean, 24-point real trajectory right after
    # to be rejected wholesale for "direction change" against that bad
    # reference. The fix: don't commit to a single starting point. Try
    # walking from each of the first several candidates and keep whichever
    # walk survives the longest — a bad bootstrap pair will always lose to
    # a walk that started from the point right after it.
    def walk_from(start_idx):
        path, rejected, reasons = [], [], {}
        prev_heading = None
        steps_so_far = []
        for d in base_run[start_idx:]:
            if not path:
                path.append(d)
                continue

            last = path[-1]
            step = _dist(last, d)
            # Per-frame rate, so a legitimately larger displacement across
            # missing frames isn't mistaken for a teleport. Without this,
            # a 2-3 frame detector gap made the next real point look like
            # a huge jump, and since every later point was then still
            # measured from that same stale position, the distances kept
            # growing and the ENTIRE rest of a real flight was discarded.
            step_rate = _rate(last, d)
            heading = _angle(last, d)
            reason = None

            # Confidence-aware tolerance. A point whose confidence is much
            # lower than the path's own established level is often the mitt,
            # or the ball mid-occlusion as it's caught — not a clean ball
            # detection. Its POSITION can still look plausible enough to
            # slip past the normal thresholds, which are tuned for genuine
            # detections. Confirmed directly from real cases: a reported
            # impact point at confidence 0.221 against a flight that ran at
            # ~0.9 throughout, and a mid-flight point at 0.254 that corrupted
            # the reference heading and caused two genuinely good points
            # right after it to be wrongly rejected. Rather than an outright
            # veto (a real detection can legitimately dip in confidence from
            # motion blur), low confidence tightens how well the point has
            # to fit the trajectory to be trusted — a low-confidence point
            # that's ALSO a great geometric fit can still get through.
            local_conf_window = [p[3] for p in path[-5:]]
            local_conf = float(np.median(local_conf_window)) if local_conf_window else d[3]
            conf_ratio = d[3] / local_conf if local_conf > 0.05 else 1.0
            tolerance_scale = 1.0
            if conf_ratio < 0.5:
                tolerance_scale = max(0.25, conf_ratio)

            # Rule 3: massive jump across the frame (per elapsed frame)
            if step_rate > max_jump:
                reason = f"jump of {step:.0f}px (> {max_jump:.0f}px/frame limit)"

            # Rule 2: step wildly out of scale with the rest of the flight
            if reason is None and len(steps_so_far) >= 3:
                med = float(np.median(steps_so_far))
                # Floor of 1.5x ensures a genuinely typical step is never
                # rejected purely for low confidence — confirmed directly
                # from a real case where tolerance_scale pushed the
                # effective limit below 1.0x, rejecting a 56px step against
                # a 57px typical (an almost exact match) just because its
                # confidence happened to be low.
                effective_ratio_limit = max(1.5, step_ratio_limit * tolerance_scale)
                # FIX: same bypass bug as Stage 1's passes_checks -- `med > 1`
                # disabled this check entirely whenever the path's established
                # steps were near-static (median <=1px), letting an unrelated
                # fast jump slip into the accepted path right at that point.
                # Flooring at 1.0 keeps the check active without changing
                # behavior at normal speeds.
                effective_med = max(med, 1.0)
                if step_rate > effective_med * effective_ratio_limit:
                    reason = f"step {step_rate:.0f}px/frame vs typical {med:.0f}px/frame"
                    if tolerance_scale < 1.0:
                        reason += f" (tightened - confidence {d[3]:.2f} vs recent {local_conf:.2f})"

            # Rule 1: direction changed too sharply
            if reason is None and prev_heading is not None:
                turn = _angle_diff(prev_heading, heading)
                # Same floor logic: a genuinely small, plausible turn must
                # never be rejected on confidence alone.
                effective_angle_limit = max(25.0, max_angle_change * tolerance_scale)
                if turn > effective_angle_limit:
                    reason = f"direction change of {turn:.0f} deg"
                    if tolerance_scale < 1.0:
                        reason += f" (tightened - confidence {d[3]:.2f} vs recent {local_conf:.2f})"

            # Rule 4: reversing direction late in the flight. Derives the
            # pitch's OWN established vertical direction rather than
            # hardcoding "must move down the screen", so this still works
            # across camera angles where the ball legitimately rises on
            # screen. Threshold scales with typical step size rather than
            # a fixed pixel count.
            if reason is None and len(path) >= apex_grace and len(steps_so_far) >= 3:
                recent_dys = [path[k][2] - path[k - 1][2] for k in range(1, len(path))]
                dominant_dy = float(np.median(recent_dys))
                dy = d[2] - last[2]
                typical_step = float(np.median(steps_so_far))
                reversal_tol = max(8.0, 0.25 * typical_step)
                if abs(dominant_dy) > reversal_tol:
                    moving_against = (dy > reversal_tol) if dominant_dy < 0 else (dy < -reversal_tol)
                    if moving_against:
                        reason = (f"reversed vertical direction ({dy:+.0f}px vs "
                                  f"established {dominant_dy:+.0f}px/frame) after "
                                  f"{len(path)} points")

            if reason:
                rejected.append(d)
                reasons[d[0]] = reason
                continue

            path.append(d)
            steps_so_far.append(step_rate)
            prev_heading = heading

        # Anything before start_idx was skipped for this attempt — record
        # it as rejected so every detection is accounted for regardless of
        # which starting point ultimately wins.
        for d in base_run[:start_idx]:
            rejected.append(d)
            reasons[d[0]] = "skipped trying a later starting point instead"

        return path, rejected, reasons

    max_bootstrap_tries = min(5, len(base_run))
    best_path, best_rejected, best_reasons = walk_from(0)
    for start_idx in range(1, max_bootstrap_tries):
        candidate_path, candidate_rejected, candidate_reasons = walk_from(start_idx)
        if len(candidate_path) > len(best_path):
            best_path, best_rejected, best_reasons = candidate_path, candidate_rejected, candidate_reasons

    path, rejected = best_path, best_rejected
    # Merge, don't overwrite - the outer `reasons` already holds entries
    # from Stage 1's lookahead-skip (single outliers skipped while
    # building runs). Overwriting it here would silently discard those
    # diagnostic reasons, since walk_from builds its own separate dict.
    reasons.update(best_reasons)

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

    # ── Stage 4.5: re-check the LATEST points against the established shape ─
    # Same idea as Stage 4, mirrored to the end of the path. A point right
    # after a frame gap — the ball briefly not detected, often because it's
    # now disappearing into the glove — can slip past the raw angle/step
    # thresholds (its heading and step size aren't extreme enough to trip
    # Rule 1/2 on their own) while still clearly not continuing the real
    # curve. Confirmed directly from a real case: a point arrived after a
    # 5-frame gap with a step of 97px against a typical ~33px, at a lower
    # confidence than the rest of the path — kinematically borderline, but
    # nowhere near where the established curve actually predicted the ball
    # would be. Fitting the curve from everything BUT the last couple
    # points and checking whether they match that extrapolation catches
    # this in a way a single instantaneous angle/step check can't.
    #
    # The fit uses only a RECENT local window, not the entire remaining
    # path. Many real pitches genuinely curve MORE sharply in their final
    # approach (real movement/break), so fitting one quadratic across the
    # whole flight blends an earlier straighter section with the later,
    # more sharply curving one — diluting the recent curvature and making
    # the extrapolation undershoot where the ball is actually headed.
    # Confirmed directly on a real case: a pitch's heading shifted from
    # ~90 to ~115 degrees over its final third (real, visible curving
    # movement caught on video), and fitting the whole path rejected the
    # genuinely continuing next point as "off the extrapolation" — when it
    # was correctly continuing the recent curve. The true catch was several
    # points further along, clearly visible in the actual footage.
    extrap_window = 10
    while len(path) >= 8:
        core_all = path[:-1]
        core = core_all[-extrap_window:] if len(core_all) > extrap_window else core_all
        last_pt = path[-1]
        f = [d[0] for d in core]
        xs = [d[1] for d in core]
        ys = [d[2] for d in core]
        try:
            px = np.polyfit(f, xs, 2)
            py = np.polyfit(f, ys, 2)
            base_tol = max(50.0, 8.0 * _fit_rmse(core))

            # Confidence cross-check. A plain quadratic model is a poor fit
            # for a REAL pitch that's genuinely curving or decelerating
            # into its final approach — confirmed directly on a real case
            # where a smoothly curving pitch's actual next point deviated
            # ~94px from the quadratic extrapolation, roughly double the
            # base tolerance, purely because the ball's real physical
            # motion isn't a clean parabola near the catch. But that
            # point's confidence (0.78) was normal for the flight, nothing
            # like the 0.15-0.30 range seen in confirmed glove-transition
            # cases. Confidence close to the path's own level means trust
            # the detection over the model; only a genuinely low-confidence
            # deviation gets the strict check.
            local_conf = float(np.median([d[3] for d in core]))
            conf_ratio = last_pt[3] / local_conf if local_conf > 0.05 else 1.0
            # FIX (validated against reviewed_labels.csv AND a dedicated
            # impact-accuracy check across all 475 pitches, not just raw
            # detection count -- impact correctness matters more than any
            # single detection): the 0.6 threshold here meant only a
            # SEVERE confidence drop (>40%) triggered strict scrutiny.
            # But checking all 40 confirmed glove-drag impact points in
            # the labeled set showed a median confidence drop of only
            # ~15% (ratio ~0.85) -- real, but nowhere near 0.6. Those
            # cases were getting the lenient 3x tolerance and sliding
            # through uncaught. Raising the threshold to 0.92 was
            # grid-searched specifically against impact-call correctness
            # (not raw detection accuracy, which can be misleading here):
            # it fixes 17 pitches' impact calls and only mildly shifts 7
            # others (each by 1-3 frames, never wildly wrong), for a net
            # of 436/475 -> 446/475 pitches with a correct impact call.
            tol = base_tol if conf_ratio < 0.92 else base_tol * 3.0

            ex = abs(last_pt[1] - np.polyval(px, last_pt[0]))
            ey = abs(last_pt[2] - np.polyval(py, last_pt[0]))
            if math.hypot(ex, ey) > tol:
                rejected.append(last_pt)
                reasons[last_pt[0]] = (f"final point off the established curve's "
                                       f"extrapolation by {math.hypot(ex, ey):.0f}px "
                                       f"— likely the glove, not the ball")
                path = core_all
            else:
                break
        except Exception:
            break

    # ── Stage 4.65: final turn vs THIS path's own established smoothness ───
    # Every other check compares against a fixed or absolute threshold. This
    # one is relative: it looks at how much the heading has been turning
    # step-to-step throughout this specific path, and checks whether the
    # very last turn is consistent with that established pattern — not just
    # "under 70 degrees" in the abstract, but "in line with how smooth or
    # curved THIS pitch has actually been." A pitch that has been turning
    # ~5 degrees per step the whole time but suddenly turns 40 degrees at
    # the very end is suspicious even though 40 is well under the
    # hard-coded 70-degree limit elsewhere — that jump is a skew relative
    # to the path's own behavior, most often the last point sliding
    # slightly onto the mitt rather than the ball itself.
    #
    # The baseline uses a RECENT local window of turns, not the whole
    # path's median. Many real pitches genuinely curve more in their final
    # approach (real movement/break) — using the whole path's median lets
    # a long earlier straight section dilute that real recent curvature
    # down to a tiny baseline, making the final turn look anomalous when
    # it's actually consistent with what the pitch has ALREADY been doing
    # for the last several points. Confirmed directly: a pitch curving at
    # roughly 15-20 degrees per step over its last 6 points still got a
    # ~2-degree established baseline (dominated by an earlier straight
    # section), wrongly flagging its genuinely continuing next turn.
    turn_window = 6
    while len(path) >= 6:
        heads = [_angle(path[i - 1], path[i]) for i in range(1, len(path))]
        turns = [_angle_diff(heads[i - 1], heads[i]) for i in range(1, len(heads))]
        if len(turns) < 4:
            break
        recent_turns = turns[max(0, len(turns) - 1 - turn_window):-1]
        established = float(np.median(recent_turns)) if recent_turns else turns[-2]
        last_turn = turns[-1]
        # Floor of 20 degrees so a naturally very straight, low-turn
        # trajectory doesn't get flagged over tiny, meaningless noise.
        turn_tolerance = max(20.0, established * 3.0)

        # Confidence cross-check, same reasoning as Stage 4.5. A real pitch
        # can break TWICE — once mid-flight, settle into a new direction,
        # then break again slightly on its final approach to the mitt.
        # That second break can still exceed even the settled, post-first-
        # break smoothness. Confirmed directly on a real case, visually
        # verified frame by frame in the actual video: the point failing
        # this check sat right at the real catcher's glove, at a normal
        # confidence (0.78) for that flight — nothing like the 0.15-0.30
        # range seen in confirmed glove-contamination cases. Trust a
        # normal-confidence detection over a smoothness assumption.
        recent_conf = float(np.median([d[3] for d in path[-(turn_window + 1):-1]]))
        conf_ratio = path[-1][3] / recent_conf if recent_conf > 0.05 else 1.0
        # FIX: same reasoning and same validated threshold as Stage 4.5's
        # analogous confidence gate above -- 0.6 was too strict to catch
        # the median ~15% confidence drop seen in real glove-drag cases.
        if conf_ratio < 0.92:
            turn_tolerance = turn_tolerance
        else:
            turn_tolerance = turn_tolerance * 3.0

        if last_turn > turn_tolerance:
            d = path[-1]
            rejected.append(d)
            reasons[d[0]] = (f"final turn of {last_turn:.0f} deg vs this path's own "
                             f"established {established:.0f} deg — skewing off the flight path")
            path = path[:-1]
        else:
            break

    # ── Stage 4.6: trim trailing points where the ball has STOPPED ─────────
    # Once the ball is caught it stops travelling, but the detector keeps
    # finding it — now sitting in the glove, which then recoils and drags
    # the detection along with it. Those trailing points are not flight.
    # They're hard to catch with the earlier checks because a point that
    # merely stops SHORT is still roughly on the curve, just not as far
    # along it, so it doesn't deviate enough in position to trip Stage 4.5.
    # The giveaway is the step size collapsing relative to the flight's
    # RECENT pace — confirmed directly from real cases where trailing steps
    # fell to 3px and 1px against an established ~25-30px/frame, at
    # confidences of 0.15-0.5 versus ~0.9 for the real flight.
    #
    # The reference has to be the RECENT local pace, not the early part of
    # the flight. An earlier version used the first 70% of steps as the
    # reference — which broke on a genuinely, continuously decelerating
    # pitch (real perspective effect as the ball nears the camera): its own
    # early, faster steps made every later, legitimately-slower point look
    # "stalled" by comparison, cutting real flight short. Confirmed
    # directly: a real trajectory decelerating smoothly from 82px/frame
    # down to 12px/frame at consistently high confidence (~0.9) got its
    # last several genuine points wrongly trimmed this way.
    #
    # Note the stall check itself is still one-directional: a ball
    # approaching the camera accelerates in pixel terms, so a SUDDEN,
    # SHARP collapse relative to the recent trend is never real flight —
    # it's the catch. Gradual deceleration consistent with the recent
    # trend is real and must survive.
    # The reliable signal is an ABRUPT COLLAPSE in step size, not absolute
    # slowness. Two earlier versions each failed one way:
    #   - Comparing against the flight's EARLY pace wrongly trimmed a
    #     genuinely, smoothly decelerating pitch (real perspective effect as
    #     the ball nears the camera), because its own faster early steps
    #     made every later legitimate point look stalled.
    #   - Comparing against a RECENT local window fixed that, but then
    #     couldn't see a multi-point stalled tail at all: once several
    #     trailing steps are ~2-3px, the local reference is itself ~2-3px,
    #     so nothing looks collapsed. Confirmed on a real case with a tail
    #     of 17, 3, 2, 2, 3, 3 px that went completely untrimmed.
    # Detecting the abrupt drop itself handles both: smooth deceleration
    # has consecutive step ratios around 0.85-0.9 and never trips it, while
    # a catch shows a sudden 0.2x-0.4x collapse at a single identifiable
    # point. Physically justified too — a ball approaching the camera
    # accelerates in pixel terms, so an abrupt slowdown is never real flight.
    # Two tiers, because step ratio alone cannot separate the last two real
    # cases. A hard collapse is conclusive on its own. A moderate slowdown
    # is ambiguous — it can be the ball genuinely decelerating into the
    # catch (real, keep it) or the detection sliding onto the glove (drop
    # it) — so for that tier confidence breaks the tie. Confirmed against
    # two real cases that sit almost on top of each other by step ratio:
    # one at 0.49 with confidence 0.82 (a real final ball detection, must
    # be kept) and one at 0.47 with confidence 0.53 against a ~0.93 flight
    # (the glove, must be dropped).
    hard_drop_ratio = 0.35
    soft_drop_ratio = 0.55
    conf_drop_ratio = 0.7
    # Third tier: a moderate step-collapse (ratio between soft_drop_ratio
    # and this) that ALSO breaks from the flight's established direction.
    # Neither signal alone is reliable in this range — loosening
    # soft_drop_ratio on its own trims real decelerating pitches
    # elsewhere; a direction break alone (Stage 4.65) doesn't fire
    # because Stage 4.65 compares TURN magnitude, not this step's
    # heading against the established heading directly, and can miss
    # the same abrupt cases. But confirmed directly across a batch of
    # verified glove-drag cases: a moderate slowdown that coincides with
    # the flight suddenly changing direction is never real ball flight —
    # a decelerating pitch still travels in a physically continuous line,
    # it doesn't also swerve.
    moderate_drop_ratio = 0.85
    moderate_angle_break = 25.0
    ref_window = 4
    if len(path) >= 4:
        steps = [_rate(path[i - 1], path[i]) for i in range(1, len(path))]
        headings = [_angle(path[i - 1], path[i]) for i in range(1, len(path))]
        path_conf = float(np.median([d[3] for d in path]))
        # Only look in the back half — an abrupt drop mid-flight is more
        # likely brief occlusion than the catch, and shouldn't truncate
        # a real trajectory that recovers afterward.
        tail_start = max(1, int(len(steps) * 0.5))
        onset = None
        for i in range(tail_start, len(steps)):
            # Reference is the MEDIAN of the preceding few steps, never a
            # single step. Comparing against one step is fragile both ways:
            #   - False fire: if the preceding step was itself anomalously
            #     large (e.g. inflated by a gap left where bad points were
            #     rejected), a perfectly normal following step looks like a
            #     "collapse". Confirmed on a real case where an 88px gap
            #     artifact made a normal 28px step read as a stall and
            #     wrongly truncated the path.
            #   - Miss: a tail that decays gradually over several steps
            #     (25, 24, 16, 8) never shows a single sharp consecutive
            #     drop, so it slipped through entirely despite ending at a
            #     third of the flight's real pace.
            ref_steps = steps[max(0, i - ref_window):i]
            if not ref_steps:
                continue
            ref = float(np.median(ref_steps))
            if ref <= 1:
                continue
            ratio = steps[i] / ref
            # the point this step lands on is the candidate first bad point
            landed_conf = path[i + 1][3] if i + 1 < len(path) else path[-1][3]
            conf_ok = path_conf <= 0.05 or (landed_conf / path_conf) >= conf_drop_ratio

            moderate_hit = False
            if soft_drop_ratio <= ratio < moderate_drop_ratio:
                ref_headings = headings[max(0, i - ref_window):i]
                if ref_headings:
                    est_heading = float(np.median(ref_headings))
                    brk = _angle_diff(est_heading, headings[i])
                    if brk > moderate_angle_break:
                        moderate_hit = True

            if ratio < hard_drop_ratio or (ratio < soft_drop_ratio and not conf_ok) or moderate_hit:
                onset = i
                break
        if onset is not None and len(path[:onset + 1]) >= 4:
            ref_steps = steps[max(0, onset - ref_window):onset]
            ref = float(np.median(ref_steps)) if ref_steps else steps[onset - 1]
            for d in path[onset + 1:]:
                rejected.append(d)
                reasons[d[0]] = (f"ball stopped moving (step fell to "
                                 f"{steps[onset]:.0f}px vs recent {ref:.0f}px) "
                                 f"— caught, not in flight")
            path = path[:onset + 1]

    # anything outside the chosen run was never in contention
    for d in dets:
        if d[0] not in base_frames:
            rejected.append(d)
            reasons.setdefault(d[0], "outside the main flight segment")

    path.sort(key=lambda d: d[0])
    rejected.sort(key=lambda d: d[0])

    # The minimum-length rule has to be re-checked here, not just when
    # picking base_run — Stage 3's point-by-point rejections can shrink a
    # run that started long enough down to just a few surviving points.
    # Confirmed directly from real cases: a selected run with well over 13
    # points still finished with as few as 2-7 points after Stage 3 threw
    # out everything that didn't fit the shape. A path that thin is exactly
    # the kind of low-evidence result this rule exists to catch.
    if len(path) < min_path_length:
        rejected.extend(path)
        for d in path:
            reasons[d[0]] = (f"final path only had {len(path)} points after "
                              f"validation, below the {min_path_length}-point minimum")
        path = []

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
        "warning": alt_warning,
    }

    return {"path": path, "rejected": rejected, "impact": impact,
            "rows": rows, "summary": summary}
