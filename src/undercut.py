"""Undercut / overcut analysis for one rival.

Car A (ours) is `gap_s` seconds BEHIND car B. B's stop is fixed at `b_pit_lap`. We scan A's stop lap:
  A stops before B  -> undercut (fresh tyres earlier, B is still on old rubber)
  A stops after B   -> overcut  (B loses time in the pit lane and on cold tyres, A keeps track position/clean air)
Both cars are compared over the same window, from now until `settle` laps after the later stop, so the fresh-tyre
gain, both pit losses and the old-tyre degradation of the car that stays out are all counted.

margin_s = (time B needs over the window - time A needs) - traffic_s - gap_s.   margin > 0 means A comes out AHEAD.
own_race_cost_s = what that stop lap costs A over the whole remaining race vs A's best single-stop lap (the price of the move).
P(success) = share of Monte Carlo worlds with margin > 0. Both cars share the race-level world (degradation,
weather) but have independent lap noise and independent pit-stop durations.

Simplifications: same pace model for both cars (optional `pace_delta_s_per_lap` shifts A), no DRS/overtaking model,
traffic is a flat cost per car A rejoins behind, and B does not react to A.
"""
import numpy as np
import pandas as pd
from . import config as C
from .planner import NAMES


def _window_time(model, world, state, comp_now, age_now, comp_new, pit_lap, n0, end, L, pit_loss, pace_delta=0.0):
    """(n_sim,) seconds for laps n0..end for a car that stops at the end of pit_lap onto comp_new."""
    laps1 = np.arange(n0, min(pit_lap, end) + 1)
    ages1 = age_now + (laps1 - n0)
    t = model.sample(world, comp_now, ages1, laps1, L, 0.0, state).sum(axis=1)
    if pit_lap < end:
        laps2 = np.arange(pit_lap + 1, end + 1)
        t = t + model.sample(world, comp_new, 1.0 + (laps2 - pit_lap - 1), laps2, L, 1.0, state).sum(axis=1) + pit_loss
    return t + pace_delta * (end - n0 + 1)


def own_race_cost(model, state, a, pit_loss, laps):
    """Seconds our own race loses (whole remaining race, mean model) by stopping on each lap in `laps` instead of on
    our best single-stop lap with the same tyres. An undercut that is 'free' in the window may not be free overall."""
    from .planner import Planner, Constraints
    pl = Planner(model, state, pit_loss, None, Constraints(max_stops=1, two_compounds=False, allowed=(a["new_compound"],)))
    base = {p: pl.evaluate([a["compound"], a["new_compound"]], [p]) for p in range(state["lap"], state["total_laps"])}
    best = min(base.values())
    return {p: base[p] - best for p in laps if p in base}


def analyse(model, state, a, b, gap_s, pit_loss_samples, b_pit_lap, scan=None, settle=4, traffic_cars=0,
            pace_delta_s_per_lap=0.0, n_sim=3000, seed=0, max_lead=6):
    """a, b: dict(compound=code, tyre_life=age_now, new_compound=code). Returns (table, summary dict).
    Default scan: from `max_lead` laps before the rival's stop to 4 laps after it."""
    rng = np.random.default_rng(seed)
    n0, L = int(state["lap"]), int(state["total_laps"])
    scan = list(scan) if scan is not None else list(range(max(n0, b_pit_lap - max_lead), min(b_pit_lap + 4, L - 1) + 1))
    cost = own_race_cost(model, state, a, float(np.median(pit_loss_samples)), scan)
    end = min(L, max(max(scan), b_pit_lap) + settle)
    wa = model.draw_world(n_sim, L, rng, state)
    wb = dict(wa)                                           # same race-level world ...
    wb["eps"] = model.draw_world(n_sim, L, rng, state)["eps"]   # ... but independent lap-to-lap noise
    ls = np.asarray(pit_loss_samples, float)
    loss_b = rng.choice(ls, n_sim)
    tb = _window_time(model, wb, state, b["compound"], b["tyre_life"], b["new_compound"], b_pit_lap, n0, end, L, loss_b)
    traffic = traffic_cars * C.DIRTY_AIR_LOSS_PER_CAR_S
    rows = []
    for pa in scan:
        loss_a = rng.choice(ls, n_sim)
        ta = _window_time(model, wa, state, a["compound"], a["tyre_life"], a["new_compound"], pa, n0, end, L, loss_a,
                          pace_delta_s_per_lap)
        margin = (tb - ta) - traffic - gap_s
        kind = "undercut" if pa < b_pit_lap else ("overcut" if pa > b_pit_lap else "same lap")
        rows.append(dict(a_pit_lap=pa, kind=kind, net_gain_s=float((tb - ta).mean()), traffic_s=traffic,
                         own_race_cost_s=float(cost.get(pa, np.nan)),
                         margin_s=float(margin.mean()), margin_p05=float(np.percentile(margin, 5)),
                         margin_p95=float(np.percentile(margin, 95)), p_success=float((margin > 0).mean())))
    t = pd.DataFrame(rows)
    summary = {}
    for kind in ("undercut", "overcut"):
        s = t[t["kind"] == kind]
        if s.empty:
            summary[kind] = None
            continue
        best = s.loc[s["margin_s"].idxmax()]
        summary[kind] = dict(best_lap=int(best["a_pit_lap"]), margin_s=float(best["margin_s"]),
                             own_cost_s=float(best["own_race_cost_s"]),
                             p_success=float(best["p_success"]), detected=bool(best["margin_s"] > 0 and best["p_success"] >= 0.6))
    summary["text"] = _text(summary, gap_s, b_pit_lap, traffic_cars, a, b)
    return t, summary


def _text(summary, gap_s, b_pit_lap, traffic_cars, a, b):
    lines = []
    for kind, verb in (("undercut", "Pit before"), ("overcut", "Pit after")):
        s = summary.get(kind)
        if s is None:
            continue
        if s["detected"]:
            lines.append(f"{kind.capitalize()} opportunity detected: {verb.lower()} the rival (lap {s['best_lap']} vs lap "
                         f"{b_pit_lap}), estimated advantage {s['margin_s']:.1f} s, P(success) {s['p_success']:.0%}; "
                         f"costs our own race {s['own_cost_s']:.1f} s versus our best stop lap.")
        else:
            lines.append(f"No {kind} opportunity: best option (lap {s['best_lap']}) leaves the car {abs(s['margin_s']):.1f} s "
                         f"{'ahead' if s['margin_s'] > 0 else 'short'}, P(success) {s['p_success']:.0%}.")
    if traffic_cars:
        lines.append(f"Includes a {traffic_cars * C.DIRTY_AIR_LOSS_PER_CAR_S:.1f} s traffic cost for {traffic_cars} car(s) "
                     f"after rejoining (assumption: {C.DIRTY_AIR_LOSS_PER_CAR_S} s per car).")
    lines.append(f"Rival ({NAMES[b['compound']]}, {b['tyre_life']:.0f} laps old) assumed to stop on lap {b_pit_lap} and not to react; "
                 f"gap {gap_s:.1f} s.")
    return lines
