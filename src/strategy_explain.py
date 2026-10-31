"""Strategy Intelligence Engine: explain WHY a strategy was chosen, what would change it, and what it risks.

Every sentence is generated from numbers computed by the planner / Monte Carlo / scenario table. Nothing here is
templated opinion: if a number is not available, the sentence is not produced.
"""
import numpy as np
from .planner import Planner, family, NAMES
from .montecarlo import compare


def _key(row):
    return (row["family"], row["pit_laps"][0] if len(row["pit_laps"]) else None)


def _flip_range(planner, attr, values, current, base_key):
    """Smallest and largest tested value of `attr` for which the best (family, first stop) is unchanged,
    scanning outward from `current`. Returns (low, high, low_flip_key, high_flip_key); flip key None = no flip."""
    orig = getattr(planner, attr)
    out = {}
    for direction, seq in (("down", [v for v in values if v < current][::-1]), ("up", [v for v in values if v > current])):
        edge, flip = None, None
        for v in seq:
            setattr(planner, attr, v)
            best = planner.enumerate().iloc[0]
            if _key(best) != base_key:
                flip = (v, best["strategy"])
                break
            edge = v
        out[direction] = (edge, flip)
    setattr(planner, attr, orig)
    return out


def explain(model, state, pit_loss, table, fam, res, summ, scen, constraints=None, neutral_share=None):
    """table: all strategies (sorted); fam: best-per-family shortlist used in the Monte Carlo (same order as res);
    res/summ: SimResult and its summary; scen: scenario_table output."""
    best = fam.iloc[0]                       # the recommended strategy (index 0 of the Monte Carlo shortlist)
    mean_best = table.iloc[0]                # the deterministic mean-optimal strategy (sensitivity is computed on this)
    why, changes, risks = [], [], []
    bk = _key(mean_best)

    # ---- why this strategy ---------------------------------------------------------------------------
    if len(fam) > 1:
        ru = fam.iloc[1]
        cmp_ = compare(res, 0, 1)
        d_pit = ru["pit_time_s"] - best["pit_time_s"]
        d_tyre = (ru["expected_delta_s"] - ru["pit_time_s"]) - (best["expected_delta_s"] - best["pit_time_s"])
        why.append(f"{best['strategy']} beats {ru['strategy']} by {cmp_['mean_advantage_s']:.1f} s on average "
                   f"(90% range {cmp_['p05']:.1f} to {cmp_['p95']:.1f} s) and is faster in {cmp_['p_i_faster']:.0%} of simulated races.")
        why.append(f"The difference comes from {'less' if d_pit >= 0 else 'more'} time in the pits ({abs(d_pit):.1f} s "
                   f"{'saved' if d_pit >= 0 else 'extra'}) and {abs(d_tyre):.1f} s {'less' if d_tyre >= 0 else 'more'} lost to "
                   f"tyre wear and lap-time differences.")
    # pit-window flatness: what does the exact lap cost?
    if len(best["pit_laps"]):
        p = best["pit_laps"][0]
        cost = {}
        for k in (-3, -1, 1, 3):
            q = (p + k,) + tuple(best["pit_laps"][1:])
            m = table[(table["compounds"] == best["compounds"]) & (table["pit_laps"] == q)]
            if len(m):
                cost[k] = float(m.iloc[0]["expected_delta_s"] - best["expected_delta_s"])
        if cost:
            why.append("Cost of moving the first stop: " + ", ".join(
                f"{abs(k)} lap{'s' if abs(k) > 1 else ''} {'earlier' if k < 0 else 'later'} = +{v:.1f} s"
                for k, v in sorted(cost.items())) + ".")

    # ---- what would change the recommendation -------------------------------------------------------
    pl = Planner(model, state, pit_loss, None, constraints)
    if _key(best) != bk:
        changes.append(f"(Recommendation differs from the mean-optimal plan {mean_best['strategy']}; the sensitivity checks "
                       f"below are for the mean-optimal plan.)")
    fl = _flip_range(pl, "pit_loss", list(np.arange(max(pit_loss - 12, 5), pit_loss + 12.1, 1.0)), pit_loss, bk)
    lo, hi = fl["down"], fl["up"]
    if lo[1] or hi[1]:
        parts = []
        if hi[1]:
            parts.append(f"a pit loss above {hi[0] if hi[0] is not None else pit_loss:.0f} s (e.g. slow stops) → {hi[1][1]}")
        if lo[1]:
            parts.append(f"a pit loss below {lo[0] if lo[0] is not None else pit_loss:.0f} s → {lo[1][1]}")
        changes.append("Pit loss: the plan holds around the estimated " + f"{pit_loss:.0f} s; " + "; ".join(parts) + ".")
    else:
        changes.append(f"Pit loss: the plan is unchanged for any pit loss from {pit_loss - 12:.0f} to {pit_loss + 12:.0f} s.")
    pl.pit_loss = pit_loss
    fd = {}
    for direction, vals in (("down", [0.9, 0.8, 0.7, 0.6, 0.5]), ("up", [1.1, 1.2, 1.3, 1.5, 1.75, 2.0])):
        for v in vals:
            pl.set_deg_mult(v)
            b2 = pl.enumerate().iloc[0]
            if _key(b2) != bk:
                fd[direction] = (v, b2["strategy"])
                break
    pl.set_deg_mult(1.0)
    if fd:
        parts = [f"degradation {int(round((v - 1) * 100)):+d}% → {s}" for v, s in
                 [fd[k] for k in ("down", "up") if k in fd]]
        changes.append("Tyre degradation: " + "; ".join(parts) + ".")
    else:
        changes.append("Tyre degradation: the plan is unchanged for degradation between -50% and +100% of the model estimate.")
    for _, r in scen[scen["changes_plan"]].iterrows():
        changes.append(f"{r['scenario']}: switch to {r['best_strategy']}; adapting is worth {r['adaptation_gain_s']:.1f} s "
                       f"versus sticking to the current plan.")

    # ---- risks --------------------------------------------------------------------------------------
    s0 = summ.iloc[0]
    risks.append(f"Probability of being the fastest of the {len(fam)} compared strategies: {s0['p_fastest']:.0%}; "
                 f"average regret {s0['mean_regret_s']:.1f} s; 10th-90th percentile finish {s0['p10']} to {s0['p90']}.")
    if len(summ) > 1:
        top_p = summ.loc[summ["p_fastest"].idxmax()]
        low_mean = summ.loc[summ["mean_s"].idxmin()]
        if top_p["strategy"] != low_mean["strategy"]:
            risks.append(f"The strategy with the best average ({low_mean['strategy']}) is not the one most likely to be "
                         f"fastest ({top_p['strategy']}); the choice depends on whether you value average time or consistency.")
    stints = list(zip(best["compounds"], np.diff([state["lap"] - 1] + list(best["pit_laps"]) + [state["total_laps"]])))
    stints[0] = (stints[0][0], stints[0][1] + state["tyre_life"] - 1)          # current stint includes laps already run
    ratio = [(n / model.life_limit(c), c, n) for c, n in stints]
    r, c, n = max(ratio)
    risks.append(f"Longest use of a set of tyres in the plan: {n:.0f} laps on {NAMES[c]}, which is {r:.0%} of the longest "
                 f"{NAMES[c]} stint seen in the data ({model.life_limit(c):.0f} laps). Beyond the data the model has no "
                 f"information (a tyre 'cliff' is not modelled).")
    if neutral_share is not None:
        best_gain = scen["adaptation_gain_s"].max() if len(scen) else 0.0
        risks.append(f"A safety car or VSC occurs in {neutral_share:.0%} of the simulated races. This plan is committed in the "
                     f"Monte Carlo; reacting to a neutralisation is worth up to {best_gain:.1f} s (scenario table).")
    risks.append("Not modelled: traffic after rejoining, pit-lane closure, rain, tyre warm-up, and rivals' strategies. "
                 "Use the undercut/overcut tool for rival-specific decisions.")
    return dict(why=why, changes=changes, risks=risks)
