"""Monte Carlo race outcomes with common random numbers.

Every simulated race draws ONE world: model uncertainty (posterior coefficients or bootstrapped residuals), per-lap
noise, a track-temperature shift (point-forecast models), a safety car / VSC (or none), and pit-stop durations.
Every strategy is priced in the SAME worlds, so differences between strategies are paired and meaningful, and
"probability of being fastest" is computed per simulated race instead of from independent draws.

A strategy here is a committed plan (pit laps fixed in advance); it does not react to a safety car. The value of
reacting is quantified separately by optimizer.scenario_table / reactive_decision.
"P(fastest)" is the chance a strategy is the quickest among those compared. It is NOT a race-win probability:
there are no opponents in this simulation.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import config as C
from .scenarios import NeutralArrays, sample_neutrals, neutral_stats
from .planner import fmt_time


@dataclass
class SimResult:
    labels: list
    totals: np.ndarray        # (n_sim, n_strategies) extra seconds (lap deltas + pit losses)
    base_s: float             # elapsed + remaining laps * reference pace (same for all strategies)
    neutral_share: float      # fraction of simulated races containing an SC/VSC


def simulate(model, state, plans, pit_loss_samples, n_sim=C.N_SIM, neutral="random", stats=None, seed=0):
    """plans: list of (compounds, pit_laps, label). neutral: "random", None, or a fixed Neutral."""
    rng = np.random.default_rng(seed)
    n0, L = int(state["lap"]), int(state["total_laps"])
    ref = float(state.get("ref_lap_s", 0.0))
    if neutral == "random":
        st, du, ki = sample_neutrals(n_sim, n0, L, stats or dict(p_sc=C.P_SC_DEFAULT, p_vsc=C.P_VSC_DEFAULT,
                                                                 starts=[], durations=[]), rng)
        arr = NeutralArrays(L, ref, st, du, ki)
    elif neutral is None:
        arr = NeutralArrays(L, ref, np.zeros(n_sim), np.zeros(n_sim), np.zeros(n_sim))
    else:
        from .scenarios import KIND_CODE
        arr = NeutralArrays(L, ref, np.full(n_sim, neutral.start), np.full(n_sim, neutral.duration),
                            np.full(n_sim, KIND_CODE[neutral.kind]))
    world = model.draw_world(n_sim, L, rng, state)
    max_stops = max(len(p) for _, p, _ in plans)
    losses = rng.choice(np.asarray(pit_loss_samples, float), size=(n_sim, max_stops + 1))
    out = np.zeros((n_sim, len(plans)))
    for k, (comps, pits, _) in enumerate(plans):
        bounds = [n0] + [p + 1 for p in pits]
        ends = list(pits) + [L]
        for i, (c, s, e) in enumerate(zip(comps, bounds, ends)):
            laps = np.arange(s, e + 1)
            start_age, fresh = (float(state["tyre_life"]), float(state["fresh"])) if i == 0 else (1.0, 1.0)
            ages = arr.age(start_age, s, laps)
            raw = model.sample(world, c, ages, laps, L, fresh, state)
            out[:, k] += np.where(arr.mask[:, laps], arr.lap_extra[:, laps], raw).sum(axis=1)
            if i < len(pits):
                out[:, k] += losses[:, i] * arr.pit_factor[:, pits[i]]
    elapsed = float(state.get("elapsed_s", (n0 - 1) * ref))
    return SimResult([lbl for _, _, lbl in plans], out, elapsed + (L - n0 + 1) * ref, float(arr.mask.any(axis=1).mean()))


def summarize(res):
    """Mean, spread and P(fastest) per strategy. regret = how much slower than the best strategy of that same race."""
    t = res.totals
    best = t.min(axis=1)
    winner = t.argmin(axis=1)
    rows = []
    for k, lbl in enumerate(res.labels):
        x = t[:, k]
        rows.append(dict(strategy=lbl, mean_s=x.mean() + res.base_s, p10_s=np.percentile(x, 10) + res.base_s,
                         p90_s=np.percentile(x, 90) + res.base_s, sd_s=x.std(),
                         mean_regret_s=float((x - best).mean()), p_fastest=float((winner == k).mean())))
    d = pd.DataFrame(rows)
    for c in ["mean_s", "p10_s", "p90_s"]:
        d[c.replace("_s", "")] = d[c].map(fmt_time)
    return d


def compare(res, i, j):
    """Paired comparison: how much faster is strategy i than j (positive = i faster), 90% interval, P(i faster)."""
    diff = res.totals[:, j] - res.totals[:, i]
    return dict(mean_advantage_s=float(diff.mean()), p05=float(np.percentile(diff, 5)),
                p95=float(np.percentile(diff, 95)), p_i_faster=float((diff > 0).mean()))
