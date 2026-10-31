"""Pit-window simulation done properly.

The same driver, on the same lap, is compared under different futures:
stay out vs pit at lap p and fit a new set of tyres. Total race time for each future is the sum of
model-predicted lap deltas plus one pit loss. Monte Carlo adds model error (bootstrapped from
calibration residuals) and pit-loss variation.

Known simplifications: no traffic, no safety cars, no two-compound rule, lap-to-lap errors treated as
independent, driver reference pace assumed known. State these in the report.
"""
import numpy as np
import pandas as pd


def _rows(lap_nums, tyre_life, compound, total_laps, fresh, state):
    return pd.DataFrame({
        "TyreLife": tyre_life, "LapFrac": np.asarray(lap_nums) / total_laps, "CompoundCode": compound,
        "FreshTyre": fresh, "TrackTemp": state["track_temp"], "AirTemp": state["air_temp"],
        "Position": state["position"]})


def strategy_totals(model, features, state, new_compound, pit_loss):
    """Return (mean total delta for each pit lap, per-lap predictions for each option)."""
    n, L = state["lap"], state["total_laps"]
    laps = np.arange(n, L + 1)
    stay = model.predict(_rows(laps, state["tyre_life"] + (laps - n), state["compound"], L,
                               state["fresh"], state)[features])
    out = {"no_stop": (float(stay.sum()), stay, None)}
    for p in range(n, L):                       # pit at the end of lap p
        before = stay[: p - n + 1]
        new_laps = np.arange(p + 1, L + 1)
        after = model.predict(_rows(new_laps, np.arange(1, len(new_laps) + 1), new_compound, L,
                                    1.0, state)[features])
        out[p] = (float(before.sum() + after.sum() + pit_loss), before, after)
    return out


def pit_window_table(model, features, state, new_compound, pit_loss):
    res = strategy_totals(model, features, state, new_compound, pit_loss)
    ref = res["no_stop"][0]
    rows = [dict(pit_lap=k, expected_total_s=v[0], vs_no_stop_s=v[0] - ref)
            for k, v in res.items() if k != "no_stop"]
    t = pd.DataFrame(rows).sort_values("expected_total_s").reset_index(drop=True)
    return t, ref


def monte_carlo(model, features, state, new_compound, pit_loss_samples, residuals,
                pit_laps, n_sim=2000, seed=0):
    """Simulate total time for chosen pit laps. residuals: signed calibration residuals (seconds)."""
    rng = np.random.default_rng(seed)
    res = strategy_totals(model, features, state, new_compound, float(np.median(pit_loss_samples)))
    out, sims = [], {}
    for p in pit_laps:
        mean_total, before, after = res[p]
        n_laps = len(before) + len(after)
        base = mean_total - float(np.median(pit_loss_samples))
        noise = rng.choice(residuals, size=(n_sim, n_laps)).sum(axis=1)
        sims[p] = base + rng.choice(pit_loss_samples, size=n_sim) + noise
        out.append(dict(pit_lap=p, mean_s=sims[p].mean(), p05=np.percentile(sims[p], 5),
                        p95=np.percentile(sims[p], 95)))
    return pd.DataFrame(out), sims


def prob_better(sim_a, sim_b):
    """P(total_a < total_b) using independent draws (approximation)."""
    return float(np.mean(sim_a < sim_b))
