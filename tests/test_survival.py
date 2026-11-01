"""Run:  python -m tests.test_survival"""
import numpy as np
import pandas as pd
from src.survival import (make_informative_stint_data, add_stint_features, person_period, fit_hazards,
                          ipw_weights, degradation_slopes, COMPS)


def _bias(strength, seeds=(0, 1, 2)):
    nb, ib = [], []
    for s in seeds:
        df, truth = make_informative_stint_data(n_races=12, seed=s, strength=strength)
        d = add_stint_features(df)
        pp = person_period(d)
        hf, hm = fit_hazards(pp)
        w = pd.Series(1.0, index=d.index)
        w.loc[pp.index] = ipw_weights(pp, hf, hm)
        n, i = degradation_slopes(d), degradation_slopes(d, weights=w.to_numpy())
        nb.append([n[f"age_{c}"] - truth[c] for c in COMPS])
        ib.append([i[f"age_{c}"] - truth[c] for c in COMPS])
    return np.mean(nb, axis=0), np.mean(ib, axis=0)


def test_informative_stopping_biases_naive_and_ipw_helps():
    nb, ib = _bias(strength=1.0)
    assert nb[0] < -0.02, nb                         # naive SOFT slope clearly too low
    assert abs(ib[0]) < 0.7 * abs(nb[0]), (nb, ib)   # IPW removes a large part of the bias
    assert all(abs(i) <= abs(n) + 0.002 for n, i in zip(nb, ib)), (nb, ib)


def test_no_informative_stopping_means_no_correction():
    nb, ib = _bias(strength=0.0)
    assert np.all(np.abs(nb) < 0.012), nb
    assert np.all(np.abs(ib - nb) < 0.012), (nb, ib)


def test_features_are_causal():
    df, _ = make_informative_stint_data(n_races=2, n_drivers=3, seed=0)
    d = add_stint_features(df)
    d2 = add_stint_features(df.assign(LapTimeS=np.where(df.LapNumber > 30, df.LapTimeS + 50, df.LapTimeS)))
    a = d[d.LapNumber <= 30].sort_values(["StintId", "LapNumber"])["StintLoss"].to_numpy()
    b = d2[d2.LapNumber <= 30].sort_values(["StintId", "LapNumber"])["StintLoss"].to_numpy()
    assert np.allclose(a, b), "StintLoss at lap<=30 changed when only laps>30 were altered (future leakage)"


if __name__ == "__main__":
    for f in (test_informative_stopping_biases_naive_and_ipw_helps, test_no_informative_stopping_means_no_correction,
              test_features_are_causal):
        f()
        print("ok  ", f.__name__)
