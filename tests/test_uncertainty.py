"""Run:  python -m tests.test_uncertainty"""
import numpy as np
from src.data import make_synthetic
from src.features import prepare, regression_set
from src.uncertainty import conformal_q, cross_race_residuals, aci_bounds, evaluate_all, crps_from_quantiles, pinball, TAUS


def test_conformal_quantile():
    assert abs(conformal_q(np.arange(1, 100), 0.1) - 90) < 1e-9


def test_calibration_never_sees_test_race():
    reg = regression_set(prepare(make_synthetic(n_races=6, seed=1), "early"))
    train = reg[reg.RaceId != "SYN_R00"]
    res = cross_race_residuals(train, ["LapFrac", "TyreLife", "CompoundCode"], "DeltaS", "linear", k=3)
    assert "SYN_R00" not in set(res.RaceId)
    assert set(res.RaceId) == set(train.RaceId)          # every training race is scored out-of-fold exactly once
    assert len(res) == len(train)


def test_aci_restores_coverage_when_the_test_race_is_noisier():
    rng = np.random.default_rng(0)
    pool = np.abs(rng.normal(0, 1.0, 5000))              # calibration residuals from a calm world (sd 1)
    y = rng.normal(0, 2.0, 4000)                          # the test race is twice as noisy
    pred = np.zeros_like(y)
    static_cov = np.mean(np.abs(y) <= conformal_q(pool, 0.1))
    lo, hi = aci_bounds(pred, y, pool, alpha=0.1, gamma=0.02)
    aci_cov = np.mean((y >= lo) & (y <= hi))
    assert static_cov < 0.75 and aci_cov > 0.86, (static_cov, aci_cov)


def test_pinball_and_crps_are_zero_for_perfect_quantiles():
    y = np.array([1.0, 2.0, 3.0])
    assert pinball(y, y, 0.9) == 0.0
    assert crps_from_quantiles(y, np.column_stack([y] * len(TAUS))) == 0.0


def test_methods_reach_nominal_coverage_on_synthetic():
    reg = regression_set(prepare(make_synthetic(n_races=7, seed=2), "early"))
    res, _ = evaluate_all(reg, ["LapFrac", "TyreLife", "CompoundCode"], with_cqr=False)
    cov = res.groupby("method")["coverage"].mean()
    assert abs(cov["cross_race_conformal"] - 0.9) < 0.04, cov
    assert abs(cov["cross_race_ACI"] - 0.9) < 0.04, cov


if __name__ == "__main__":
    for f in (test_conformal_quantile, test_calibration_never_sees_test_race,
              test_aci_restores_coverage_when_the_test_race_is_noisier,
              test_pinball_and_crps_are_zero_for_perfect_quantiles, test_methods_reach_nominal_coverage_on_synthetic):
        f(); print("ok  ", f.__name__)
