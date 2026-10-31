"""Offline self-test on synthetic races with known physics.  Run:  python -m tests.test_pipeline"""
import numpy as np
from src import config as C
from src.data import make_synthetic
from src.features import prepare, regression_set, pit_set, pit_loss_samples
from src.models import get_models
from src.evaluate import loro_regression, conformal_quantile
from src.simulate import pit_window_table


def main():
    df = prepare(make_synthetic(n_races=6, seed=1))
    reg = regression_set(df)

    # no label leakage: PitThisLap must not be recoverable from a feature
    ps = pit_set(df)
    assert not any(c in C.PIT_FEATURES for c in ["PitInTimeS", "PitOutTimeS", "AdjustedLapTime"])
    assert ps["PitThisLap"].sum() > 0

    # cleaning removed in/out laps and safety-car laps
    assert not reg["IsInLap"].any() and not reg["IsOutLap"].any() and (reg["TrackStatus"] == "1").all()

    # model beats the baseline on held-out races
    res, summ = loro_regression(reg, C.REG_FEATURES, get_models(), alpha=0.1)
    mae = res.groupby("model")["MAE"].mean()
    assert mae["hist_gbr"] < mae["baseline_compound_mean"], mae
    cov = res[res.model == "hist_gbr"]["coverage"].mean()
    assert 0.85 <= cov <= 0.95, f"conformal coverage off target: {cov:.3f}"

    # conformal quantile math
    assert abs(conformal_quantile(np.arange(1, 100), 0.1) - 90) < 1e-9

    # estimated pit loss is near the true synthetic value (17 s out-lap + 2 s in-lap = ~19 s)
    loss = float(np.median(pit_loss_samples(df)["loss_s"]))
    assert 15 < loss < 26, loss
    assert pit_loss_samples(df)["loss_s"].max() <= 60

    # simulation: pitting earlier on old SOFTs should beat pitting much later
    models = get_models()
    m = models["hist_gbr"].fit(reg[C.REG_FEATURES], reg["DeltaS"])
    state = dict(lap=15, total_laps=60, tyre_life=15, compound=0, fresh=0.0,
                 track_temp=35.0, air_temp=25.0, position=8.0)
    table, _ = pit_window_table(m, C.REG_FEATURES, state, C.COMPOUNDS["HARD"], loss)
    assert len(table) == 60 - 15
    print("all tests passed | held-out MAE:", mae.round(3).to_dict(), "| coverage:", round(cov, 3),
          "| pit loss:", round(loss, 1))


if __name__ == "__main__":
    main()
