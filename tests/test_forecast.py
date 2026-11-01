"""Run:  python -m tests.test_forecast"""
import numpy as np
from src.data import make_synthetic
from src.features import prepare
from src.forecast import horizon_table, loro_horizon


def test_targets_use_only_the_future_lap_and_no_pit_crossing():
    df = prepare(make_synthetic(n_races=3, seed=1), "early")
    t = horizon_table(df, 5)
    assert len(t) > 500
    assert (t["Stint"].notna()).all()
    # every pair stays inside one stint: tyre age at t+5 is age(t)+5
    base = df.set_index(["RaceId", "Driver", "LapNumber"])["TyreLife"]
    ahead = base.reindex(list(zip(t.RaceId, t.Driver, t.LapNumber + 5))).to_numpy()
    assert np.allclose(ahead, t["TyreLife"].to_numpy() + 5)


def test_models_beat_persistence_on_synthetic_drift():
    df = prepare(make_synthetic(n_races=6, seed=3), "early")
    res, summ, wins, n = loro_horizon(df, horizons=(10,))
    mae = res.groupby("model")["MAE"].mean()
    assert mae["linear"] < mae["persistence"], mae


if __name__ == "__main__":
    test_targets_use_only_the_future_lap_and_no_pit_crossing(); print("ok   targets")
    test_models_beat_persistence_on_synthetic_drift(); print("ok   models vs persistence")
