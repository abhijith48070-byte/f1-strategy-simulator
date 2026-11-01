"""Experiment 4: do temporal / conventional ML models beat persistence at 5, 10, 15 laps ahead?
    python run_forecast.py --synthetic
    python run_forecast.py --exclude 2024_R08
"""
import argparse
import pandas as pd
from src.data import make_synthetic
from src.features import prepare
from src.forecast import loro_horizon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default="data/laps_openf1.csv")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    raw = make_synthetic(n_races=8, seed=5) if a.synthetic else pd.read_csv(a.csv)
    raw = raw[~raw["RaceId"].isin(a.exclude)]
    df = prepare(raw, "early")
    res, summ, wins, nrace = loro_horizon(df)
    print("Lap-time change forecast, leave-one-race-out (seconds; lower is better)")
    print(summ.to_string())
    print("\nRaces where the model beats persistence (MAE):")
    for (h, m), w in wins.items():
        print(f"  h={h:2d} {m:9s} {w}/{nrace[h]}")
    res.to_csv(f"{a.out}/forecast_per_race.csv", index=False)


if __name__ == "__main__":
    main()
