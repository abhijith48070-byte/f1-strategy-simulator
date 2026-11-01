"""Experiment 2: how much does informative pit-stop behaviour distort tyre-degradation estimates?
    python run_survivorship.py --synthetic            (known truth; shows the method works when its assumptions hold)
    python run_survivorship.py --exclude 2024_R08     (real data: data/laps_openf1.csv, cluster bootstrap over races)
"""
import argparse
import pandas as pd
from src.data import make_synthetic  # noqa: F401  (kept so the import path is exercised)
from src.features import prepare
from src.survival import compare_slopes, make_informative_stint_data, effective_sample_fraction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default="data/laps_openf1.csv")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    if a.synthetic:
        df, truth = make_informative_stint_data(n_races=12, seed=0)
        res, w = compare_slopes(df, n_boot=0)
        res["truth"] = [truth[0], truth[1], truth[2]]
        print("Synthetic races with informative stopping (known truth), s per lap of tyre age")
    else:
        raw = pd.read_csv(a.csv)
        raw = raw[~raw["RaceId"].isin(a.exclude)]
        df = prepare(raw, "full")    # 'full' is used here only as a data-cleaning rule; no reference pace enters the estimate
        res, w = compare_slopes(df, n_boot=a.boot)
        print(f"Real data, {df['RaceId'].nunique()} races; ipw_minus_naive CI is a 90% cluster bootstrap over races")
    print(res.round(4).to_string())
    print(f"\nEffective sample fraction of the IPW weights: {effective_sample_fraction(w):.2f} (max weight {w.max():.1f})")
    print("If ipw_minus_naive is small relative to its CI, informative pit-stop decisions do not materially bias the estimates here.")
    res.to_csv(f"{a.out}/survivorship_{'synthetic' if a.synthetic else 'real'}.csv")


if __name__ == "__main__":
    main()
