"""Experiment 3: can conformal methods give calibrated uncertainty for unseen races?
    python run_uncertainty.py --synthetic
    python run_uncertainty.py --exclude 2024_R08 --kind linear
"""
import argparse
import pandas as pd
from src import config as C
from src.data import make_synthetic
from src.features import prepare, regression_set
from src.uncertainty import evaluate_all, summarize, conditional_coverage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default="data/laps_openf1.csv")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--kind", choices=["linear", "hgb"], default="linear")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--no-cqr", action="store_true", help="skip quantile models (much faster)")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    raw = make_synthetic(n_races=10, seed=4) if a.synthetic else pd.read_csv(a.csv)
    raw = raw[~raw["RaceId"].isin(a.exclude)]
    reg = regression_set(prepare(raw, "early"))
    feats = [f for f in ["LapFrac", "TyreLife", "CompoundCode"] if reg[f].notna().any()]
    print(f"{reg['RaceId'].nunique()} races, {len(reg)} laps, causal reference, point model = {a.kind}, features {feats}\n"
          f"nominal coverage {1 - a.alpha:.0%}; calibration never uses the test race\n")
    res, store = evaluate_all(reg, feats, alpha=a.alpha, kind=a.kind, with_cqr=not a.no_cqr)
    s = summarize(res, a.alpha)
    print(s.to_string())
    by_age, by_comp = conditional_coverage(store)
    print("\nConditional coverage by tyre age:\n", by_age.to_string(), "\n\nBy compound:\n", by_comp.to_string())
    res.to_csv(f"{a.out}/uncertainty_per_race.csv", index=False); s.to_csv(f"{a.out}/uncertainty_summary.csv")


if __name__ == "__main__":
    main()
