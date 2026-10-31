"""Same-laps leakage test and within-driver evaluation.
    python run_leakage_check.py
    python run_leakage_check.py --exclude 2024_R08
    python run_leakage_check.py --synthetic
"""
import argparse
from pathlib import Path
import pandas as pd
from src import config as C
from src.data import make_synthetic
from src.features import prepare, regression_set
from src.models import get_models
from src.evaluate import loro_regression
from src.within import loro_within

KEY = ["RaceId", "Driver", "LapNumber"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default="data/laps_openf1.csv")
    ap.add_argument("--exclude", nargs="*", default=[], help="RaceIds to drop, e.g. 2024_R08")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    raw = make_synthetic(n_races=8, seed=2) if a.synthetic else pd.read_csv(a.csv)
    if a.exclude:
        raw = raw[~raw["RaceId"].isin(a.exclude)]
    full = regression_set(prepare(raw, "full"))
    early = regression_set(prepare(raw, "early"))
    common = full[KEY].merge(early[KEY], on=KEY)
    full, early = full.merge(common, on=KEY), early.merge(common, on=KEY)
    feats = [f for f in C.REG_FEATURES if early[f].notna().any()]
    models = get_models()
    print(f"{raw['RaceId'].nunique()} races | {len(common)} laps present in BOTH versions | features: {feats}")

    rf, sf = loro_regression(full, feats, models, alpha=C.ALPHA)
    re_, se = loro_regression(early, feats, models, alpha=C.ALPHA)
    t = pd.DataFrame({
        "MAE_full_ref": sf[("MAE", "mean")], "MAE_causal_ref": se[("MAE", "mean")],
        "R2_full_ref": sf[("R2", "mean")], "R2_causal_ref": se[("R2", "mean")],
        "cover_full_ref": sf[("coverage", "mean")], "cover_causal_ref": se[("coverage", "mean")]}).round(3)
    print("\n1) Same laps, two targets (lower MAE is better; coverage target 0.90)")
    print(t.to_string())

    res, summ, beats, n_races = loro_within(full, feats, models)
    print("\n2) Within-driver evaluation (target = lap time minus the driver's own race mean)")
    print(summ.to_string())
    print(f"\nRaces (of {n_races}) where the model beats the flat line:\n{beats.to_string()}")

    out = Path(a.out); out.mkdir(exist_ok=True)
    t.to_csv(out / "leakage_same_laps.csv"); res.to_csv(out / "within_driver_per_race.csv", index=False)
    summ.to_csv(out / "within_driver_summary.csv")


if __name__ == "__main__":
    main()
