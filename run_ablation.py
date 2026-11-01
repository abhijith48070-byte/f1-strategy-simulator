"""Within-driver ablation: does tyre age add anything beyond race progress (fuel burn + track evolution)?
    python run_ablation.py --exclude 2024_R08
    python run_ablation.py --synthetic
"""
import argparse
import pandas as pd
from src.data import make_synthetic
from src.features import prepare, regression_set
from src.models import get_models
from src.within import loro_within

SETS = {
    "LapFrac only":                      ["LapFrac"],
    "TyreLife only":                     ["TyreLife"],
    "LapFrac + TyreLife":                ["LapFrac", "TyreLife"],
    "LapFrac + TyreLife + Compound":     ["LapFrac", "TyreLife", "CompoundCode"],
    "all features":                      ["LapFrac", "TyreLife", "CompoundCode", "FreshTyre", "TrackTemp", "AirTemp"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default="data/laps_openf1.csv")
    ap.add_argument("--exclude", nargs="*", default=[])
    a = ap.parse_args()
    raw = make_synthetic(n_races=8, seed=2) if a.synthetic else pd.read_csv(a.csv)
    if a.exclude:
        raw = raw[~raw["RaceId"].isin(a.exclude)]
    reg = regression_set(prepare(raw, "full"))
    models = {k: v for k, v in get_models().items() if k in ("linear", "hist_gbr", "lightgbm")}
    print(f"{reg['RaceId'].nunique()} races, {len(reg)} laps. Skill = 1 - SSE/SSE(flat line), leave-one-race-out.\n")
    per_race = {}
    rows = []
    for label, feats in SETS.items():
        feats = [f for f in feats if reg[f].notna().any()]
        res, summ, _, _ = loro_within(reg, feats, models)
        for m in models:
            r = res[res.model == m].set_index("test_race")["skill_vs_zero"]
            per_race[(label, m)] = r
            rows.append(dict(features=label, model=m, skill_mean=r.mean(), skill_sd=r.std()))
    t = pd.DataFrame(rows).pivot(index="features", columns="model", values="skill_mean").reindex(SETS.keys()).round(3)
    print(t.to_string())
    print("\nPaired gain from adding TyreLife to LapFrac (skill difference, per held-out race):")
    for m in models:
        d = per_race[("LapFrac + TyreLife", m)] - per_race[("LapFrac only", m)]
        print(f"  {m:9s} mean {d.mean():+.3f}, sd {d.std():.3f}, better in {(d > 0).sum()}/{len(d)} races")


if __name__ == "__main__":
    main()
