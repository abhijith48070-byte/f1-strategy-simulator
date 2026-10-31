"""Run the full experiment:  python run_experiment.py            (real FastF1 data, needs internet)
                            python run_experiment.py --synthetic (offline self-test on fake races)"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone

from src import config as C
from src.data import build_dataset, make_synthetic
from src.features import prepare, regression_set, pit_set, pit_loss_samples
from src.models import get_models
from src.evaluate import loro_regression, loro_pit_classifier
from src.explain import permutation_loro, shap_summary
from src.simulate import pit_window_table, monte_carlo, prob_better


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--year", type=int, default=C.YEAR)
    ap.add_argument("--rounds", type=int, nargs="+", default=C.DEFAULT_ROUNDS)
    ap.add_argument("--out", default="results")
    ap.add_argument("--source", choices=["openf1", "fastf1"], default="openf1",
                    help="openf1 = free, no account (default). fastf1 needs an F1TV Access account.")
    ap.add_argument("--ref-mode", choices=["full", "early", "fe"], default="full",
                    help="full = v2 behaviour; early = causal reference from laps 2-8 only (no future information); "
                         "fe = unbiased tyre slopes (see src/features.py)")
    ap.add_argument("--refresh", action="store_true", help="re-download data")
    ap.add_argument("--demo-lap", type=int, default=20)
    ap.add_argument("--demo-tyre-life", type=float, default=14)
    ap.add_argument("--demo-compound", default="MEDIUM")
    ap.add_argument("--demo-new-compound", default="HARD")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    raw = (make_synthetic() if a.synthetic else
           build_dataset(a.rounds, a.year, csv_path=f"data/laps_{a.source}.csv", refresh=a.refresh,
                         source=a.source))
    df = prepare(raw, ref_mode=a.ref_mode)
    print(f"reference mode: {a.ref_mode}")
    reg = regression_set(df)
    reg_features = [f for f in C.REG_FEATURES if reg[f].notna().any()]
    pit_features = [f for f in C.PIT_FEATURES if df[f].notna().any()]
    dropped = sorted(set(C.REG_FEATURES + C.PIT_FEATURES) - set(reg_features) - set(pit_features))
    if dropped:
        print("features with no data in this source, skipped:", dropped)
    print(f"{df.RaceId.nunique()} races | {len(df)} laps | {len(reg)} clean laps used for regression")

    # 1) regression, leave-one-race-out
    models = get_models()
    res, summary = loro_regression(reg, reg_features, models, alpha=C.ALPHA)
    res.to_csv(out / "loro_regression_per_race.csv", index=False)
    summary.to_csv(out / "loro_regression_summary.csv")
    print("\nLeave-one-race-out lap-delta prediction (seconds; lower MAE is better)")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary)

    pv = res.pivot(index="test_race", columns="model", values="MAE")
    print("\nHead-to-head on held-out races (lower MAE wins):")
    for m in pv.columns:
        if m == "baseline_compound_mean":
            continue
        print(f"  {m}: beats baseline in {(pv[m] < pv['baseline_compound_mean']).sum()}/{len(pv)} races"
              + (f", beats linear in {(pv[m] < pv['linear']).sum()}/{len(pv)}" if m != "linear" else ""))

    # 2) pit-stop classifier
    pit = loro_pit_classifier(pit_set(df), pit_features)
    pit.to_csv(out / "loro_pit_classifier.csv", index=False)
    print("\nPit-stop prediction (average precision vs. base rate)")
    print(pit.round(3).to_string(index=False))

    # 3) interpretability
    best = (res[res.model != "baseline_compound_mean"].groupby("model")["MAE"].mean().idxmin())
    print(f"\nBest model by held-out MAE: {best}")
    imp = permutation_loro(reg, reg_features, models[best])
    imp.to_csv(out / "permutation_importance.csv")
    print("\nPermutation importance on held-out races:")
    print(imp.round(4))
    shap_summary(reg, reg_features, "DeltaS", str(out / "shap_summary.png"))

    # 4) pit-window simulation with the final model
    races = sorted(reg.RaceId.unique())
    cal_r = races[-1]
    final = clone(models[best]).fit(reg[reg.RaceId != cal_r][reg_features], reg[reg.RaceId != cal_r]["DeltaS"])
    cal = reg[reg.RaceId == cal_r]
    resid = (cal["DeltaS"] - final.predict(cal[reg_features])).to_numpy()
    losses = pit_loss_samples(df)["loss_s"].to_numpy()
    print(f"\nPit loss samples: n={len(losses)}, median {np.median(losses):.1f}s, "
          f"5-95% range {np.percentile(losses, 5):.1f}-{np.percentile(losses, 95):.1f}s")
    L = int(df["TotalLaps"].median())
    state = dict(lap=a.demo_lap, total_laps=L, tyre_life=a.demo_tyre_life,
                 compound=C.COMPOUNDS[a.demo_compound], fresh=0.0,
                 track_temp=float(reg["TrackTemp"].median()), air_temp=float(reg["AirTemp"].median()),
                 position=10.0)
    table, ref = pit_window_table(final, reg_features, state, C.COMPOUNDS[a.demo_new_compound],
                                  float(np.median(losses)))
    table.to_csv(out / "pit_window_demo.csv", index=False)
    print(f"\nPit-window demo: {a.demo_compound} tyres, {a.demo_tyre_life:.0f} laps old, lap {a.demo_lap} of {L}, "
          f"switching to {a.demo_new_compound}; median pit loss {np.median(losses):.1f}s")
    print(table.head(5).round(2).to_string(index=False))
    cc = C.COMPOUNDS[a.demo_compound]
    max_life = float(reg.loc[reg["CompoundCode"] == cc, "TyreLife"].quantile(0.95))
    last_age = a.demo_tyre_life + (L - a.demo_lap)
    if last_age > max_life:
        print(f"WARNING: 'no stop' needs {a.demo_compound} tyres up to {last_age:.0f} laps old, but 95% of "
              f"{a.demo_compound} laps in the data are <= {max_life:.0f}. Tree models cannot extrapolate, and old-tyre "
              f"laps only exist when the tyres were still fast enough to stay out (survivorship bias), so "
              f"degradation is understated. Treat 'vs_no_stop' as unreliable; compare pit laps with each other.")
    top = table["pit_lap"].head(3).tolist()
    mc, sims = monte_carlo(final, reg_features, state, C.COMPOUNDS[a.demo_new_compound], losses, resid,
                           top + [int(table["pit_lap"].iloc[-1])], n_sim=2000)
    print("\nMonte Carlo (total delta, seconds; approximate - see simulate.py for assumptions)")
    print(mc.round(2).to_string(index=False))
    print(f"P(pit lap {top[0]} beats pit lap {top[-1]}) ~ {prob_better(sims[top[0]], sims[top[-1]]):.2f} (independent draws)")

    plt.figure(figsize=(7, 4))
    t = table.sort_values("pit_lap")
    plt.plot(t["pit_lap"], t["vs_no_stop_s"])
    plt.axhline(0, ls="--", c="grey")
    plt.xlabel("Pit lap"); plt.ylabel("Expected total time vs. no stop (s)")
    plt.title("Pit window (lower is better)"); plt.tight_layout()
    plt.savefig(out / "pit_window_demo.png", dpi=150)


if __name__ == "__main__":
    main()
