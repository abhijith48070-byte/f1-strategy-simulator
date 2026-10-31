"""Strategy engine CLI: optimizer + safety-car scenarios + Bayesian degradation + Monte Carlo + undercut/overcut.

  python run_strategy.py --synthetic                         offline demo
  python run_strategy.py                                     uses data/laps_openf1.csv (run run_experiment.py first)
  python run_strategy.py --lap 22 --compound MEDIUM --tyre-life 22 --model bayes
  python run_strategy.py --rival-gap 2.5 --rival-pit-lap 26  adds the undercut/overcut analysis
  python run_strategy.py --backtest-lap 25 --driver VER      decision using only information available at lap 25
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from src import config as C
from src.data import build_dataset, make_synthetic
from src.engine import build_context, make_state, recommend, backtest_decision, describe_reactive
from src.planner import fmt_time, NAMES
from src.undercut import analyse
from src import plots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--csv", default="data/laps_openf1.csv")
    ap.add_argument("--model", choices=["bayes", "hist_gbr"], default="bayes")
    ap.add_argument("--race"); ap.add_argument("--driver")
    ap.add_argument("--lap", type=int, default=10)
    ap.add_argument("--compound", default="MEDIUM")
    ap.add_argument("--tyre-life", type=float, default=None, help="default: lap (started the race on this set)")
    ap.add_argument("--objective", choices=["mean", "p90"], default="mean")
    ap.add_argument("--n-sim", type=int, default=C.N_SIM)
    ap.add_argument("--rival-gap", type=float, help="seconds behind a rival; enables undercut/overcut analysis")
    ap.add_argument("--rival-pit-lap", type=int)
    ap.add_argument("--rival-traffic-cars", type=int, default=0)
    ap.add_argument("--backtest-lap", type=int)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)

    if a.synthetic:
        raw = make_synthetic(n_races=6, seed=1)
    elif Path(a.csv).exists():
        raw = pd.read_csv(a.csv)
    else:
        raise SystemExit(f"{a.csv} not found. Run `python run_experiment.py` first (downloads real races) or use --synthetic.")

    race = a.race or sorted(raw["RaceId"].unique())[-1]
    driver = a.driver or raw[raw["RaceId"] == race]["Driver"].iloc[0]

    if a.backtest_lap:
        bt = backtest_decision(raw, race, driver, a.backtest_lap, a.model, n_sim=min(a.n_sim, 2000))
        print(f"BACKTEST {race} {driver} at lap {a.backtest_lap} (trained on other races; only laps <= {a.backtest_lap} used)")
        print("  recommended:", bt["recommended"])
        print("  actually did:", bt["actual"])
        print(f"  model's estimate of recommended vs actual: {bt['model_gain_vs_actual_s']:+.1f} s "
              f"(a model-based estimate, not an observed counterfactual)")
        return

    ctx = build_context(raw)
    print(f"{len(ctx.races)} races | pit loss median {ctx.pit_loss:.1f} s (n={len(ctx.pit_losses)}) | "
          f"SC/VSC rates from {ctx.stats['source']}: SC {ctx.stats['p_sc']:.0%}, VSC {ctx.stats['p_vsc']:.0%} | "
          f"tyre-life limits {{{', '.join(f'{NAMES[c]}: {v:.0f}' for c, v in ctx.max_life.items())}}}")
    print("\nBayesian tyre degradation (s per lap of tyre age):")
    print(ctx.bayes.degradation_summary().to_string(index=False))

    tyre_life = a.tyre_life if a.tyre_life is not None else a.lap
    state = make_state(ctx, a.model, race, driver, a.lap, a.compound, tyre_life)
    rec = recommend(ctx, a.model, state, n_sim=a.n_sim, objective=a.objective)
    b = rec.best
    print(f"\n{race} / {driver}: lap {state['lap']} of {state['total_laps']}, {a.compound} tyres {tyre_life:.0f} laps old "
          f"(model: {a.model}, objective: {a.objective})")
    print(f"RECOMMENDED: {b['strategy']}   predicted finish {rec.summary.iloc[0]['mean']}")
    adv = rec.explanation["why"][0] if rec.explanation["why"] else ""
    print(adv)
    print("\nStrategy comparison (Monte Carlo, %d simulated races, SC/VSC included):" % a.n_sim)
    print(rec.summary[["strategy", "mean", "p90", "p_fastest", "mean_regret_s"]].round(3).to_string(index=False))
    print("\nSafety car scenarios (each assumed known now):")
    print(rec.scenarios[["scenario", "best_strategy", "first_pit_lap", "adaptation_gain_s"]].round(1).to_string(index=False))
    for name, r in rec.reactive:
        print("  " + describe_reactive(name, r))
    for k, title in (("why", "WHY"), ("changes", "WHAT WOULD CHANGE IT"), ("risks", "RISKS")):
        print(f"\n{title}")
        for line in rec.explanation[k]:
            print("  -", line)

    rec.summary.to_csv(out / "strategy_comparison.csv", index=False)
    rec.scenarios.to_csv(out / "scenario_table.csv", index=False)
    ctx.bayes.degradation_summary().to_csv(out / "bayes_degradation.csv", index=False)
    rec.table.drop(columns=["compounds", "pit_laps"]).head(200).to_csv(out / "all_strategies_top200.csv", index=False)
    for name, fn in (("strategy_comparison", plots.strategy_comparison), ("paired_distributions", plots.paired_distributions),
                     ("pit_window", plots.pit_window), ("scenarios", plots.scenarios)):
        fn(rec).savefig(out / f"{name}.png", dpi=140)
    plots.degradation(ctx.bayes).savefig(out / "bayes_degradation.png", dpi=140)

    if a.rival_gap is not None:
        c0 = state["compound"]
        new = b["compounds"][1] if len(b["compounds"]) > 1 else c0
        pb = a.rival_pit_lap or int(b["pit_laps"][0]) + 2 if len(b["pit_laps"]) else state["lap"] + 3
        t, s = analyse(ctx.models[a.model], state, dict(compound=c0, tyre_life=state["tyre_life"], new_compound=new),
                       dict(compound=c0, tyre_life=state["tyre_life"], new_compound=new), a.rival_gap, ctx.pit_losses,
                       pb, traffic_cars=a.rival_traffic_cars)
        print("\nUNDERCUT / OVERCUT vs the car ahead")
        print(t[["a_pit_lap", "kind", "net_gain_s", "own_race_cost_s", "margin_s", "p_success"]].round(2).to_string(index=False))
        for line in s["text"]:
            print(" ", line)
        t.to_csv(out / "undercut_overcut.csv", index=False)
        plots.undercut(t, pb).savefig(out / "undercut_overcut.png", dpi=140)
    print(f"\nfiles written to {out}/")


if __name__ == "__main__":
    main()
