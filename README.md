# F1 Strategy Simulator v3: an uncertainty-aware race-strategy optimization engine

Predicts lap-time loss from tyre age, compound and race progress, then **recommends a pit strategy automatically**,
with probabilistic outcomes, safety-car awareness, undercut/overcut analysis and plain-language explanations.

> **Status:** everything below is tested on synthetic races with known physics (`python -m tests.test_strategy`).
> It has **not** been run on real OpenF1 data in this environment (no network), and the Streamlit UI has been
> exercised only through a stub (`tests.test_app_smoke`), not in a browser. Real-data results: *add after running*.
> This is a replay/what-if tool. It is **not** a live race system: there is no live data pipeline.

## What v3 adds
| Feature | Where | What it does |
|---|---|---|
| Dynamic pit-stop optimizer | `src/planner.py`, `src/optimizer.py` | Enumerates every legal 0/1/2-stop strategy (compound sequence + pit laps) from the current state: min stint, two-compound rule, tyre-life limits taken from the data. Ranks by mean time, then re-ranks a shortlist with Monte Carlo (mean or 90th-percentile objective). |
| Safety car / VSC simulator | `src/scenarios.py`, `src/optimizer.py` | No SC, SC on laps 15/25/40, VSC. Cheaper stops, slower laps and reduced tyre wear under neutralisation. `scenario_table` = best plan per scenario and what re-planning is worth; `reactive_decision` = "SC just came out on lap 20, pit now?" using only the state at that lap. |
| Bayesian tyre degradation | `src/bayes.py` | Hierarchical regression (partial pooling across races, or drivers with `group_col="Driver"`), pure-NumPy Gibbs sampler. Reports degradation as mean ± posterior sd plus race-to-race spread; forecasts carry parameter uncertainty. |
| Monte Carlo outcomes | `src/montecarlo.py` | Thousands of simulated races with common random numbers: posterior/residual uncertainty, per-lap noise, weather shift (tree model), random SC/VSC (rate estimated from the data), pit-stop duration variation. Reports mean, 10th/90th percentile, **P(fastest)**, regret and paired advantages. |
| Undercut / overcut | `src/undercut.py` | Scans our stop lap against a rival's: fresh-tyre gain, both pit losses, old-tyre cost, traffic, gap; P(success) by simulation; cost to our own race. |
| Strategy dashboard | `app.py` | `streamlit run app.py` |
| Strategy Intelligence Engine | `src/strategy_explain.py` | Why this strategy, what would change it (pit-loss and degradation thresholds, SC scenarios), what it risks. Every sentence is computed from model output. |
| Leakage fixes | `src/features.py`, `src/engine.py` | `prepare(ref_mode="early")` builds targets with no future laps; `state_from_race` / `backtest_decision` use only information available at the decision lap (tested). |

### Two findings worth knowing
1. **The v2 target biases tyre degradation toward zero.** `DeltaS = lap - driver's 15th-percentile lap` depends on the driver's own stint mix.
   On synthetic data with known physics (HARD 0.03 s/lap) the v2 target gives 0.002; `ref_mode="fe"` and `"early"` give 0.030 / 0.031.
   The Bayesian model avoids it by using a within-driver-race estimator on raw lap times.
2. **v2 held-out numbers were optimistic.** With the leak-free `early` reference the synthetic held-out MAE of the boosted model rises from 0.215 to 0.264 s.
   Report the `early` numbers (`python run_experiment.py --ref-mode early`).

## Run
```bash
pip install -r requirements.txt
python run_experiment.py --ref-mode early          # real data (downloads from OpenF1, caches to data/), leak-free evaluation
python run_strategy.py --synthetic                 # full strategy report offline
python run_strategy.py --lap 22 --compound MEDIUM --tyre-life 22 --rival-gap 2.5 --rival-pit-lap 26
python run_strategy.py --backtest-lap 25 --driver VER   # decision from information available at lap 25 only
streamlit run app.py
python -m tests.test_strategy && python -m tests.test_pipeline && python -m tests.test_openf1_parse && python -m tests.test_app_smoke
```

## Strategy-engine assumptions (edit `src/config.py`)
These are **assumptions, not fitted values**: SC/VSC lap-time factors (1.40 / 1.30), pit-loss factors under SC/VSC (0.50 / 0.65),
tyre wear on neutralised laps (0.3 / 0.4 of a lap), default SC/VSC durations, traffic cost per car (0.6 s), weather shift sd (3 C).
Only the SC/VSC *frequency* and start/duration distribution are estimated from data, and the OpenF1 loader cannot tell SC from VSC.

## Limitations of the strategy engine
- Monte Carlo strategies are committed plans (they do not react to a random SC); the value of reacting is shown separately in the scenario table.
- At most one neutralisation per simulated race; pit-lane closure, field compression and restarts are not modelled.
- The model sees the model's lap times only: no traffic, rivals' strategies, rain, tyre warm-up or driver-specific skill (unless you group by driver).
- The Bayesian model is linear in tyre age (no cliff); stints longer than anything observed are excluded. With a handful of races the race-to-race spread depends on the prior.
- The tree model cannot price counterfactual stints (a hard tyre early in the race); use it for evaluation, not for planning. The dashboard warns when it is selected.
- Absolute finish times depend on an estimated reference pace; rankings do not. P(fastest) is among the compared strategies, not a race-win probability.
- The backtest's "gain vs actual" is the model's own estimate, not an observed counterfactual.

## v2 background
## What changed from v1
| Issue in v1 | Fix in v2 |
|---|---|
| Random 70/30 lap split, so adjacent laps of the same stint leak between train and test | **Leave-one-race-out**: the test race is never seen in training |
| Pit label built with `shift(-1)` across the whole table (crosses driver boundaries); `AdjustedLapTime` used next-lap and pit-time data, which leaks the label | Label is the in-lap flag; features use only information known at the start of the lap; everything is grouped by (race, driver) |
| Pit-window code compared predictions from *unrelated test rows* of two models | Counterfactual simulation: same driver, same lap, stay out vs pit at lap *p* |
| One race (Monaco 2024) | Multiple circuits (`src/config.py`) |
| No baselines, no intervals | Compound-mean baseline, linear model, gradient boosting (+LightGBM if installed), 90% conformal intervals |
| Raw feature importance | Permutation importance on held-out races (+ SHAP if installed) |

## Data source
Default is **OpenF1** (free public API, no account or key, 2023 onward, limited to 30 requests/minute so the
first download takes a few minutes). FastF1 3.9+ now requires an F1TV Access account; use it only with
`--source fastf1` if you have one. OpenF1 has no per-lap position, so the `Position` feature is skipped.

## Run
```bash
pip install -r requirements.txt
python run_experiment.py                 # downloads races from OpenF1 (needs internet), caches to data/
python run_experiment.py --synthetic     # offline self-test
python -m tests.test_pipeline            # unit tests
python -m tests.test_openf1_parse        # parser test
```
Outputs go to `results/`: per-race metrics, summary table, permutation importance, `shap_summary.png`,
pit-window table and plot.

## Method
1. **Cleaning** (`src/features.py`): green-flag laps only, in/out laps removed, no rain, laps >7% slower than
   the driver's median dropped.
2. **Target**: `DeltaS` = lap time − the driver's reference pace in that race (15th percentile of clean laps).
   This removes circuit and driver baselines so races are comparable.
3. **Models**: compound-mean baseline, linear, HistGradientBoosting (+ LightGBM). Features: tyre life, lap fraction
   (fuel proxy), compound, fresh-tyre flag, track/air temperature, position.
4. **Uncertainty**: split conformal prediction. For each held-out race one other race is the calibration set.
   Reported: empirical coverage (target 90%) and interval width.
5. **Pit window** (`src/simulate.py`): total predicted time for "pit at lap p" vs "stay out", plus Monte Carlo using
   bootstrapped residuals and observed pit-loss variation.

## Results
Fill in from `results/loro_regression_summary.csv` after running on real data:

| Model | Held-out MAE (s) | R² | 90% interval coverage |
|---|---|---|---|
| Compound-mean baseline | | | |
| Linear | | | |
| HistGradientBoosting | | | |

## Limitations (state these in your write-up)
- **Reference pace is known** (driver's 15th-percentile lap in the same race). In real use it would be estimated
  from early laps or practice.
- `LapFrac` mixes fuel burn and track evolution; tyre wear is not cleanly isolated.
- Tree models cannot extrapolate beyond tyre ages seen in training; "no stop" estimates are unreliable
  when they exceed it (the runner prints a warning).
- No traffic, safety cars/VSC, the two-compound rule, or undercut/overcut effects.
- Lap-to-lap model errors are sampled independently in Monte Carlo, which understates correlation.
- Pit loss is estimated from data and includes small tyre-age effects.

- **Survivorship bias:** laps on very old tyres only exist when the tyres were still fast enough to stay out
  (drivers pit when degradation bites), so degradation at high tyre age is understated and "no stop" looks
  better than it is.
- Pit-loss samples are restricted to 8-60 s; red-flag and glitch laps otherwise produce losses of hundreds of seconds.
- The demo state (lap 20 of the median race length) is illustrative, not a real race situation.

## Next steps
Survival model for pit timing; driver-level pooling on real data; rival-aware strategy (game-theoretic); closing the pit lane under SC.
