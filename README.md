# F1 Strategy Simulator v2: held-out-race evaluation, uncertainty and pit-window simulation

Predicts how much slower or faster a driver's lap is than their own reference pace, from tyre age,
compound, race progress and weather, and uses that to compare pit strategies with uncertainty.

> **Status:** pipeline tested on synthetic races with known physics (`python -m tests.test_pipeline`).
> Real-data results: *add your table here after running on FastF1 data* (see "Results").

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
Safety-car randomness in the simulator, strategy optimiser over compound sequences, survival model for pit timing.
