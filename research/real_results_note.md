# F1 Strategy Simulator (v4)

Leakage-free evaluation, hierarchical Bayesian tyre degradation, survivorship-bias correction, calibrated prediction intervals and Monte Carlo pit strategy on 2024 F1 data (OpenF1, no account needed).
See `STATUS.md` for exactly what is implemented and what is not, and `research/REPORT.md` for results and their verification status.

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python run_experiment.py                      # leave-one-race-out baselines (causal reference by default)
python run_ablation.py --exclude 2024_R08     # within-driver ablation
python run_survivorship.py --exclude 2024_R08 # Experiment 2
python run_uncertainty.py --exclude 2024_R08  # Experiment 3 (add --no-cqr for speed)
python run_forecast.py --exclude 2024_R08     # Experiment 4
python run_strategy.py --lap 22 --compound MEDIUM --tyre-life 22
streamlit run app.py
python -m tests.test_survival && python -m tests.test_uncertainty && python -m tests.test_forecast  # offline tests
```
Add `--synthetic` to the experiment scripts for an offline run on fake races with known physics (never report these numbers as results).

## Built with AI assistance (Claude)
Evaluation design, corrections and verification history are in the commit log and `research/REPORT.md`.


## Results (2024 season, 23 usable races, leave-one-race-out; details and verification status in `research/REPORT.md`)

**Within-driver skill** (lap time minus the driver's own race mean; no reference pace): race progress alone explains about 40% of within-race variation; adding tyre age raises it to
about 48-50%, and helps in 18-19 of 22 held-out races (about +0.09, sign test p of about 0.004; races are not fully independent). Compound adds nothing; temperature features hurt trees.

**Absolute lap-time delta, causal reference (laps 2-8):**

| Model | Held-out MAE (s) | R² | 90% interval coverage (single-race conformal) |
|---|---|---|---|
| Compound-mean baseline | 1.049 | -0.402 | 0.871 |
| Linear | 0.931 | -0.155 | 0.836 |
| LightGBM | 1.008 | -0.398 | 0.806 |
| HistGradientBoosting | 1.053 | -0.548 | 0.805 |

The race-to-race sd of MAE is 0.4-0.6 s, so most differences between models are within noise. `run_uncertainty.py` tests cross-race conformal, ACI and CQR against this under-coverage.

**Hierarchical Bayesian degradation** (s per lap of tyre age): SOFT 0.092 ± 0.012, MEDIUM 0.064 ± 0.010, HARD 0.052 ± 0.008. Race-to-race sd (about 0.04) exceeds the compound gaps; MEDIUM vs HARD is not distinguishable.

**Strategy engine** (`run_strategy.py`, `app.py`): Monte Carlo comparison of pit strategies with safety-car scenarios. A modelled what-if, not validated advice; wide uncertainty and no backtest against real decisions.

## Layout
`src/` data loading, features, models, evaluation, Bayesian model, survival/IPW, uncertainty, forecasting, optimizer, Monte Carlo, scenarios. `tests/` offline tests on synthetic races with known physics.
Strategy-engine assumptions live in `src/config.py`.

## Limitations
- The causal reference is unstable in wet and red-flag races; Monaco 2024 is excluded from some analyses.
- No traffic, per-lap position or rival data; safety car and VSC cannot be separated; the tyre model is linear.
- Old-tyre laps only exist when tyres were still fast enough to stay out (survivorship bias); `run_survivorship.py` tests this under a sequential-ignorability assumption.
- Not implemented: PyMC model, LSTM/TFT, dynamic programming/MPC/opponent models, PPO, MLflow/DVC/FastAPI (see `STATUS.md`).

## Real-data results for the v4 experiments (2024 season; Monaco excluded)
- **Survivorship (23 races):** naive vs IPW slopes differ by +0.001 to +0.004 s/lap, all 90% cluster-bootstrap intervals include 0 (effective sample fraction 0.94). No detectable bias from informative pit decisions under the observed-history assumption.
- **Interval calibration (21 races, linear model, causal reference):** single-race conformal covers 79.5% for nominal 90%; cross-race conformal 89.6%; cross-race + adaptive conformal (ACI) 90.5% (sd 0.02) with narrower intervals (width 2.9 s vs 3.1 s; interval score 3.89 vs 4.45). Coverage by tyre age stays 0.88-0.91 with ACI.
- **Forecasting (5/10/15 laps):** linear and gradient boosting do not beat persistence (linear MAE 0.69/0.84/1.04 s vs 0.67/0.81/1.00 s). A negative result; LSTM/TFT were not tried.
