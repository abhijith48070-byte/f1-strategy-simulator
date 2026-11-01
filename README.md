# F1 Strategy Simulator

Leave-one-race-out evaluation, Bayesian tyre-degradation modelling and Monte Carlo pit-strategy comparison on 2024 F1 race data (OpenF1, no account needed).

## Results (2024 season, 23 usable races, leave-one-race-out)

**1. Within-driver skill (the most reliable result).** Target = lap time minus the driver's own race mean; no reference pace is used.
Race progress alone explains about 40% of within-race variation; adding tyre age raises that to about 48-50%. Tyre age helps in 18-19 of
22 held-out races (about +0.09 skill, sign test p of about 0.004; races are not fully independent). Compound adds nothing, and temperature
features hurt tree models. Full output: `results/ablation.txt`.

**2. Absolute lap-time delta (causal reference = laps 2-8).** Harder, and not reliably predictable for unseen races:

| Model | Held-out MAE (s) | R² | 90% interval coverage |
|---|---|---|---|
| Compound-mean baseline | 1.049 | -0.402 | 0.871 |
| Linear | 0.931 | -0.155 | 0.836 |
| LightGBM | 1.008 | -0.398 | 0.806 |
| HistGradientBoosting | 1.053 | -0.548 | 0.805 |

22 held-out races; the race-to-race sd of MAE is 0.4-0.6 s, so most differences between models are within noise. Linear beats the baseline
in 17 of 22 races; trees roughly tie it. Nominal 90% conformal intervals cover only 80-84%.

**3. Bayesian tyre degradation** (hierarchical model, s per lap of tyre age, mean ± posterior sd): SOFT 0.092 ± 0.012, MEDIUM 0.064 ± 0.010,
HARD 0.052 ± 0.008. The ordering is plausible, but the race-to-race sd (about 0.04) is larger than the gaps between compounds, and MEDIUM vs HARD is not distinguishable.

**4. Strategy engine** (`run_strategy.py`, `app.py`): Monte Carlo comparison of pit strategies including safety-car scenarios. It gives a modelled
what-if, not validated advice: it has not been backtested against real team decisions, and its uncertainty is wide (e.g. a 90% range of -18 to +32 s against the
runner-up in the example run).

`results/leakage_same_laps.csv` compares a whole-race reference pace with the causal reference on identical laps.

## Run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_experiment.py --ref-mode early      # downloads 2024 races from OpenF1 (slow, rate-limited), caches in data/
python run_ablation.py --exclude 2024_R08      # within-driver ablation
python run_leakage_check.py                    # reference-pace comparison
python run_strategy.py --lap 22 --compound MEDIUM --tyre-life 22
streamlit run app.py                           # local dashboard
python -m tests.test_strategy && python -m tests.test_pipeline && python -m tests.test_openf1_parse
```

## Layout
`src/` data loading, features, models, evaluation, Bayesian model, optimizer, Monte Carlo, safety-car scenarios. `tests/` offline tests on synthetic races with known
physics. `legacy/` the original v1 script (random split and label leakage, fixed in later versions).

## Limitations
- The causal reference is unstable in wet and red-flag races. Monaco 2024 is excluded from the ablation.
- No traffic, per-lap position or rival data; safety car and VSC cannot be separated; the tyre model is linear.
- Old-tyre laps only exist when tyres were still fast enough to stay out (survivorship bias), so degradation at high tyre age is understated.
- The pit-window demo in `run_experiment.py` uses tree models and is not valid under the causal target; use `run_strategy.py`.
- Strategy-engine assumptions are in `src/config.py`.

## Development notes
Built with AI assistance (Claude). The evaluation design and corrections (leave-one-race-out, leakage checks, within-driver evaluation) are documented in the commit history.
