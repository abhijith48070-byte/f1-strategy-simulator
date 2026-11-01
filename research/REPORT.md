# Technical report (draft): leakage-free evaluation and uncertainty for F1 tyre degradation and strategy

Data: 2024 season, OpenF1 laps/stints/weather/race-control, 23 usable races. Everything below is leave-one-race-out unless stated.
**Status key:** VERIFIED = run on real data and reported by the author; TO RUN = code exists and passes offline tests, real-data numbers not yet collected; NOT DONE = not implemented.

## Experiment 1: Do models generalise to unseen circuits? (partly VERIFIED)
- *Hypothesis:* lap-time structure (race progress, tyre age) transfers across circuits.
- *Within-driver skill* (lap time minus the driver's own race mean; no reference pace): race progress alone ≈ 0.39-0.40, + tyre age ≈ 0.48-0.50;
  tyre age helps in 18-19 of 22 held-out races (≈ +0.09, sign test p ≈ 0.004; races are not fully independent). Compound adds nothing; temperature hurts trees. VERIFIED.
- *Absolute lap delta, causal reference:* MAE linear 0.931, LightGBM 1.008, HistGB 1.053, compound-mean baseline 1.049; R² < 0 for all; trees ≈ baseline. VERIFIED.
- *Hierarchical Bayesian (NumPy Gibbs sampler, partial pooling by race × compound):* SOFT 0.092 ± 0.012, MEDIUM 0.064 ± 0.010, HARD 0.052 ± 0.008 s/lap;
  race-to-race sd ≈ 0.04 exceeds compound gaps; MEDIUM vs HARD not distinguishable. VERIFIED.
- *PyMC hierarchical model with driver/circuit/weather effects and predictive log-likelihood comparison:* NOT DONE.

## Experiment 2: Survivorship bias in degradation estimates (code ready: TO RUN)
`python run_survivorship.py --exclude 2024_R08`. Discrete-time pit hazard + stabilised IPW; compares naive vs weighted slopes with a cluster bootstrap over races.
Offline result (synthetic, known truth, strong informative stopping): naive SOFT slope 0.059 vs truth 0.09; IPW reduces the bias substantially but not fully, with an effective
sample fraction of 0.3-0.5 (variance cost). With no informative stopping IPW leaves estimates unchanged. Real teams use information we do not observe, so on real data this is a sensitivity analysis.

## Experiment 3: Calibrated uncertainty (code ready: TO RUN)
`python run_uncertainty.py --exclude 2024_R08`. Single-race conformal (v3: 80-84% coverage for nominal 90%) vs cross-race conformal, ACI and CQR; reports coverage, width,
interval score, CRPS (from a quantile grid) and conditional coverage by tyre age and compound. Calibration never uses the test race. On synthetic data cross-race methods reach ≈ 0.90.

## Experiment 4: Multi-horizon forecasting (code ready: TO RUN)
`python run_forecast.py --exclude 2024_R08`. Persistence vs linear vs gradient boosting at 5/10/15 laps. LSTM and Temporal Fusion Transformer: NOT DONE.

## Experiment 5: Uncertainty-aware strategy optimisation (NOT validated)
The strategy engine (enumeration + Monte Carlo with safety-car scenarios, Bayesian degradation) runs and is unit-tested, but has not been backtested against real team decisions;
example recommendations have 90% ranges such as -18 to +32 s vs the runner-up. Dynamic programming, MPC, opponent modelling, Pareto optimisation and PPO: NOT DONE.

## Limitations
Causal reference unstable in wet/red-flag races; no traffic, per-lap position or rival data; SC and VSC not separable; linear tyre model; survivorship correction assumes pit decisions depend
only on observed history; one season of data.
