F1 Strategy Simulator (v4)

An uncertainty-aware Formula 1 race strategy optimization system featuring leakage-free evaluation, hierarchical Bayesian tyre degradation modelling, survivorship-bias correction, calibrated prediction intervals and Monte Carlo pit strategy simulation.

Built using 2024 Formula 1 race data from OpenF1 (no account required).

See STATUS.md for implementation status and research/REPORT.md for experimental results and verification details.

Run

python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python run_experiment.py
python run_ablation.py --exclude 2024_R08
python run_survivorship.py --exclude 2024_R08
python run_uncertainty.py --exclude 2024_R08
python run_forecast.py --exclude 2024_R08
python run_strategy.py --lap 22 --compound MEDIUM --tyre-life 22
streamlit run app.py
python -m tests.test_survival && python -m tests.test_uncertainty && python -m tests.test_forecast

Add --synthetic to experiment scripts to run offline experiments using synthetic races with known underlying physics. Synthetic results must not be reported as real-world findings.

Research Results

Dataset: 2024 Formula 1 season, 23 usable races, leave-one-race-out evaluation.

1. Within-Driver Tyre Degradation Analysis

Race progress alone explains approximately 40% of within-race lap-time variation.

Adding tyre age increases explained variation to approximately 48–50%.

* Tyre age improves results in 18–19 of 22 held-out races.
* Improvement: approximately +0.09.
* Sign-test p-value: approximately 0.004.
* Compound information provides no measurable additional improvement.
* Temperature features negatively affect tree-based models.

Races are not fully independent, so statistical significance should be interpreted cautiously.

2. Absolute Lap-Time Prediction

Causal reference: laps 2–8.

Model	Held-out MAE (s)	R²	90% Interval Coverage
Compound-mean baseline	1.049	-0.402	0.871
Linear Regression	0.931	-0.155	0.836
LightGBM	1.008	-0.398	0.806
HistGradientBoosting	1.053	-0.548	0.805

Race-to-race MAE standard deviation is approximately 0.4–0.6 seconds. Most differences between models are therefore within the observed variability.

Cross-race conformal prediction, Adaptive Conformal Inference (ACI) and Conformalized Quantile Regression (CQR) are investigated separately.

3. Hierarchical Bayesian Tyre Degradation

Estimated tyre degradation in seconds per lap of tyre age:

Compound	Estimated Degradation
SOFT	0.092 ± 0.012
MEDIUM	0.064 ± 0.010
HARD	0.052 ± 0.008

Race-to-race standard deviation is approximately 0.04 seconds per lap.

The differences between compounds are therefore relatively small compared with race-level variation. MEDIUM versus HARD degradation is not statistically distinguishable.

4. Survivorship-Bias Analysis

Dataset: 23 races.

Comparison of naive and inverse-probability-weighted (IPW) degradation estimates:

* Estimated slope differences: +0.001 to +0.004 seconds per lap.
* All 90% cluster-bootstrap confidence intervals include zero.
* Effective sample fraction: 0.94.

No detectable survivorship bias was identified under the observed-history and sequential-ignorability assumptions.

5. Prediction Interval Calibration

Dataset: 21 races, excluding Monaco.

Linear model with causal reference.

Method	Empirical Coverage	Interval Width
Single-race Conformal	79.5%	—
Cross-race Conformal	89.6%	3.1 s
Cross-race + ACI	90.5%	2.9 s

ACI improves empirical coverage while producing narrower prediction intervals.

Interval score improves from 4.45 to 3.89.

Coverage across tyre-age groups remains approximately 0.88–0.91 with ACI.

6. Multi-Horizon Forecasting

Forecast horizons: 5, 10 and 15 laps.

Model	5-Lap MAE	10-Lap MAE	15-Lap MAE
Persistence	0.67 s	0.81 s	1.00 s
Linear Regression	0.69 s	0.84 s	1.04 s

Linear regression and gradient boosting do not outperform the persistence baseline.

This is an important negative result: additional model complexity does not necessarily improve forecasting performance.

LSTM and Temporal Fusion Transformer models have not been evaluated.

Strategy Engine

The strategy engine provides:

* Monte Carlo pit-stop strategy comparisons.
* Safety car scenario simulation.
* Virtual safety car scenario simulation.
* Tyre degradation estimates.
* Strategy comparison and recommendations.
* Uncertainty-aware race-time projections.

Run:

python run_strategy.py --lap 22 --compound MEDIUM --tyre-life 22

Launch the interactive dashboard:

streamlit run app.py

Important: Strategy recommendations are currently modelled what-if scenarios, not validated real-world racing advice. The engine has not been comprehensively backtested against actual race decisions.

Technology Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* LightGBM
* SciPy
* PyMC-compatible Bayesian modelling foundations
* OpenF1 API
* Streamlit
* Matplotlib

Limitations

* The causal reference is unstable in wet and red-flag races.
* Monaco 2024 is excluded from certain analyses.
* The dataset lacks traffic, per-lap position and comprehensive rival information.
* Safety Car and Virtual Safety Car scenarios cannot currently be distinguished using the available model inputs.
* The tyre degradation model assumes linear degradation.
* Survivorship-bias correction relies on a sequential-ignorability assumption.
* Strategy recommendations require further historical validation.

Future Development

Potential research extensions include:

* Hierarchical Bayesian modelling using richer driver and circuit effects.
* Nonlinear tyre degradation modelling.
* Improved survivorship-bias correction.
* Advanced temporal forecasting.
* Opponent-aware strategy optimization.
* Dynamic programming and Model Predictive Control.
* Reinforcement learning for sequential race decisions.

Project Objective

The objective is to investigate how machine learning, Bayesian statistics, probabilistic forecasting and stochastic optimization can improve race strategy decisions under uncertainty.

The project emphasizes rigorous evaluation, reproducibility, uncertainty quantification and honest reporting of both positive and negative experimental results.
