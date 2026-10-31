"""Hierarchical Bayesian tyre-degradation model (pure NumPy Gibbs sampler, no PyMC needed).

    LapTime_i = base_u + a[c_i] + b[c_i] * TyreAge_i + f * LapNumber_i + eps_i,      eps_i ~ N(0, sigma^2)
    theta_g   = [a_SOFT, a_HARD, b_SOFT, b_MEDIUM, b_HARD, f]  ~  N(mu, diag(tau^2))     (partial pooling)

* base_u is a free level for every (race, driver) "unit". It is removed by demeaning within the unit
  (within estimator), so no reference-pace definition can bias the tyre slopes. Why this matters: the v2
  target DeltaS = lap - driver's 15th-percentile lap makes degradation look smaller than it is
  (on synthetic data with known physics: HARD 0.002 vs the true 0.03 s/lap). See tests/test_strategy.py.
* a_MEDIUM = 0: compound pace offsets are relative to the medium tyre.
* b[c] = degradation in s per lap of tyre age; f = fuel effect in s per lap of race progress (negative).
* group g = race (default) or driver (group_col="Driver"). Partial pooling shrinks sparse groups to the
  population mean. A NEW group is forecast with theta_new ~ N(mu, diag(tau^2)): parameter uncertainty and
  group-to-group variability are both in the forecast, so degradation is "0.05 +/- 0.01", not "0.05".

Priors: mu flat; tau_j^2 ~ InvGamma(2, s_j^2) (weak, centred on scale s_j); sigma^2 ~ InvGamma(1, 0.01).
With a handful of races tau is prior-sensitive; say so in the write-up.

Limitations: linear in tyre age (no cliff), so long stints are optimistic; the optimizer also refuses
stints longer than anything observed. Weather/temperature enters only through the group variance.
"""
import numpy as np
import pandas as pd

COMP_NAMES = ["SOFT", "MEDIUM", "HARD"]
PARAM_NAMES = ["a_SOFT", "a_HARD", "b_SOFT", "b_MEDIUM", "b_HARD", "f_per_lap"]
TAU_PRIOR_SCALE = np.array([0.5, 0.5, 0.03, 0.03, 0.03, 0.01])   # prior scale of between-group sd
P = 6


def design_matrix(compound_code, tyre_age, lap_number):
    """X rows for theta = [a_S, a_H, b_S, b_M, b_H, f]. compound_code: 0=SOFT 1=MEDIUM 2=HARD."""
    c = np.asarray(compound_code, dtype=int)
    n = len(c)
    X = np.zeros((n, P))
    idx = np.arange(n)
    X[c == 0, 0] = 1.0
    X[c == 2, 1] = 1.0
    X[idx, 2 + c] = np.asarray(tyre_age, dtype=float)
    X[:, 5] = np.asarray(lap_number, dtype=float)
    return X


class HierarchicalTyreModel:
    def __init__(self, n_iter=2500, burn=800, seed=0):
        self.n_iter, self.burn, self.seed = n_iter, burn, seed

    def fit(self, df, group_col="RaceId", y_col="LapTimeS", unit_cols=("RaceId", "Driver")):
        """df: clean green-flag laps with CompoundCode, TyreLife, LapNumber, LapTimeS, Driver, RaceId."""
        d = df.dropna(subset=["CompoundCode", "TyreLife", "LapNumber", y_col, group_col, *unit_cols]).copy()
        self.group_col = group_col
        self.groups_ = sorted(d[group_col].unique())
        G = len(self.groups_)
        if G < 2:
            raise ValueError("need at least 2 groups for a hierarchical model")
        X = design_matrix(d["CompoundCode"], d["TyreLife"], d["LapNumber"])
        y = d[y_col].to_numpy(float)
        unit = d[list(unit_cols)].astype(str).agg("|".join, axis=1).to_numpy()
        # within transformation: subtract each unit's mean (removes the free per-driver-race level)
        Xd = pd.DataFrame(X).groupby(unit).transform("mean").to_numpy()
        yd = pd.Series(y).groupby(unit).transform("mean").to_numpy()
        X, y = X - Xd, y - yd
        self._unit_dof = len(set(unit))
        gcol = d[group_col].to_numpy()
        XtX, Xty, yty, n_g = [], [], [], []
        for g in self.groups_:
            m = gcol == g
            XtX.append(X[m].T @ X[m]); Xty.append(X[m].T @ y[m]); yty.append(float(y[m] @ y[m])); n_g.append(int(m.sum()))
        n_eff = sum(n_g) - self._unit_dof            # degrees of freedom lost to the unit means
        rng = np.random.default_rng(self.seed)

        theta = np.tile(np.linalg.solve(sum(XtX) + 1e-6 * np.eye(P), sum(Xty)), (G, 1))
        mu, tau2, sig2 = theta.mean(axis=0), TAU_PRIOR_SCALE ** 2, 0.1
        keep = self.n_iter - self.burn
        self.mu_, self.tau_ = np.zeros((keep, P)), np.zeros((keep, P))
        self.sigma_, self.theta_ = np.zeros(keep), np.zeros((keep, G, P))
        for it in range(self.n_iter):
            prior_prec = 1.0 / tau2
            ssr = 0.0
            for gi in range(G):
                cov = np.linalg.inv(XtX[gi] / sig2 + np.diag(prior_prec))
                cov = (cov + cov.T) / 2
                theta[gi] = cov @ (Xty[gi] / sig2 + mu * prior_prec) + np.linalg.cholesky(cov) @ rng.standard_normal(P)
                th = theta[gi]
                ssr += yty[gi] - 2 * th @ Xty[gi] + th @ XtX[gi] @ th
            mu = theta.mean(axis=0) + np.sqrt(tau2 / G) * rng.standard_normal(P)
            tau2 = (TAU_PRIOR_SCALE ** 2 + 0.5 * ((theta - mu) ** 2).sum(axis=0)) / rng.gamma(2 + G / 2, 1.0, size=P)
            sig2 = (0.01 + 0.5 * max(ssr, 1e-9)) / rng.gamma(1 + max(n_eff, 1) / 2, 1.0)
            if it >= self.burn:
                k = it - self.burn
                self.mu_[k], self.tau_[k], self.sigma_[k], self.theta_[k] = mu, np.sqrt(tau2), np.sqrt(sig2), theta
        self.max_age_ = d.groupby("CompoundCode")["TyreLife"].max().to_dict()
        return self

    # ------------------------------------------------------------------ summaries
    def degradation_summary(self):
        """Per compound: population-mean degradation (s/lap) +/- posterior sd, the race-to-race sd (tau) and the
        90% predictive interval for a NEW group."""
        rng = np.random.default_rng(1)
        rows = []
        for c, name in enumerate(COMP_NAMES):
            mu_b, tau_b = self.mu_[:, 2 + c], self.tau_[:, 2 + c]
            new = mu_b + tau_b * rng.standard_normal(len(mu_b))
            rows.append(dict(compound=name, deg_s_per_lap=mu_b.mean(), posterior_sd=mu_b.std(),
                             between_group_sd=tau_b.mean(), new_group_p05=np.percentile(new, 5),
                             new_group_p95=np.percentile(new, 95)))
        return pd.DataFrame(rows).round(4)

    def group_degradation(self):
        """Shrunken per-group degradation (posterior mean +/- sd), one row per group x compound."""
        rows = [dict(group=g, compound=name, deg_s_per_lap=self.theta_[:, gi, 2 + c].mean(),
                     sd=self.theta_[:, gi, 2 + c].std())
                for gi, g in enumerate(self.groups_) for c, name in enumerate(COMP_NAMES)]
        return pd.DataFrame(rows).round(4)

    # ------------------------------------------------------------------ prediction
    def mean_theta(self):
        return self.mu_.mean(axis=0)

    def draw_theta(self, n, rng, group=None):
        """n coefficient draws for a new group (default) or a known one, plus matching residual sd."""
        k = rng.integers(0, len(self.sigma_), size=n)
        if group is not None and group in self.groups_:
            return self.theta_[k, self.groups_.index(group), :], self.sigma_[k]
        return self.mu_[k] + self.tau_[k] * rng.standard_normal((n, P)), self.sigma_[k]

    def predict_mean(self, compound_code, tyre_age, lap_number):
        """Lap-time change vs a fresh MEDIUM tyre at lap 0 (no level). Add `base_pace` for seconds."""
        return design_matrix(compound_code, tyre_age, lap_number) @ self.mean_theta()

    def base_pace(self, laps):
        """Level of a driver in a race, estimated ONLY from the laps passed in (pass laps up to the current lap):
        mean(lap time - model prediction)."""
        x = laps.dropna(subset=["CompoundCode", "TyreLife", "LapNumber", "LapTimeS"])
        return float(np.mean(x["LapTimeS"].to_numpy() - self.predict_mean(x["CompoundCode"], x["TyreLife"], x["LapNumber"])))
