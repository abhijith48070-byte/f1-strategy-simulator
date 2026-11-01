"""Calibrated prediction intervals across races.

v3 calibrated each held-out race with ONE other race, which is noisy and under-covered (80-84% for a nominal 90%).
Here the calibration set is built from several races, always excluding the test race:
  * cross-race conformal: K-fold over TRAINING races. Each fold's model never saw its own races, and its residuals on them
    form the calibration pool. The final model is trained on all training races. Nothing from the test race is used.
  * adaptive conformal inference (ACI, Gibbs & Candes 2021): inside the test race, alpha is nudged after every lap so that
    long-run coverage tracks the target even when the race is noisier than the calibration races.
  * conformalised quantile regression (CQR, Romano et al. 2019): quantile models + a conformal correction.
Metrics: coverage, width, interval score (Winkler), pinball loss, CRPS approximated from a quantile grid, and coverage by
tyre age and compound (conditional coverage). Nothing is tuned on the test race.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TAUS = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


def point_model(kind="linear"):
    if kind == "linear":
        return make_pipeline(StandardScaler(), LinearRegression())
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.06, max_iter=150, random_state=0)


def quantile_model(tau):
    return HistGradientBoostingRegressor(loss="quantile", quantile=float(tau), max_depth=3, learning_rate=0.06,
                                         max_iter=120, random_state=0)


def conformal_q(scores, alpha):
    s = np.sort(np.asarray(scores))
    k = min(int(np.ceil((len(s) + 1) * (1 - alpha))), len(s))
    return float(s[k - 1])


def pinball(y, q, tau):
    d = np.asarray(y) - np.asarray(q)
    return float(np.mean(np.maximum(tau * d, (tau - 1) * d)))


def crps_from_quantiles(y, Q, taus=TAUS):
    """CRPS ~ 2 * mean pinball over a quantile grid (Gneiting & Ranjan). Q has one column per tau."""
    return float(2 * np.mean([pinball(y, Q[:, j], t) for j, t in enumerate(taus)]))


def interval_score(y, lo, hi, alpha):
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)))


def _race_folds(races, k):
    races = list(races)
    return [races[i::k] for i in range(min(k, len(races)))]


def cross_race_residuals(train, features, target, kind, k=5):
    """Out-of-race residuals (signed) and the rows they belong to, from K-fold over training races."""
    parts = []
    for fold in _race_folds(sorted(train["RaceId"].unique()), k):
        tr, ca = train[~train["RaceId"].isin(fold)], train[train["RaceId"].isin(fold)]
        m = clone(point_model(kind)).fit(tr[features], tr[target])
        parts.append(ca.assign(_res=ca[target].to_numpy() - m.predict(ca[features])))
    return pd.concat(parts)


def aci_bounds(pred, y, pool_abs, alpha=0.1, gamma=0.02):
    """Sequential intervals pred +- q_t. alpha_t <- alpha_t + gamma * (alpha - miss_t). y is only used AFTER each interval is formed."""
    a_t, lo, hi = alpha, np.empty(len(pred)), np.empty(len(pred))
    for i in range(len(pred)):
        q = conformal_q(pool_abs, min(max(a_t, 1e-4), 0.9999))
        lo[i], hi[i] = pred[i] - q, pred[i] + q
        miss = float(y[i] < lo[i] or y[i] > hi[i])
        a_t += gamma * (alpha - miss)
    return lo, hi


def evaluate_all(df, features, target="DeltaS", alpha=0.1, kind="linear", k=5, gamma=0.02, with_cqr=True):
    """Leave-one-race-out. Returns per-race results and the stacked prediction rows for conditional coverage."""
    races = sorted(df["RaceId"].unique())
    rows, store = [], []
    for r in races:
        train, test = df[df["RaceId"] != r], df[df["RaceId"] == r].sort_values(["LapNumber", "Driver"])
        y = test[target].to_numpy()
        base = clone(point_model(kind)).fit(train[features], train[target])
        pred = base.predict(test[features])
        out = {}
        # (a) v3 style: one calibration race
        cal_r = [x for x in races if x != r][0]
        cal = train[train["RaceId"] == cal_r]
        tr1 = train[train["RaceId"] != cal_r]
        m1 = clone(point_model(kind)).fit(tr1[features], tr1[target])
        q1 = conformal_q(np.abs(cal[target] - m1.predict(cal[features])), alpha)
        p1 = m1.predict(test[features])
        out["single_race_conformal"] = (p1 - q1, p1 + q1)
        # (b) cross-race pooled conformal
        res = cross_race_residuals(train, features, target, kind, k)
        pool = np.abs(res["_res"].to_numpy())
        q = conformal_q(pool, alpha)
        out["cross_race_conformal"] = (pred - q, pred + q)
        # (c) + adaptive
        out["cross_race_ACI"] = aci_bounds(pred, y, pool, alpha, gamma)
        # (d) quantile regression raw and (e) CQR
        if with_cqr:
            lo_m = quantile_model(alpha / 2).fit(train[features], train[target])
            hi_m = quantile_model(1 - alpha / 2).fit(train[features], train[target])
            Q = np.column_stack([quantile_model(t).fit(train[features], train[target]).predict(test[features]) for t in TAUS])
            Q = np.sort(Q, axis=1)
            lo, hi = lo_m.predict(test[features]), hi_m.predict(test[features])
            out["quantile_regression_raw"] = (lo, hi)
            sc = []
            for fold in _race_folds(sorted(train["RaceId"].unique()), k):
                tr, ca = train[~train["RaceId"].isin(fold)], train[train["RaceId"].isin(fold)]
                lm, hm = quantile_model(alpha / 2).fit(tr[features], tr[target]), quantile_model(1 - alpha / 2).fit(tr[features], tr[target])
                sc.append(np.maximum(lm.predict(ca[features]) - ca[target], ca[target] - hm.predict(ca[features])))
            qc = conformal_q(np.concatenate(sc), alpha)
            out["CQR"] = (lo - qc, hi + qc)
            crps = crps_from_quantiles(y, Q)
        for name, (l, h) in out.items():
            rows.append(dict(test_race=r, method=name, coverage=float(np.mean((y >= l) & (y <= h))),
                             width=float(np.mean(h - l)), interval_score=interval_score(y, l, h, alpha),
                             crps_qr=crps if (with_cqr and name == "quantile_regression_raw") else np.nan))
            if name in ("single_race_conformal", "cross_race_conformal", "cross_race_ACI", "CQR"):
                store.append(test.assign(method=name, _lo=l, _hi=h))
    return pd.DataFrame(rows), pd.concat(store)


def summarize(res, alpha=0.1):
    s = res.groupby("method").agg(coverage=("coverage", "mean"), coverage_sd=("coverage", "std"),
                                  width=("width", "mean"), interval_score=("interval_score", "mean"),
                                  crps_qr=("crps_qr", "mean")).round(3)
    s["abs_coverage_gap"] = (s["coverage"] - (1 - alpha)).abs().round(3)
    return s.sort_values("abs_coverage_gap")


def conditional_coverage(store, target="DeltaS"):
    d = store.copy()
    d["hit"] = ((d[target] >= d["_lo"]) & (d[target] <= d["_hi"])).astype(float)
    d["age_bin"] = pd.cut(d["TyreLife"], [0, 10, 20, 100], labels=["1-10", "11-20", "21+"])
    d["compound"] = d["CompoundCode"].map({0: "SOFT", 1: "MEDIUM", 2: "HARD"})
    return (d.groupby(["method", "age_bin"], observed=True)["hit"].mean().unstack().round(3),
            d.groupby(["method", "compound"])["hit"].mean().unstack().round(3))
