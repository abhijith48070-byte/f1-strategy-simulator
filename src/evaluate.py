"""Leave-one-race-out evaluation with split-conformal prediction intervals.

For each held-out race we train on the other races, except one calibration race that is
used only to set the interval width. The test race is never seen during training or calibration.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score


def conformal_quantile(abs_residuals, alpha):
    r = np.sort(np.asarray(abs_residuals))
    n = len(r)
    k = min(int(np.ceil((n + 1) * (1 - alpha))), n)
    return float(r[k - 1])


def loro_regression(df, features, models, target="DeltaS", alpha=0.10):
    races = sorted(df["RaceId"].unique())
    if len(races) < 3:
        raise ValueError("Need at least 3 races (train, calibration, test).")
    rows = []
    for i, test_r in enumerate(races):
        cal_r = races[(i + 1) % len(races)]
        train = df[~df["RaceId"].isin([test_r, cal_r])]
        cal = df[df["RaceId"] == cal_r]
        test = df[df["RaceId"] == test_r]
        for name, proto in models.items():
            m = clone(proto).fit(train[features], train[target])
            p = m.predict(test[features])
            q = conformal_quantile(np.abs(cal[target] - m.predict(cal[features])), alpha)
            err = test[target].to_numpy() - p
            rows.append(dict(test_race=test_r, model=name, n=len(test),
                             MAE=mean_absolute_error(test[target], p),
                             RMSE=float(np.sqrt(np.mean(err ** 2))),
                             R2=r2_score(test[target], p),
                             coverage=float(np.mean(np.abs(err) <= q)), width_s=2 * q))
    res = pd.DataFrame(rows)
    summary = (res.groupby("model")[["MAE", "RMSE", "R2", "coverage", "width_s"]]
               .agg(["mean", "std"]).round(3))
    return res, summary


def loro_pit_classifier(df, features, label="PitThisLap"):
    """Pit-stop prediction. Reports average precision against the base rate (a random model's score)."""
    races = sorted(df["RaceId"].unique())
    rows = []
    for test_r in races:
        train, test = df[df["RaceId"] != test_r], df[df["RaceId"] == test_r]
        clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0)
        clf.fit(train[features], train[label])
        p = clf.predict_proba(test[features])[:, 1]
        rows.append(dict(test_race=test_r, base_rate=test[label].mean(),
                         avg_precision=average_precision_score(test[label], p)))
    return pd.DataFrame(rows)
