"""Multi-horizon lap-time forecasting (5, 10, 15 laps ahead) with leave-one-race-out evaluation.

Target = lap time at t+h minus the driver's recent level (mean of the last 3 clean laps up to t), within one stint.
Features at time t only: tyre age, race progress, compound and the recent trend. 'Persistence' (predict no change) is the baseline.
Neural models (LSTM, Temporal Fusion Transformer) are NOT included: they need PyTorch and ~20k laps is small for them.
"""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATS = ["TyreLife", "LapFrac", "CompoundCode", "trend"]


def horizon_table(df, h):
    d = df.sort_values(["RaceId", "Driver", "LapNumber"]).reset_index(drop=True).copy()
    d["StintId"] = d["RaceId"].astype(str) + "|" + d["Driver"].astype(str) + "|" + d["Stint"].fillna(0).astype(str)
    y = d["LapTimeS"].where(d["Clean"])
    d["lvl"] = y.groupby(d["StintId"]).transform(lambda s: s.rolling(3, min_periods=2).mean())
    d["trend"] = (d["lvl"] - d.groupby("StintId")["lvl"].shift(3)).fillna(0.0)
    d["target"] = y.groupby(d["StintId"]).shift(-h) - d["lvl"]
    ahead = d.groupby("StintId")["LapNumber"].shift(-h)
    ok = (ahead - d["LapNumber"] == h) & d["target"].notna() & d["lvl"].notna() & d["Clean"] & d["CompoundCode"].notna()
    return d[ok].reset_index(drop=True)


def models():
    return {
        "linear": make_pipeline(StandardScaler(), LinearRegression()),
        "hist_gbr": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.06, max_iter=150, random_state=0),
    }


def loro_horizon(df, horizons=(5, 10, 15)):
    rows = []
    for h in horizons:
        t = horizon_table(df, h)
        for r in sorted(t["RaceId"].unique()):
            tr, te = t[t["RaceId"] != r], t[t["RaceId"] == r]
            if len(te) < 50:
                continue
            e0 = te["target"].to_numpy()
            rows.append(dict(horizon=h, test_race=r, model="persistence", n=len(te),
                             MAE=np.abs(e0).mean(), RMSE=float(np.sqrt((e0 ** 2).mean()))))
            for name, proto in models().items():
                m = clone(proto).fit(tr[FEATS], tr["target"])
                e = te["target"].to_numpy() - m.predict(te[FEATS])
                rows.append(dict(horizon=h, test_race=r, model=name, n=len(te),
                                 MAE=np.abs(e).mean(), RMSE=float(np.sqrt((e ** 2).mean()))))
    res = pd.DataFrame(rows)
    summ = res.groupby(["horizon", "model"])[["MAE", "RMSE"]].agg(["mean", "std"]).round(3)
    piv = res.pivot_table(index=["horizon", "test_race"], columns="model", values="MAE")
    wins = {(h, m): int((piv.loc[h][m] < piv.loc[h]["persistence"]).sum()) for h in horizons for m in models()
            if h in piv.index.get_level_values(0)}
    return res, summ, wins, {h: int(piv.loc[h].shape[0]) for h in horizons if h in piv.index.get_level_values(0)}
