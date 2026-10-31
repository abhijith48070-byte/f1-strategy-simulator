"""Within-driver evaluation: does the model get the SHAPE of a driver's race right?

Target = lap time minus the mean lap time of the same driver in the same race; predictions are demeaned the same way.
No reference pace is used and nothing is learned from the held-out race.
skill_vs_zero = 1 - SSE(model) / SSE(predict 'no change within the race'). 0 = no better than a flat line.
"""
import pandas as pd
from sklearn.base import clone

UNIT = ["RaceId", "Driver"]


def loro_within(df, features, models):
    d = df.copy()
    d["y_c"] = d["LapTimeS"] - d.groupby(UNIT)["LapTimeS"].transform("mean")
    rows = []
    for test_r in sorted(d["RaceId"].unique()):
        train, test = d[d["RaceId"] != test_r], d[d["RaceId"] == test_r]
        sst, mae0 = (test["y_c"] ** 2).sum(), test["y_c"].abs().mean()
        rows.append(dict(test_race=test_r, model="zero (flat within race)", n=len(test), MAE=mae0, skill_vs_zero=0.0))
        for name, proto in models.items():
            m = clone(proto).fit(train[features], train["y_c"])
            p = pd.Series(m.predict(test[features]), index=test.index)
            p_c = p - p.groupby([test["RaceId"], test["Driver"]]).transform("mean")
            err = test["y_c"] - p_c
            rows.append(dict(test_race=test_r, model=name, n=len(test), MAE=err.abs().mean(),
                             skill_vs_zero=1 - (err ** 2).sum() / sst))
    res = pd.DataFrame(rows)
    summ = res.groupby("model")[["MAE", "skill_vs_zero"]].agg(["mean", "std"]).round(3)
    wins = res[res.model != "zero (flat within race)"].merge(
        res[res.model == "zero (flat within race)"][["test_race", "MAE"]], on="test_race", suffixes=("", "_zero"))
    beats = (wins.assign(win=wins["MAE"] < wins["MAE_zero"]).groupby("model")["win"].sum())
    return res, summ, beats, d["RaceId"].nunique()
