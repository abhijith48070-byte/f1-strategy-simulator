"""Interpretability: SHAP if installed, otherwise leave-one-race-out permutation importance."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance


def permutation_loro(df, features, model, target="DeltaS", n_repeats=5, seed=0):
    rows = []
    for r in sorted(df["RaceId"].unique()):
        train, test = df[df["RaceId"] != r], df[df["RaceId"] == r]
        m = clone(model).fit(train[features], train[target])
        pi = permutation_importance(m, test[features], test[target], n_repeats=n_repeats,
                                    random_state=seed, scoring="neg_mean_absolute_error")
        rows.append(pd.Series(pi.importances_mean, index=features, name=r))
    t = pd.DataFrame(rows)
    return pd.DataFrame({"mean_MAE_increase_s": t.mean(), "std_across_races": t.std()}).sort_values(
        "mean_MAE_increase_s", ascending=False)


def shap_summary(df, features, target, out_png):
    try:
        import shap, lightgbm as lgb
        import matplotlib.pyplot as plt
    except ImportError:
        print("shap/lightgbm not installed - skipping SHAP plot (permutation importance still runs).")
        return False
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, verbose=-1)
    m.fit(df[features], df[target])
    sv = shap.TreeExplainer(m).shap_values(df[features])
    shap.summary_plot(sv, df[features], show=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return True
