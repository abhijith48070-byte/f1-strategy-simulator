"""Model zoo. Every model sees the same features; the simplest baseline is the one to beat."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class MeanByCompound(BaseEstimator, RegressorMixin):
    """Baseline: predict the training-set mean delta for each compound."""
    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.means_ = y.groupby(X["CompoundCode"]).mean().to_dict()
        self.global_ = float(y.mean())
        return self

    def predict(self, X):
        return X["CompoundCode"].map(self.means_).fillna(self.global_).to_numpy()


def get_models():
    models = {
        "baseline_compound_mean": MeanByCompound(),
        "linear": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()),
        "hist_gbr": HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300,
                                                  random_state=0),
    }
    try:
        import lightgbm as lgb
        models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    except ImportError:
        pass
    return models
