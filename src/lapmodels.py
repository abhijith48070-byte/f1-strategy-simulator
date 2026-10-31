"""Lap-time models behind one interface so the optimizer / Monte Carlo do not care which one they get.

    mean(comp, age, lap, total, fresh, state)        -> lap-time delta in seconds (vs the model's reference pace)
    draw_world(n, L, rng, state)                      -> shared random draws for n simulated races
    sample(world, comp, age, lap, total, fresh, state)-> (n, m) simulated deltas for a stint
    reference_pace(laps)                              -> seconds; add to a delta to get a lap time

`age` may be 1-D (m,) or 2-D (n, m) (tyre age differs per simulation when safety-car laps wear tyres less).
`lap` is always 1-D (m,) of integer lap numbers. Noise is indexed by lap number and shared by every strategy in a
simulated race (common random numbers), so differences between strategies are not drowned in noise.
"""
import numpy as np
import pandas as pd
from .bayes import design_matrix, P as N_PARAMS


class LapModel:
    max_life = {}
    name = "base"

    def life_limit(self, comp):
        return self.max_life.get(comp, 10 ** 6)


class BayesLapModel(LapModel):
    """Wraps HierarchicalTyreModel. Uncertainty = posterior of the coefficients + new-race variability + noise."""
    name = "bayes"

    def __init__(self, bayes, max_life=None, group=None):
        self.bayes, self.group = bayes, group
        self.max_life = dict(max_life or bayes.max_age_)

    def _lin(self, theta, comp, age, lap):
        # theta (..., 6): returns a + b*age + f*lap for one compound code; works for mean (6,) or draws (n,6)
        th = np.atleast_2d(theta)
        a = 0.0 if comp == 1 else th[:, 0 if comp == 0 else 1]
        b, f = th[:, 2 + comp], th[:, 5]
        age = np.asarray(age, float)
        if age.ndim == 1:
            age = age[None, :]
        return np.asarray(a)[..., None] + b[:, None] * age + f[:, None] * np.asarray(lap, float)[None, :]

    def mean(self, comp, age, lap, total, fresh, state):
        return self._lin(self.bayes.mean_theta(), comp, age, lap)[0]

    def draw_world(self, n, L, rng, state):
        th, sg = self.bayes.draw_theta(n, rng, group=self.group)
        return dict(theta=th, eps=sg[:, None] * rng.standard_normal((n, L + 2)))

    def sample(self, world, comp, age, lap, total, fresh, state):
        return self._lin(world["theta"], comp, age, lap) + world["eps"][:, np.asarray(lap, int)]

    def reference_pace(self, laps):
        return self.bayes.base_pace(laps)


class GBRLapModel(LapModel):
    """Wraps any sklearn-style regressor trained on DeltaS. Uncertainty = bootstrapped calibration residuals +
    a shared track-temperature shift per simulated race. (No parameter uncertainty: a point-forecast model.)"""

    def __init__(self, model, features, residuals, max_life, temp_sd=3.0, name="hist_gbr"):
        self.model, self.features, self.residuals = model, list(features), np.asarray(residuals, float)
        self.max_life, self.temp_sd, self.name = dict(max_life), temp_sd, name

    def _predict(self, comp, age, lap, total, fresh, state, temp_shift=None):
        age = np.asarray(age, float)
        n = 1 if age.ndim == 1 else age.shape[0]
        age2 = np.broadcast_to(age, (n, len(lap)))
        lap2 = np.broadcast_to(np.asarray(lap, float), (n, len(lap)))
        shift = np.zeros(n) if temp_shift is None else np.asarray(temp_shift, float)
        cols = {"TyreLife": age2.ravel(), "LapFrac": (lap2 / total).ravel(),
                "CompoundCode": np.full(age2.size, comp, dtype=float), "FreshTyre": np.full(age2.size, fresh),
                "TrackTemp": (state["track_temp"] + shift[:, None] + 0 * age2).ravel(),
                "AirTemp": (state["air_temp"] + 0.5 * shift[:, None] + 0 * age2).ravel(),
                "Position": np.full(age2.size, state["position"])}
        X = pd.DataFrame({k: cols[k] for k in self.features})
        return self.model.predict(X).reshape(n, len(lap))

    def mean(self, comp, age, lap, total, fresh, state):
        return self._predict(comp, age, lap, total, fresh, state)[0]

    def draw_world(self, n, L, rng, state):
        return dict(temp_shift=self.temp_sd * rng.standard_normal(n),
                    eps=rng.choice(self.residuals, size=(n, L + 2)))

    def sample(self, world, comp, age, lap, total, fresh, state):
        return (self._predict(comp, age, lap, total, fresh, state, world["temp_shift"])
                + world["eps"][:, np.asarray(lap, int)])

    def reference_pace(self, laps):
        return float(laps["DriverRef"].median())
