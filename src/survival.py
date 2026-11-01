"""Survivorship bias in tyre degradation, and a correction by inverse-probability weighting (IPW).

Problem: laps on old tyres only exist for stints that were still worth continuing. If drivers pit when their tyres degrade
quickly, old-age laps come disproportionately from slow-degrading stints, so a pooled slope understates degradation.

Method (discrete-time survival analysis):
  1. Person-period data: one row per lap of a stint, event = the stint ends with a pit stop on that lap.
  2. Hazard model h_t = P(pit at end of lap t | stint still running, features known at that time). Features use only
     past laps: tyre age, compound, race progress and the stint's observed pace loss so far.
  3. A lap at age a is observed only if the stint survived to a, so weight it by 1 / prod_{k<a}(1 - h_k) (stabilised by the
     age-and-compound-only survival, truncated). This re-creates the population of stints that STARTED.
  4. Estimate within-driver slopes with weighted least squares; compare naive (unweighted) and IPW.
Assumption (sequential ignorability): the pit decision depends only on the observed history in the features. Real teams use
information we do not have (tyre temperatures, rivals), so on real data this is a sensitivity analysis, not proof.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

COMPS = [0, 1, 2]


def _stint_id(df):
    return df["RaceId"].astype(str) + "|" + df["Driver"].astype(str) + "|" + df["Stint"].astype(str)


def add_stint_features(df):
    """Adds StintId and StintLoss (mean of the last 3 laps minus mean of the first 3 laps of the stint, using only
    laps up to the previous one). All quantities are causal."""
    d = df.sort_values(["RaceId", "Driver", "LapNumber"]).reset_index(drop=True).copy()
    d["StintId"] = _stint_id(d)
    y = d["LapTimeS"].where(d["Clean"])
    g = y.groupby(d["StintId"])
    first3 = g.transform(lambda s: s.dropna().iloc[:3].mean() if s.notna().sum() >= 3 else np.nan)
    last3 = g.transform(lambda s: s.shift(1).rolling(3, min_periods=3).mean())
    d["StintLoss"] = (last3 - first3)
    d["HasLoss"] = d["StintLoss"].notna().astype(float)
    d["StintLoss"] = d["StintLoss"].fillna(0.0)
    return d


def person_period(df):
    d = df[df["CompoundCode"].notna() & df["TyreLife"].notna() & df["Stint"].notna()].copy()
    d["Event"] = d["IsInLap"].astype(int)
    return d


def _hazard_X(d, full=True):
    cols = [d["TyreLife"], d["TyreLife"] ** 2 / 50.0]
    for c in COMPS:
        cols.append((d["CompoundCode"] == c).astype(float))
    if full:
        cols += [d["LapFrac"], d["StintLoss"], d["HasLoss"], d["StintLoss"] * d["HasLoss"]]
    return np.column_stack(cols)


def fit_hazards(pp):
    """Returns (full-model hazards, age+compound-only hazards) for every row of pp."""
    y = pp["Event"].to_numpy()
    out = []
    for full in (True, False):
        X = _hazard_X(pp, full)
        sc = StandardScaler().fit(X)
        m = LogisticRegression(C=1.0, max_iter=1000).fit(sc.transform(X), y)
        out.append(np.clip(m.predict_proba(sc.transform(X))[:, 1], 1e-6, 0.95))
    return out[0], out[1]


def ipw_weights(pp, h_full, h_marg, clip=(0.02, 100.0)):
    """Stabilised IPW: S_marginal(a) / S_full(a), where S(a) = prod_{k<a}(1 - h_k) within each stint."""
    d = pp.assign(_lf=np.log1p(-h_full), _lm=np.log1p(-h_marg))
    d = d.sort_values(["StintId", "LapNumber"])
    cum_f = d.groupby("StintId")["_lf"].cumsum() - d["_lf"]       # exclude the current lap
    cum_m = d.groupby("StintId")["_lm"].cumsum() - d["_lm"]
    w = np.exp(cum_m - cum_f)
    w = np.clip(w, *clip)      # tighter clip = less variance but more residual bias; see effective_sample_fraction
    return w.reindex(pp.index).to_numpy()


def effective_sample_fraction(w):
    """Kish effective sample size / n. Near 1 = weights are harmless; small = the correction is noisy."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum() / len(w))


def degradation_slopes(df, weights=None, only_clean=True):
    """Weighted within-driver regression of lap time on race progress and tyre age by compound.
    Demeans every variable within (race, driver), so absolute pace levels drop out."""
    d = df[df["Clean"]].copy() if only_clean else df.copy()
    d = d[d["CompoundCode"].notna() & d["TyreLife"].notna()]
    w = np.ones(len(d)) if weights is None else pd.Series(weights, index=df.index).loc[d.index].to_numpy()
    cols = {"LapFrac": d["LapFrac"].to_numpy()}
    for c in COMPS:
        cols[f"age_{c}"] = (d["TyreLife"] * (d["CompoundCode"] == c)).to_numpy()
        if c != COMPS[0]:
            cols[f"is_{c}"] = (d["CompoundCode"] == c).astype(float).to_numpy()
    X = pd.DataFrame(cols, index=d.index)
    X["y"] = d["LapTimeS"].to_numpy()
    X[["RaceId", "Driver"]] = d[["RaceId", "Driver"]].to_numpy()
    Xc = X.drop(columns=["RaceId", "Driver"]) - X.groupby(["RaceId", "Driver"])[list(cols) + ["y"]].transform("mean")
    A = Xc[list(cols)].to_numpy() * np.sqrt(w)[:, None]
    b = Xc["y"].to_numpy() * np.sqrt(w)
    beta = np.linalg.lstsq(A, b, rcond=None)[0]
    return pd.Series(beta, index=list(cols))


def compare_slopes(df, n_boot=0, seed=0):
    """Naive vs IPW degradation (s per lap of tyre age) per compound; optional cluster bootstrap over races."""
    def one(data):
        d = add_stint_features(data)
        pp = person_period(d)
        h_full, h_marg = fit_hazards(pp)
        w = pd.Series(np.nan, index=d.index)
        w.loc[pp.index] = ipw_weights(pp, h_full, h_marg)
        w = w.fillna(1.0)
        naive = degradation_slopes(d)
        ipw = degradation_slopes(d, weights=w.to_numpy())
        return naive, ipw, w
    naive, ipw, w = one(df)
    res = pd.DataFrame({"naive": [naive[f"age_{c}"] for c in COMPS], "ipw": [ipw[f"age_{c}"] for c in COMPS]},
                       index=["SOFT", "MEDIUM", "HARD"])
    res["ipw_minus_naive"] = res["ipw"] - res["naive"]
    if n_boot:
        rng = np.random.default_rng(seed)
        races = df["RaceId"].unique()
        diffs = []
        for _ in range(n_boot):
            pick = rng.choice(races, size=len(races), replace=True)
            parts = [df[df["RaceId"] == r].assign(RaceId=f"{r}#{i}") for i, r in enumerate(pick)]
            n_, i_, _w = one(pd.concat(parts, ignore_index=True))
            diffs.append([i_[f"age_{c}"] - n_[f"age_{c}"] for c in COMPS])
        diffs = np.array(diffs)
        res["diff_ci_lo"], res["diff_ci_hi"] = np.percentile(diffs, 5, axis=0), np.percentile(diffs, 95, axis=0)
    return res, w


def make_informative_stint_data(n_races=10, n_drivers=20, seed=0, strength=1.0):
    """Synthetic races with KNOWN mean degradation (SOFT .09, MEDIUM .05, HARD .03 s/lap) where each stint has its own
    wear rate and drivers pit earlier when their observed pace loss is high (informative stopping)."""
    rng = np.random.default_rng(seed)
    mean_deg = {0: 0.09, 1: 0.05, 2: 0.03}
    rows = []
    for r in range(n_races):
        total = int(rng.integers(52, 66))
        base = rng.uniform(80, 100)
        for d in range(n_drivers):
            offset = rng.normal(0, 0.5)
            comp = int(rng.choice([0, 1]))
            age, stint, times = 0, 1, []
            wear = max(0.004, rng.normal(mean_deg[comp], 0.03))
            for lap in range(1, total + 1):
                age += 1
                t = base + offset - 0.035 * lap + wear * age + rng.normal(0, 0.25)
                times.append(t)
                loss = (np.mean(times[-3:]) - np.mean(times[:3])) if age >= 6 else 0.0
                pit = age >= 8 and rng.random() < 0.01 + 0.35 * strength / (1 + np.exp(-6 * (loss - 0.9)))
                pit = pit or age >= 45
                last = lap == total
                rows.append(dict(RaceId=f"S{r:02d}", Driver=f"D{d:02d}", LapNumber=lap, Stint=stint,
                                 CompoundCode=float(comp), TyreLife=float(age), LapFrac=lap / total,
                                 LapTimeS=t, TrackStatus="1", IsInLap=bool(pit and not last),
                                 IsOutLap=False, Clean=True, TotalLaps=total))
                if pit and not last:
                    comp = int(rng.choice([c for c in (0, 1, 2) if c != comp]))
                    wear = max(0.004, rng.normal(mean_deg[comp], 0.03))
                    age, stint, times = 0, stint + 1, []
    return pd.DataFrame(rows), mean_deg
