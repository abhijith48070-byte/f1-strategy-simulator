"""Cleaning + feature/target construction.

Fixes vs. the first version of this project:
 * everything is sorted and grouped by (race, driver) so shifts never cross drivers
 * the pit-stop label is the in-lap flag; no feature uses next-lap or pit-time information
 * the regression target is lap time relative to the driver's own reference pace,
   which removes the circuit and driver baseline so races can be compared
"""
import numpy as np
import pandas as pd
from .config import COMPOUNDS, SLOW_LAP_CUTOFF, REF_QUANTILE


def prepare(df, ref_mode="full", early_laps=8):
    """ref_mode decides what DeltaS is measured against:
      "full"  (v2 default) driver's 15th-percentile clean lap over the WHOLE race. Simple, but it reads laps that
              are in the future of any lap being predicted, and it depends on the driver's own stint mix, which
              biases tyre slopes toward zero (see tests/test_strategy.py).
      "early" CAUSAL. Median of the driver's clean laps 2..early_laps only; the slow-lap filter uses a running median
              of past laps; laps <= early_laps get no target. Nothing from a lap's future is used to build its target.
      "fe"    unbiased tyre slopes: a within-(race, driver) regression on all clean laps estimates the tyre/fuel
              effects, and the reference is the driver's fitted pace at tyre age 0 / lap 0 (fresh MEDIUM).
              Uses the whole race, so it is NOT causal; use it to train planning models on past races.
    """
    if ref_mode not in ("full", "early", "fe"):
        raise ValueError(ref_mode)
    df = df.sort_values(["RaceId", "Driver", "LapNumber"]).reset_index(drop=True).copy()
    df["CompoundCode"] = df["Compound"].map(COMPOUNDS)
    df["LapFrac"] = df["LapNumber"] / df["TotalLaps"]
    df["IsInLap"] = df["PitInTimeS"].notna()
    df["IsOutLap"] = df["PitOutTimeS"].notna()
    df["FreshTyre"] = df["FreshTyre"].astype(float)
    df["TrackStatus"] = df["TrackStatus"].astype(str)

    clean = (
        (df["TrackStatus"] == "1") & ~df["IsInLap"] & ~df["IsOutLap"] & (df["LapNumber"] > 1)
        & df["LapTimeS"].notna() & df["CompoundCode"].notna() & df["TyreLife"].notna()
    )
    if "Rainfall" in df:
        clean &= ~df["Rainfall"].fillna(False).astype(bool)
    if "Deleted" in df:
        clean &= ~df["Deleted"].fillna(False).astype(bool)
    if "IsAccurate" in df:
        clean &= df["IsAccurate"].fillna(False).astype(bool)

    if ref_mode == "early":      # running median of PAST eligible laps only (needs 3 before it applies)
        med = (df.loc[clean].groupby(["RaceId", "Driver"])["LapTimeS"]
               .transform(lambda x: x.expanding(3).median().shift(1)))
        med = med.fillna(np.inf)
    else:
        med = df.loc[clean].groupby(["RaceId", "Driver"])["LapTimeS"].transform("median")
    keep_idx = med.index[df.loc[med.index, "LapTimeS"] <= SLOW_LAP_CUTOFF * med]
    df["Clean"] = False
    df.loc[keep_idx, "Clean"] = True

    if ref_mode == "full":
        ref = (df[df["Clean"]].groupby(["RaceId", "Driver"])["LapTimeS"]
               .quantile(REF_QUANTILE).rename("DriverRef").reset_index())
    elif ref_mode == "early":
        e = df[df["Clean"] & (df["LapNumber"] <= early_laps)]
        g = e.groupby(["RaceId", "Driver"])["LapTimeS"]
        ref = g.median().where(g.count() >= 3).rename("DriverRef").reset_index()
    else:
        ref = _fe_reference(df)
    df = df.merge(ref, on=["RaceId", "Driver"], how="left")
    df["DeltaS"] = df["LapTimeS"] - df["DriverRef"]
    if ref_mode == "early":
        df.loc[df["LapNumber"] <= early_laps, "DeltaS"] = np.nan
    df.attrs["ref_mode"] = ref_mode
    df["PrevDeltaS"] = df.groupby(["RaceId", "Driver"])["DeltaS"].shift(1)
    return df


def _fe_reference(df):
    """Driver reference = fitted pace at tyre age 0, lap 0, MEDIUM (within-unit regression, pooled over races)."""
    from .bayes import design_matrix
    d = df[df["Clean"]].dropna(subset=["CompoundCode", "TyreLife", "LapNumber", "LapTimeS"])
    X = design_matrix(d["CompoundCode"], d["TyreLife"], d["LapNumber"])
    key = (d["RaceId"].astype(str) + "|" + d["Driver"].astype(str)).to_numpy()
    Xc = X - pd.DataFrame(X).groupby(key).transform("mean").to_numpy()
    y = d["LapTimeS"].to_numpy()
    yc = y - pd.Series(y).groupby(key).transform("mean").to_numpy()
    theta = np.linalg.lstsq(Xc, yc, rcond=None)[0]
    base = pd.Series(y - X @ theta).groupby(key).mean()
    out = pd.DataFrame({"unit": base.index, "DriverRef": base.values})
    out[["RaceId", "Driver"]] = out["unit"].str.split("|", n=1, expand=True)
    return out[["RaceId", "Driver", "DriverRef"]]


def regression_set(df):
    """Clean green-flag laps with a defined target."""
    return df[df["Clean"] & df["DeltaS"].notna()].reset_index(drop=True)


def pit_set(df):
    """All laps with a known tyre state; label = this lap is a pit in-lap."""
    d = df[df["CompoundCode"].notna() & df["TyreLife"].notna()].copy()
    d["PitThisLap"] = d["IsInLap"].astype(int)
    return d.reset_index(drop=True)


def pit_loss_samples(df):
    """One pit-loss estimate per green-flag stop: in-lap + out-lap - 2 * reference pace.
    Approximate: it also contains small tyre-age effects."""
    d = df.copy()
    g = d.groupby(["RaceId", "Driver"])
    d["NextLapTimeS"] = g["LapTimeS"].shift(-1)
    d["NextTrack"] = g["TrackStatus"].shift(-1)
    m = (d["IsInLap"] & (d["TrackStatus"] == "1") & (d["NextTrack"] == "1")
         & d["LapTimeS"].notna() & d["NextLapTimeS"].notna() & d["DriverRef"].notna())
    out = d.loc[m, ["RaceId"]].copy()
    out["loss_s"] = d.loc[m, "LapTimeS"] + d.loc[m, "NextLapTimeS"] - 2 * d.loc[m, "DriverRef"]
    # Physical pit losses are roughly 15-35 s. Red-flag stops and timing glitches give losses of hundreds of
    # seconds, which are not racing pit stops, so keep only the plausible range.
    return out[(out["loss_s"] >= 8) & (out["loss_s"] <= 60)]


def estimate_pit_loss(df):
    s = pit_loss_samples(df)
    return s.groupby("RaceId")["loss_s"].median()
