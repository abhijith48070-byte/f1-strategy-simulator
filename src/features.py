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


def prepare(df):
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

    med = df.loc[clean].groupby(["RaceId", "Driver"])["LapTimeS"].transform("median")
    keep_idx = med.index[df.loc[med.index, "LapTimeS"] <= SLOW_LAP_CUTOFF * med]
    df["Clean"] = False
    df.loc[keep_idx, "Clean"] = True

    ref = (df[df["Clean"]].groupby(["RaceId", "Driver"])["LapTimeS"]
           .quantile(REF_QUANTILE).rename("DriverRef").reset_index())
    df = df.merge(ref, on=["RaceId", "Driver"], how="left")
    df["DeltaS"] = df["LapTimeS"] - df["DriverRef"]
    df["PrevDeltaS"] = df.groupby(["RaceId", "Driver"])["DeltaS"].shift(1)
    return df


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
