"""Safety car (SC) / virtual safety car (VSC) modelling.

A neutralised lap changes three things, all parameterised in src/config.py (assumptions, not fitted):
  1. lap time becomes reference pace * SC_LAP_FACTOR (same for every strategy)
  2. tyres wear much less (tyre age advances by SC_WEAR_FACTOR of a lap)
  3. a stop whose in-lap is neutralised loses only SC_PIT_LOSS_FACTOR of the green-flag pit loss
The same arrays are used for one deterministic scenario (n=1) and for thousands of random Monte Carlo worlds.

Simplifications: at most one neutralisation per race; pit lane never closed; field compression and restarts
are not modelled (they affect position, not the time-based comparison used here).
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import config as C

KIND_CODE = {None: 0, "SC": 1, "VSC": 2}


@dataclass(frozen=True)
class Neutral:
    kind: str          # "SC" or "VSC"
    start: int         # first neutralised lap
    duration: int

    @property
    def laps(self):
        return range(self.start, self.start + self.duration)

    @property
    def label(self):
        return f"{self.kind} lap {self.start}" + (f" ({self.duration} laps)" if self.duration != 1 else "")


def default_duration(kind):
    return C.DEFAULT_SC_DURATION if kind == "SC" else C.DEFAULT_VSC_DURATION


class NeutralArrays:
    """Per-simulation lap arrays, shape (n, L+2), index = lap number (column 0 and L+1 unused)."""

    def __init__(self, L, ref_lap_s, starts, durations, kinds):
        starts, durations, kinds = (np.atleast_1d(np.asarray(a, int)) for a in (starts, durations, kinds))
        n = len(starts)
        laps = np.arange(L + 2)[None, :]
        self.mask = (kinds[:, None] > 0) & (laps >= starts[:, None]) & (laps < (starts + durations)[:, None])
        lapf = np.array([1.0, C.SC_LAP_FACTOR, C.VSC_LAP_FACTOR])[kinds]
        wear = np.array([1.0, C.SC_WEAR_FACTOR, C.VSC_WEAR_FACTOR])[kinds]
        pit = np.array([1.0, C.SC_PIT_LOSS_FACTOR, C.VSC_PIT_LOSS_FACTOR])[kinds]
        self.lap_extra = np.where(self.mask, ref_lap_s * (lapf[:, None] - 1.0), 0.0)   # extra seconds on a neutral lap
        w = np.where(self.mask, wear[:, None], 1.0)
        w[:, 0] = 0.0
        self.W = np.cumsum(w, axis=1)            # W[:, l] = effective laps of wear accumulated up to and including lap l
        self.pit_factor = np.where(self.mask, pit[:, None], 1.0)
        self.n = n

    @classmethod
    def single(cls, L, ref_lap_s, neutral=None):
        if neutral is None:
            return cls(L, ref_lap_s, [0], [0], [0])
        return cls(L, ref_lap_s, [neutral.start], [neutral.duration], [KIND_CODE[neutral.kind]])

    def age(self, start_age, s, laps):
        """Effective tyre age on `laps` for a stint that starts on lap s with age `start_age` on lap s. -> (n, m)"""
        laps = np.asarray(laps, int)
        return start_age + self.W[:, laps - 1] - self.W[:, [s - 1]]


def neutral_stats(df, min_races=5):
    """Estimate how often races are neutralised from the data: share of races with a neutralised stretch,
    start position (fraction of race distance) and duration of each stretch. The OpenF1 loader marks SC and VSC
    laps with the same status, so they cannot be told apart: the total rate is split using the config ratio."""
    ts = df["TrackStatus"].astype(str)
    neutral_lap = ts.str.contains("[4567]", regex=True)
    frac = neutral_lap.groupby([df["RaceId"], df["LapNumber"]]).mean()
    starts, durs, with_event = [], [], 0
    races = df["RaceId"].unique()
    for r in races:
        s = frac.loc[r]
        laps = s.index[(s > 0.5).to_numpy()].to_numpy()
        total = int(df.loc[df["RaceId"] == r, "TotalLaps"].iloc[0])
        if len(laps) == 0:
            continue
        with_event += 1
        run_start = prev = laps[0]
        for l in list(laps[1:]) + [None]:
            if l is None or l != prev + 1:
                starts.append(run_start / total); durs.append(prev - run_start + 1)
                if l is not None:
                    run_start = l
            if l is not None:
                prev = l
    if len(races) < min_races:
        return dict(p_sc=C.P_SC_DEFAULT, p_vsc=C.P_VSC_DEFAULT, starts=[], durations=[], source="config defaults")
    p_any = with_event / len(races)
    share_sc = C.P_SC_DEFAULT / (C.P_SC_DEFAULT + C.P_VSC_DEFAULT)
    return dict(p_sc=p_any * share_sc, p_vsc=p_any * (1 - share_sc), starts=starts, durations=durs,
                source=f"{len(races)} races in the data")


def sample_neutrals(n, first_lap, L, stats, rng):
    """n random neutralisations (or none) starting at or after `first_lap`. Returns (starts, durations, kinds)."""
    u = rng.random(n)
    kinds = np.where(u < stats["p_sc"], 1, np.where(u < stats["p_sc"] + stats["p_vsc"], 2, 0))
    lo, hi = max(first_lap, 2), max(L - 3, max(first_lap, 2))
    if stats["starts"]:
        st = np.round(rng.choice(stats["starts"], size=n) * L).astype(int)
        st = np.clip(st, lo, hi)
        dur = rng.choice(stats["durations"], size=n).astype(int)
    else:
        st = rng.integers(lo, hi + 1, size=n)
        dur = np.where(kinds == 1, C.DEFAULT_SC_DURATION, C.DEFAULT_VSC_DURATION)
    dur = np.clip(dur, 1, 8)
    return st, dur, kinds
