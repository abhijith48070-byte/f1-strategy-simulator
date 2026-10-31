"""Data loading: FastF1 -> tidy lap table. Also a synthetic generator for offline testing."""
from pathlib import Path
import numpy as np
import pandas as pd

KEEP = ["Driver", "Team", "LapNumber", "LapTime", "Stint", "Compound", "TyreLife", "FreshTyre",
        "TrackStatus", "PitInTime", "PitOutTime", "IsAccurate", "Position", "Deleted"]


def load_race(year, rnd, cache_dir="cache"):
    """Download one race with FastF1 and return a plain DataFrame of laps."""
    import fastf1
    Path(cache_dir).mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    session = fastf1.get_session(year, rnd, "R")
    session.load(laps=True, telemetry=False, weather=True, messages=False)
    laps = session.laps
    try:
        w = laps.get_weather_data().reset_index(drop=True)
        weather = w[["TrackTemp", "AirTemp", "Rainfall"]]
    except Exception:
        weather = pd.DataFrame({"TrackTemp": np.nan, "AirTemp": np.nan, "Rainfall": False},
                               index=range(len(laps)))
    out = pd.DataFrame(laps).reset_index(drop=True)
    out = out[[c for c in KEEP if c in out.columns]].copy()
    for c in ["TrackTemp", "AirTemp", "Rainfall"]:
        out[c] = weather[c].values
    out["LapTimeS"] = out["LapTime"].dt.total_seconds()
    out["PitInTimeS"] = out["PitInTime"].dt.total_seconds()
    out["PitOutTimeS"] = out["PitOutTime"].dt.total_seconds()
    out = out.drop(columns=["LapTime", "PitInTime", "PitOutTime"])
    out["RaceId"] = f"{year}_R{rnd:02d}"
    out["EventName"] = str(session.event["EventName"])
    out["TotalLaps"] = int(out["LapNumber"].max())
    return out


def build_dataset(rounds, year, csv_path="data/laps_raw.csv", cache_dir="cache", refresh=False, source="openf1"):
    """Load all races once, then reuse the CSV so reruns work offline.
    source: 'openf1' (free, no account; default) or 'fastf1' (needs an F1TV Access account since FastF1 3.9)."""
    p = Path(csv_path)
    if p.exists() and not refresh:
        return pd.read_csv(p)
    frames = []
    for r in rounds:
        print(f"loading {year} round {r} from {source} ...", flush=True)
        if source == "openf1":
            from .data_openf1 import load_race as load_one
            frames.append(load_one(year, r))
        else:
            frames.append(load_race(year, r, cache_dir))
    df = pd.concat(frames, ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df


def make_synthetic(n_races=6, n_drivers=16, seed=0):
    """Fake but plausible races with known physics, so the pipeline can be tested offline.
    Tyre degradation (s/lap): SOFT 0.09, MEDIUM 0.05, HARD 0.03. Fuel effect: -0.035 s/lap."""
    rng = np.random.default_rng(seed)
    deg = {"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.03}
    rows = []
    for r in range(n_races):
        base = rng.uniform(78, 100)
        total = int(rng.integers(52, 70))
        track_temp = rng.uniform(25, 45)
        sc_laps = set(rng.choice(np.arange(5, total - 5), size=2, replace=False).tolist())
        for d in range(n_drivers):
            offset = rng.normal(0, 0.4)
            c1 = rng.choice(["SOFT", "MEDIUM"])
            pit_lap = int(rng.integers(14, 32))
            for lap in range(1, total + 1):
                stint = 1 if lap <= pit_lap else 2
                comp = c1 if stint == 1 else "HARD"
                tyre_life = lap if stint == 1 else lap - pit_lap
                fuel = -0.035 * lap
                t = base + offset + deg[comp] * tyre_life + fuel + rng.normal(0, 0.25)
                status = "4" if lap in sc_laps else "1"
                if status == "4":
                    t += 25
                in_lap, out_lap = lap == pit_lap, lap == pit_lap + 1
                if in_lap:
                    t += 2.0
                if out_lap:
                    t += 17.0
                rows.append(dict(
                    Driver=f"D{d:02d}", Team=f"T{d // 2}", LapNumber=lap, Stint=stint, Compound=comp,
                    TyreLife=float(tyre_life), FreshTyre=True, TrackStatus=status,
                    PitInTimeS=1000.0 if in_lap else np.nan, PitOutTimeS=1020.0 if out_lap else np.nan,
                    IsAccurate=True, Position=float(1 + (d + lap // 10) % n_drivers), Deleted=False,
                    TrackTemp=track_temp + rng.normal(0, 0.5), AirTemp=track_temp - 10, Rainfall=False,
                    LapTimeS=t, RaceId=f"SYN_R{r:02d}", EventName=f"Synthetic {r}", TotalLaps=total))
    return pd.DataFrame(rows)
