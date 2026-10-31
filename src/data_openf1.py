"""OpenF1 loader: free public API, no account or key, data from 2023 onward.
Rate limit: 3 requests/second and 30/minute, so downloads are throttled (about 6 calls per race).

The parsing is in `build_race_df`, a pure function that is unit-tested offline
(tests/test_openf1_parse.py). The HTTP part is thin so it is easy to debug."""
import time
import numpy as np
import pandas as pd

BASE = "https://api.openf1.org/v1"
_last_call = [0.0]


def get(endpoint, **params):
    import requests
    for attempt in range(6):
        wait = 2.3 - (time.time() - _last_call[0])      # stay under 30 calls/minute
        if wait > 0:
            time.sleep(wait)
        r = requests.get(f"{BASE}/{endpoint}", params=params, timeout=60)
        _last_call[0] = time.time()
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:                         # OpenF1 uses 404 for "no results"
            return []
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(6 * (attempt + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"OpenF1 request failed repeatedly: {endpoint} {params}")


def neutral_periods(rc):
    """(start, end) timestamps of Safety Car / Virtual Safety Car periods from race-control messages."""
    if rc.empty:
        return []
    rc = rc.sort_values("date")
    periods, start = [], None
    for _, r in rc.iterrows():
        msg = str(r.get("message", "")).upper()
        cat = str(r.get("category", "")).upper()
        if cat != "SAFETYCAR" and "SAFETY CAR" not in msg:
            continue
        if "DEPLOYED" in msg:
            start = start or r["date"]
        elif "IN THIS LAP" in msg or "ENDING" in msg:
            if start is not None:
                periods.append((start, r["date"]))
                start = None
    if start is not None:
        periods.append((start, rc["date"].max()))
    return periods


def build_race_df(laps, stints, weather, drivers, race_control, race_id, event_name):
    """Convert OpenF1 JSON lists into the same table the FastF1 loader produces."""
    L = pd.DataFrame(laps)
    if L.empty:
        raise ValueError(f"no laps returned for {race_id}")
    L["date_start"] = pd.to_datetime(L["date_start"], utc=True, errors="coerce")
    L = L.sort_values(["driver_number", "lap_number"]).reset_index(drop=True)

    # tyre info from stints
    S = pd.DataFrame(stints)
    L["Compound"], L["TyreLife"], L["FreshTyre"], L["Stint"] = None, np.nan, np.nan, np.nan
    for _, s in S.iterrows():
        m = ((L["driver_number"] == s["driver_number"]) & (L["lap_number"] >= s["lap_start"])
             & (L["lap_number"] <= s["lap_end"]))
        age0 = s.get("tyre_age_at_start", 0)
        age0 = 0 if pd.isna(age0) else age0
        L.loc[m, "Compound"] = str(s["compound"]).upper()
        L.loc[m, "TyreLife"] = age0 + (L.loc[m, "lap_number"] - s["lap_start"] + 1)
        L.loc[m, "FreshTyre"] = float(age0 == 0)
        L.loc[m, "Stint"] = s.get("stint_number", np.nan)

    # pit laps: the out-lap is flagged; the in-lap is the lap before it
    out_flag = L["is_pit_out_lap"].fillna(False).astype(bool)
    in_flag = out_flag.groupby(L["driver_number"]).shift(-1, fill_value=False)
    L["PitOutTimeS"] = np.where(out_flag, 1.0, np.nan)
    L["PitInTimeS"] = np.where(in_flag, 1.0, np.nan)

    # weather: nearest sample in time
    W = pd.DataFrame(weather)
    if not W.empty and L["date_start"].notna().any():
        W["date"] = pd.to_datetime(W["date"], utc=True, errors="coerce")
        W = W.dropna(subset=["date"]).sort_values("date")
        left = L[["date_start"]].copy().reset_index().dropna().sort_values("date_start")
        mw = pd.merge_asof(left, W[["date", "track_temperature", "air_temperature", "rainfall"]],
                           left_on="date_start", right_on="date", direction="nearest")
        mw = mw.set_index("index")
        L["TrackTemp"] = mw["track_temperature"]
        L["AirTemp"] = mw["air_temperature"]
        L["Rainfall"] = mw["rainfall"].fillna(0).astype(float) > 0
    else:
        L["TrackTemp"], L["AirTemp"], L["Rainfall"] = np.nan, np.nan, False

    # safety car / VSC laps -> TrackStatus "4"; everything else "1"
    rc = pd.DataFrame(race_control)
    if not rc.empty:
        rc["date"] = pd.to_datetime(rc["date"], utc=True, errors="coerce")
        rc = rc.dropna(subset=["date"])
    L["TrackStatus"] = "1"
    dur = pd.to_timedelta(L["lap_duration"].fillna(0), unit="s")
    lap_end = L["date_start"] + dur
    for a, b in neutral_periods(rc):
        L.loc[(L["date_start"] <= b) & (lap_end >= a), "TrackStatus"] = "4"

    names = pd.DataFrame(drivers)
    if not names.empty:
        names = names.drop_duplicates("driver_number").set_index("driver_number")
        L["Driver"] = L["driver_number"].map(names["name_acronym"]).fillna(L["driver_number"].astype(str))
        L["Team"] = L["driver_number"].map(names["team_name"])
    else:
        L["Driver"], L["Team"] = L["driver_number"].astype(str), None

    out = pd.DataFrame({
        "Driver": L["Driver"], "Team": L["Team"], "LapNumber": L["lap_number"].astype(int),
        "Stint": L["Stint"], "Compound": L["Compound"], "TyreLife": L["TyreLife"],
        "FreshTyre": L["FreshTyre"], "TrackStatus": L["TrackStatus"], "PitInTimeS": L["PitInTimeS"],
        "PitOutTimeS": L["PitOutTimeS"], "IsAccurate": L["lap_duration"].notna(), "Deleted": False,
        "Position": np.nan,          # not fetched (position endpoint is very large); the code drops empty features
        "TrackTemp": L["TrackTemp"], "AirTemp": L["AirTemp"], "Rainfall": L["Rainfall"],
        "LapTimeS": L["lap_duration"].astype(float), "RaceId": race_id, "EventName": event_name,
        "TotalLaps": int(L["lap_number"].max())})
    return out


def load_race(year, rnd):
    """rnd = n-th Grand Prix of the season (1-based), same numbering as the F1 calendar."""
    sessions = get("sessions", year=year, session_name="Race")
    sessions = sorted(sessions, key=lambda s: s["date_start"])
    if not 1 <= rnd <= len(sessions):
        raise ValueError(f"round {rnd} not found: OpenF1 lists {len(sessions)} races for {year}")
    s = sessions[rnd - 1]
    k = s["session_key"]
    name = f'{s.get("country_name", "")} {s.get("circuit_short_name", "")}'.strip()
    return build_race_df(get("laps", session_key=k), get("stints", session_key=k),
                         get("weather", session_key=k), get("drivers", session_key=k),
                         get("race_control", session_key=k), f"{year}_R{rnd:02d}", name)
