"""Offline test of the OpenF1 parser using hand-written payloads shaped like the API's JSON.
Field names follow the OpenF1 docs as I know them; if the live API differs, this is where it will show.
Run:  python -m tests.test_openf1_parse"""
import numpy as np
import pandas as pd
from src import config as C
from src.data import make_synthetic
from src.data_openf1 import build_race_df
from src.features import prepare, regression_set, pit_set
from src.models import get_models
from src.evaluate import loro_regression


def fake_payload():
    t0 = pd.Timestamp("2024-03-02 15:00:00", tz="UTC")
    laps = []
    for drv in (1, 44):
        for lap in range(1, 9):
            laps.append(dict(driver_number=drv, lap_number=lap,
                             date_start=(t0 + pd.Timedelta(seconds=95 * (lap - 1))).isoformat(),
                             lap_duration=95.0 + 0.1 * lap + (20 if (drv == 1 and lap == 5) else 0)
                             + (3 if (drv == 1 and lap == 4) else 0),
                             is_pit_out_lap=(drv == 1 and lap == 5)))
    stints = [dict(driver_number=1, stint_number=1, lap_start=1, lap_end=4, compound="SOFT", tyre_age_at_start=0),
              dict(driver_number=1, stint_number=2, lap_start=5, lap_end=8, compound="HARD", tyre_age_at_start=0),
              dict(driver_number=44, stint_number=1, lap_start=1, lap_end=8, compound="MEDIUM", tyre_age_at_start=2)]
    weather = [dict(date=(t0 + pd.Timedelta(minutes=m)).isoformat(), track_temperature=30 + m * 0.1,
                    air_temperature=22.0, rainfall=0) for m in range(0, 15)]
    drivers = [dict(driver_number=1, name_acronym="VER", team_name="Red Bull"),
               dict(driver_number=44, name_acronym="HAM", team_name="Ferrari")]
    rc = [dict(date=(t0 + pd.Timedelta(seconds=95 * 5 + 10)).isoformat(), category="SafetyCar",
               message="SAFETY CAR DEPLOYED", lap_number=6),
          dict(date=(t0 + pd.Timedelta(seconds=95 * 6 + 10)).isoformat(), category="SafetyCar",
               message="SAFETY CAR IN THIS LAP", lap_number=7)]
    return laps, stints, weather, drivers, rc


def main():
    df = build_race_df(*fake_payload(), race_id="TEST_R01", event_name="Test GP")
    ver = df[df.Driver == "VER"].set_index("LapNumber")
    ham = df[df.Driver == "HAM"].set_index("LapNumber")
    assert ver.loc[4, "PitInTimeS"] == 1.0 and ver.loc[5, "PitOutTimeS"] == 1.0   # in-lap precedes out-lap
    assert ver.loc[1, "TyreLife"] == 1 and ver.loc[4, "TyreLife"] == 4 and ver.loc[5, "TyreLife"] == 1
    assert ham.loc[1, "TyreLife"] == 3 and ham.loc[1, "FreshTyre"] == 0.0           # used set: 2 laps old at start
    assert ver.loc[1, "Compound"] == "SOFT" and ver.loc[6, "Compound"] == "HARD"
    assert (df["TrackStatus"] == "4").sum() > 0                                       # safety-car laps flagged
    assert ver.loc[1, "TrackTemp"] > 29 and not ver["Rainfall"].any()
    assert df["TotalLaps"].iloc[0] == 8
    prepare(df)                                                                       # must not crash

    # the pipeline still runs when a feature (Position) is empty, as it will be with OpenF1
    syn = make_synthetic(n_races=5, seed=3)
    syn["Position"] = np.nan
    reg = regression_set(prepare(syn))
    feats = [f for f in C.REG_FEATURES if reg[f].notna().any()]
    assert "Position" not in feats
    res, _ = loro_regression(reg, feats, get_models(), alpha=0.1)
    assert res["MAE"].notna().all()
    print("openf1 parser tests passed")


if __name__ == "__main__":
    main()
