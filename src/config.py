"""Central settings. Edit these, not the code."""

YEAR = 2024

# 2024 calendar round numbers (verify against fastf1.get_event_schedule(2024)).
# Mixed circuits on purpose: leave-one-race-out only means something if the
# held-out race is a different track from the training races.
DEFAULT_ROUNDS = [1, 2, 3, 4, 8, 10, 13, 16]   # Bahrain, Saudi, Australia, Japan, Monaco, Spain, Hungary, Italy

COMPOUNDS = {"SOFT": 0, "MEDIUM": 1, "HARD": 2}
CODE_TO_COMPOUND = {v: k for k, v in COMPOUNDS.items()}

ALPHA = 0.10                 # 90% conformal intervals
SLOW_LAP_CUTOFF = 1.07       # drop laps >7% slower than the driver's median clean lap
REF_QUANTILE = 0.15          # driver reference pace = 15th percentile of clean laps

REG_FEATURES = ["TyreLife", "LapFrac", "CompoundCode", "FreshTyre", "TrackTemp", "AirTemp", "Position"]
PIT_FEATURES = ["TyreLife", "LapFrac", "CompoundCode", "Stint", "Position", "PrevDeltaS"]

# ---------------------------------------------------------------------------
# Strategy engine settings (v3). These are ASSUMPTIONS, not fitted values: edit and re-run.
# ---------------------------------------------------------------------------
MAX_STOPS = 2                    # optimizer searches 0, 1 and 2 stops
MIN_STINT_LAPS = 8               # shortest stint the optimizer may propose
REQUIRE_TWO_COMPOUNDS = True     # dry-race rule: at least two different compounds over the race
LIFE_QUANTILE = 0.99             # a stint may not run past this quantile of observed tyre age per compound

# Neutralisation (safety car / virtual safety car). Typical textbook values; tune per circuit.
SC_LAP_FACTOR = 1.40             # SC lap time = reference lap * this
VSC_LAP_FACTOR = 1.30
SC_PIT_LOSS_FACTOR = 0.50        # a stop under SC costs about half of a green-flag stop
VSC_PIT_LOSS_FACTOR = 0.65
SC_WEAR_FACTOR = 0.30            # tyre age advances this fraction of a lap per SC lap (low speed, low load)
VSC_WEAR_FACTOR = 0.40
DEFAULT_SC_DURATION = 4          # laps
DEFAULT_VSC_DURATION = 3
SCENARIO_SC_LAPS = (15, 25, 40)
SCENARIO_VSC_LAP = 25

# Monte Carlo
N_SIM = 5000
TRACK_TEMP_SD = 3.0              # weather variation (deg C) shared by all laps of a simulated race
P_SC_DEFAULT = 0.45              # used only when the dataset has too few races to estimate it
P_VSC_DEFAULT = 0.25

# Undercut / overcut
DIRTY_AIR_LOSS_PER_CAR_S = 0.6   # time lost per car you must queue behind after rejoining (total over the window)
