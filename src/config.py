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
