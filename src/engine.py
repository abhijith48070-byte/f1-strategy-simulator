"""Glue: build models + parameters from a lap table, produce a full recommendation, and backtest a decision
using only information available at that lap."""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from sklearn.base import clone
from . import config as C
from .features import prepare, regression_set, pit_loss_samples
from .models import get_models
from .bayes import HierarchicalTyreModel
from .lapmodels import BayesLapModel, GBRLapModel
from .planner import Planner, Constraints, best_per_family, family, strategy_label
from .optimizer import scenario_table, reactive_decision, default_scenarios
from .montecarlo import simulate, summarize
from .scenarios import neutral_stats
from .strategy_explain import explain


@dataclass
class Context:
    df: pd.DataFrame
    reg: pd.DataFrame
    features: list
    bayes: HierarchicalTyreModel
    models: dict                     # name -> LapModel
    pit_losses: np.ndarray
    stats: dict
    max_life: dict
    allowed: tuple
    races: list = field(default_factory=list)

    @property
    def pit_loss(self):
        return float(np.median(self.pit_losses))


def build_context(raw, bayes_group="RaceId", seed=0, gbr=True, bayes_iter=2500, ref_mode="fe"):
    df = prepare(raw, ref_mode=ref_mode)
    reg = regression_set(df)
    features = [f for f in C.REG_FEATURES if reg[f].notna().any()]
    races = sorted(reg["RaceId"].unique())
    counts = reg.groupby("CompoundCode").size()
    allowed = tuple(int(c) for c in (0, 1, 2) if counts.get(c, 0) >= 200)
    max_life = {int(c): float(reg.loc[reg["CompoundCode"] == c, "TyreLife"].quantile(C.LIFE_QUANTILE)) for c in allowed}
    bayes = HierarchicalTyreModel(n_iter=bayes_iter, burn=int(bayes_iter * 0.3), seed=seed).fit(reg, group_col=bayes_group)
    models = {"bayes": BayesLapModel(bayes, max_life=max_life)}
    if gbr and len(races) >= 3:
        cal = races[-1]
        tr = reg[reg["RaceId"] != cal]
        m = clone(get_models()["hist_gbr"]).fit(tr[features], tr["DeltaS"])
        c = reg[reg["RaceId"] == cal]
        resid = (c["DeltaS"] - m.predict(c[features])).to_numpy()
        models["hist_gbr"] = GBRLapModel(m, features, resid, max_life, temp_sd=C.TRACK_TEMP_SD)
    losses = pit_loss_samples(df)["loss_s"].to_numpy()
    if len(losses) < 5:
        losses = np.array([22.0])
    return Context(df, reg, features, bayes, models, losses, neutral_stats(df), max_life, allowed, races)


def make_state(ctx, model_key, race_id, driver, lap, compound, tyre_life, position=10.0, used=None):
    """Race state for a driver at the START of `lap`. Level (reference pace) uses only that driver's laps before `lap`
    (or the whole race if there are none) and is only needed for absolute times, never for the ranking."""
    race = ctx.reg[ctx.reg["RaceId"] == race_id]
    full = ctx.df[ctx.df["RaceId"] == race_id]
    L = int(full["TotalLaps"].iloc[0])
    mine = race[(race["Driver"] == driver) & (race["LapNumber"] < lap)]
    ref_laps = mine if len(mine) >= 3 else race
    ref = ctx.models[model_key].reference_pace(ref_laps)
    c0 = compound if isinstance(compound, int) else C.COMPOUNDS[compound]
    return dict(lap=int(lap), total_laps=L, tyre_life=float(tyre_life), compound=c0, fresh=0.0,
                track_temp=float(race["TrackTemp"].median()) if race["TrackTemp"].notna().any() else 30.0,
                air_temp=float(race["AirTemp"].median()) if race["AirTemp"].notna().any() else 22.0,
                position=float(position), used=set(used) if used else {c0}, ref_lap_s=float(ref),
                elapsed_s=float((lap - 1) * ref))


@dataclass
class Recommendation:
    model: str
    state: dict
    pit_loss: float
    table: pd.DataFrame          # every legal strategy, deterministic mean
    shortlist: pd.DataFrame      # best pit laps per compound sequence, recommended first
    summary: pd.DataFrame        # Monte Carlo summary, same order
    sim: object
    scenarios: pd.DataFrame
    reactive: list
    explanation: dict
    objective: str

    @property
    def best(self):
        return self.shortlist.iloc[0]


def recommend(ctx, model_key, state, n_sim=C.N_SIM, k=6, objective="mean", constraints=None, seed=0):
    """objective: 'mean' (fastest on average) or 'p90' (best 90th-percentile finish = risk-averse)."""
    model = ctx.models[model_key]
    cons = constraints or Constraints(allowed=ctx.allowed)
    pit_loss = ctx.pit_loss
    table = Planner(model, state, pit_loss, None, cons).enumerate()
    fam = best_per_family(table, k)
    plans = [(r.compounds, r.pit_laps, r.strategy) for r in fam.itertuples()]
    res = simulate(model, state, plans, ctx.pit_losses, n_sim=n_sim, neutral="random", stats=ctx.stats, seed=seed)
    summ = summarize(res)
    col = {"mean": "mean_s", "p90": "p90_s"}[objective]
    order = summ[col].argsort(kind="stable").to_numpy()
    fam, summ = fam.iloc[order].reset_index(drop=True), summ.iloc[order].reset_index(drop=True)
    res.totals, res.labels = res.totals[:, order], [res.labels[i] for i in order]
    scen = scenario_table(model, state, pit_loss, None, cons)
    reactive = []
    for name, nt in default_scenarios(state["lap"], state["total_laps"]):
        if nt is not None:
            try:
                reactive.append((name, reactive_decision(model, state, nt, pit_loss, cons)))
            except ValueError:
                pass
    expl = explain(model, state, pit_loss, table, fam, res, summ, scen, cons, neutral_share=res.neutral_share)
    if model_key == "hist_gbr":
        expl["risks"].insert(0, "WARNING: hist_gbr is a tree model trained on observed (tyre age, lap) combinations; it has never "
                                "seen e.g. a hard tyre early in the race or a stint longer than the data, so counterfactual "
                                "strategy prices from it can be arbitrary. Use the Bayesian model for planning.")
    return Recommendation(model_key, state, pit_loss, table, fam, summ, res, scen, reactive, expl, objective)


# ----------------------------------------------------------------------------------------------------------
# Backtest with no future information
# ----------------------------------------------------------------------------------------------------------
def state_from_race(race_df, driver, lap, total_laps=None):
    """Race state at the start of `lap`, built ONLY from rows with LapNumber <= lap (tyre compound/age at the start
    of the current lap are known) and lap times of laps < lap. Rows after `lap` are never read."""
    d = race_df[(race_df["Driver"] == driver) & (race_df["LapNumber"] <= lap)].sort_values("LapNumber")
    now = d[d["LapNumber"] == lap].iloc[0]
    past = d[d["LapNumber"] < lap]
    ok = (past["TrackStatus"].astype(str) == "1") & past["PitInTimeS"].isna() & past["PitOutTimeS"].isna() \
        & past["LapTimeS"].notna() & (past["LapNumber"] > 1)
    past = past[ok]
    if len(past):
        past = past[past["LapTimeS"] <= C.SLOW_LAP_CUTOFF * past["LapTimeS"].median()]
    used = set(C.COMPOUNDS[c] for c in d["Compound"].dropna().unique() if c in C.COMPOUNDS)
    return dict(lap=int(lap), compound=C.COMPOUNDS[now["Compound"]], tyre_life=float(now["TyreLife"]),
                fresh=float(now["FreshTyre"]) if pd.notna(now["FreshTyre"]) else 0.0, used=used,
                past_laps=past.assign(CompoundCode=past["Compound"].map(C.COMPOUNDS), LapNumber=past["LapNumber"]),
                position=float(now["Position"]) if pd.notna(now.get("Position", np.nan)) else 10.0)


def actual_plan(race_df, driver, lap):
    """What the driver actually did from `lap` on (compounds per stint, stop laps). Used ONLY to score the decision."""
    d = race_df[(race_df["Driver"] == driver) & (race_df["LapNumber"] >= lap)].sort_values("LapNumber")
    comps = [C.COMPOUNDS[d["Compound"].iloc[0]]]
    pits = []
    for _, r in d.iterrows():
        if pd.notna(r["PitInTimeS"]) and r["LapNumber"] < d["LapNumber"].max():
            pits.append(int(r["LapNumber"]))
            nxt = d[d["LapNumber"] == r["LapNumber"] + 1]
            comps.append(C.COMPOUNDS[nxt["Compound"].iloc[0]] if len(nxt) and nxt["Compound"].iloc[0] in C.COMPOUNDS else comps[-1])
    return comps, pits


def backtest_decision(raw, race_id, driver, lap, model_key="bayes", n_sim=1500, seed=0):
    """Train on every OTHER race, build the state from information available at `lap`, recommend, then compare with
    what the driver actually did. The score is the MODEL's estimate of the time difference (the actual counterfactual
    cannot be observed), so it measures decision consistency, not guaranteed time gained."""
    race_raw = raw[raw["RaceId"] == race_id]
    ctx = build_context(raw[raw["RaceId"] != race_id], gbr=(model_key != "bayes"), seed=seed)
    s = state_from_race(race_raw, driver, lap)
    model = ctx.models[model_key]
    L = int(race_raw["TotalLaps"].iloc[0])
    ref = model.reference_pace(s.pop("past_laps")) if model_key == "bayes" else float(ctx.df["DriverRef"].median())
    state = dict(s, total_laps=L, track_temp=float(race_raw["TrackTemp"].iloc[:lap].median()) if race_raw["TrackTemp"].notna().any() else 30.0,
                 air_temp=float(race_raw["AirTemp"].iloc[:lap].median()) if race_raw["AirTemp"].notna().any() else 22.0,
                 ref_lap_s=ref, elapsed_s=(lap - 1) * ref)
    rec = recommend(ctx, model_key, state, n_sim=n_sim, seed=seed)
    comps, pits = actual_plan(race_raw, driver, lap)
    pl = Planner(model, state, ctx.pit_loss, None, Constraints(allowed=ctx.allowed))
    actual = pl.evaluate(comps, pits)
    best = rec.best
    return dict(recommended=best["strategy"], actual=strategy_label(comps, pits),
                model_gain_vs_actual_s=actual - pl.evaluate(best["compounds"], best["pit_laps"]), state=state, rec=rec)


def describe_reactive(name, r):
    """One readable line for a reactive_decision result."""
    normal = f"pit lap {r['normal_first_pit']}" if r["normal_first_pit"] is not None else "no further stop"
    g = r["gain_vs_staying_out_s"]
    if r["decision"].startswith("Pit"):
        extra = f"stopping under it is worth {g:.1f} s vs staying out" if g == g else ""
    else:
        extra = f"a stop under it would cost {-g:.1f} s" if g == g else ""
    return f"{name}: plan without it = {normal}; with it = {r['decision']}" + (f" ({extra})" if extra else "")
