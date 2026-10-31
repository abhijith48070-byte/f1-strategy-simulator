"""Optimizer layer: best strategy, safety-car scenario table, and the reactive 'SC just came out, pit now?' call."""
import numpy as np
import pandas as pd
from . import config as C
from .planner import Planner, Constraints, best_per_family, fmt_time
from .scenarios import Neutral, default_duration


def first_stop(row):
    return row["pit_laps"][0] if len(row["pit_laps"]) else None


def action_now(row, lap):
    p = first_stop(row)
    if p is None:
        return "Stay out (no stop)"
    return "Pit now" if p == lap else f"Stay out, pit lap {p}"


def default_scenarios(lap, total_laps):
    """No SC, SC on laps 15/25/40 (those still in the future), one VSC. All from config."""
    out = [("No neutralisation", None)]
    for l in C.SCENARIO_SC_LAPS:
        if lap < l < total_laps - 2:
            out.append((f"SC lap {l}", Neutral("SC", l, default_duration("SC"))))
    if lap < C.SCENARIO_VSC_LAP < total_laps - 2:
        out.append((f"VSC lap {C.SCENARIO_VSC_LAP}", Neutral("VSC", C.SCENARIO_VSC_LAP, default_duration("VSC"))))
    return out


def scenario_table(model, state, pit_loss, scenarios=None, constraints=None):
    """For each scenario assumed KNOWN from the start: best strategy, when to stop, and what adapting is worth
    (baseline plan priced under the scenario minus the scenario's own best)."""
    scenarios = scenarios or default_scenarios(state["lap"], state["total_laps"])
    base_planner = Planner(model, state, pit_loss, None, constraints)
    base_tab = base_planner.enumerate()
    base_best = base_tab.iloc[0]
    rows = []
    for name, nt in scenarios:
        pl = base_planner if nt is None else Planner(model, state, pit_loss, nt, constraints)
        tab = base_tab if nt is None else pl.enumerate()
        best = tab.iloc[0]
        keep_plan = pl.evaluate(base_best["compounds"], base_best["pit_laps"])
        rows.append(dict(scenario=name, best_strategy=best["strategy"], first_pit_lap=first_stop(best),
                         pit_time_s=best["pit_time_s"], race_time=fmt_time(best["race_time_s"]),
                         race_time_s=best["race_time_s"],
                         adaptation_gain_s=keep_plan - best["expected_delta_s"],
                         changes_plan=(best["family"] != base_best["family"]
                                       or first_stop(best) != first_stop(base_best))))
    return pd.DataFrame(rows)


def state_on_plan(model, state, pit_loss, lap, constraints=None):
    """Race state at `lap` if the green-flag optimal plan from `state` is followed until then (stops already
    made are in the past: new compound, age, compounds used)."""
    plan = Planner(model, state, pit_loss, None, constraints).enumerate().iloc[0]
    comps, pits = plan["compounds"], plan["pit_laps"]
    n, ref = int(state["lap"]), float(state.get("ref_lap_s", 0.0))
    st = dict(state)
    done = [p for p in pits if p < lap]                      # stops completed before `lap` starts
    st["lap"] = lap
    st["compound"] = comps[len(done)]
    st["used"] = set(state.get("used", {state["compound"]})) | set(comps[: len(done) + 1])
    if done:
        st["tyre_life"], st["fresh"] = float(lap - done[-1]), 1.0
    else:
        st["tyre_life"] = state["tyre_life"] + (lap - n)
    st["elapsed_s"] = state.get("elapsed_s", (n - 1) * ref) + (lap - n) * ref
    return st


def reactive_decision(model, state, neutral, pit_loss, constraints=None):
    """The race is at lap `neutral.start` (SC/VSC has just been called). Using only information available now
    (tyre age advanced by the green laps since `state`), should the car stop under the neutralisation?

    Reported: the plan that would have been followed without it, the best plan now, the gain of stopping under the
    neutralisation vs the best plan that stays out until after it, and the cost/benefit of stopping on the FIRST
    neutralised lap vs the best lap inside the window. The model slightly prefers the last SC lap (tyres wear a
    little on SC laps); pit-lane closure and field compression are not modelled, so treat that as a tie.
    """
    n, s = int(state["lap"]), neutral.start
    if not n <= s < state["total_laps"] - 1:
        raise ValueError("neutralisation must start after the current lap and before the end")
    ref = float(state.get("ref_lap_s", 0.0))
    st = state_on_plan(model, state, pit_loss, s, constraints)
    normal = Planner(model, st, pit_loss, None, constraints).enumerate()
    under = Planner(model, st, pit_loss, neutral, constraints).enumerate()
    end = neutral.start + neutral.duration - 1
    fp = [p[0] if len(p) else None for p in under["pit_laps"]]
    in_window = under[[p is not None and s <= p <= end for p in fp]]
    outside = under[[p is None or p > end for p in fp]]
    first_lap = under[[p == s for p in fp]]
    best, nb = under.iloc[0], normal.iloc[0]
    p1 = first_stop(best)
    gain_window = (float(outside.iloc[0]["expected_delta_s"] - in_window.iloc[0]["expected_delta_s"])
                   if len(in_window) and len(outside) else np.nan)
    if p1 is not None and s <= p1 <= end:
        decision = "Pit now" if p1 == s else f"Pit under the {neutral.kind} (laps {s}-{end})"
    else:
        decision = action_now(best, s)
    return dict(neutral=neutral, normal_plan=nb["strategy"], normal_first_pit=first_stop(nb),
                reactive_plan=best["strategy"], reactive_first_pit=p1, decision=decision,
                gain_vs_staying_out_s=gain_window,
                cost_of_first_lap_vs_best_in_window_s=(float(first_lap.iloc[0]["expected_delta_s"]
                                                             - in_window.iloc[0]["expected_delta_s"])
                                                       if len(first_lap) and len(in_window) else np.nan),
                normal_pit_loss_s=pit_loss,
                neutral_pit_loss_s=pit_loss * (C.SC_PIT_LOSS_FACTOR if neutral.kind == "SC" else C.VSC_PIT_LOSS_FACTOR))
