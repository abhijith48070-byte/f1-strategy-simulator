"""Tests for the strategy engine.  Run:  python -m tests.test_strategy

Most tests use synthetic races whose physics are known exactly (src/data.py: degradation SOFT 0.09 / MEDIUM 0.05 /
HARD 0.03 s per lap of tyre age, fuel -0.035 s per lap, pit loss about 19-21 s), so a result can be checked against the
truth and against an independent brute-force implementation.
"""
import numpy as np
import pandas as pd
from src import config as C
from src.data import make_synthetic
from src.features import prepare, regression_set, pit_loss_samples
from src.bayes import HierarchicalTyreModel
from src.lapmodels import BayesLapModel, LapModel
from src.planner import Planner, Constraints, best_per_family
from src.optimizer import scenario_table, reactive_decision
from src.scenarios import Neutral, neutral_stats, NeutralArrays
from src.montecarlo import simulate, summarize, compare
from src.undercut import analyse
from src.engine import build_context, make_state, recommend, state_from_race, backtest_decision

DEG = {0: 0.09, 1: 0.05, 2: 0.03}
FUEL = -0.035


class TruthModel(LapModel):
    """Exact synthetic physics, no noise: used to check the optimizer against brute force."""
    name = "truth"
    max_life = {0: 30, 1: 30, 2: 60}

    def mean(self, comp, age, lap, total, fresh, state):
        return DEG[comp] * np.asarray(age, float) + FUEL * np.asarray(lap, float)

    def draw_world(self, n, L, rng, state):
        return dict(eps=np.zeros((n, L + 2)))

    def sample(self, world, comp, age, lap, total, fresh, state):
        age = np.asarray(age, float)
        age = age[None, :] if age.ndim == 1 else age
        return DEG[comp] * age + FUEL * np.asarray(lap, float)[None, :] + world["eps"][:, np.asarray(lap, int)]

    def reference_pace(self, laps):
        return 90.0


STATE = dict(lap=10, total_laps=58, tyre_life=10.0, compound=1, fresh=0.0, track_temp=35, air_temp=25, position=8,
             ref_lap_s=90.0, used={1})
LOSS = 21.0


def brute_force_best_one_stop(model, state, loss):
    """Independent, deliberately naive implementation: lap by lap, no tables."""
    n, L, best = state["lap"], state["total_laps"], (1e9, None)
    for c1 in (0, 1, 2):
        if c1 == state["compound"]:
            continue                                  # two-compound rule: one stop must change compound
        for p in range(n, L - C.MIN_STINT_LAPS + 1):
            t, age = 0.0, state["tyre_life"]
            ok = True
            for lap in range(n, p + 1):
                t += DEG[1] * age + FUEL * lap
                age += 1
            if age - 1 > model.max_life[1]:
                ok = False
            age = 1
            for lap in range(p + 1, L + 1):
                t += DEG[c1] * age + FUEL * lap
                age += 1
            if age - 1 > model.max_life[c1]:
                ok = False
            if ok and t + loss < best[0]:
                best = (t + loss, (c1, p))
    return best


def test_bayes_recovers_physics_and_v2_target_was_biased():
    raw = make_synthetic(n_races=6, seed=1)
    reg = regression_set(prepare(raw, ref_mode="full"))
    m = HierarchicalTyreModel().fit(reg)
    slopes = m.mean_theta()[2:5]
    assert np.all(np.abs(slopes - [0.09, 0.05, 0.03]) < 0.012), slopes
    summ = m.degradation_summary()
    assert (summ["posterior_sd"] > 0).all()
    # the v2 target (lap - 15th-percentile reference) underestimates HARD degradation; documented finding
    from src.bayes import design_matrix
    X = design_matrix(reg.CompoundCode, reg.TyreLife, reg.LapNumber)
    th = np.linalg.lstsq(np.c_[X, np.ones(len(X))], reg.DeltaS.values, rcond=None)[0]
    assert th[4] < 0.015, th
    # the fixed-effect reference removes that bias
    reg_fe = regression_set(prepare(raw, ref_mode="fe"))
    X = design_matrix(reg_fe.CompoundCode, reg_fe.TyreLife, reg_fe.LapNumber)
    th = np.linalg.lstsq(np.c_[X, np.ones(len(X))], reg_fe.DeltaS.values, rcond=None)[0]
    assert abs(th[4] - 0.03) < 0.006, th
    return slopes


def test_early_reference_uses_no_future_laps():
    raw = make_synthetic(n_races=4, seed=2)
    mod = raw.copy()
    late = mod["LapNumber"] >= 30
    mod.loc[late, "LapTimeS"] += np.random.default_rng(0).normal(0, 3, late.sum()) + 2
    a, b = prepare(raw, ref_mode="early"), prepare(mod, ref_mode="early")
    k = a["LapNumber"] < 30
    assert np.allclose(a.loc[k, "DeltaS"].fillna(-999), b.loc[k, "DeltaS"].fillna(-999)), "early mode leaks future laps"
    assert (a.loc[k, "Clean"] == b.loc[k, "Clean"]).all()
    # the v2 'full' reference does read the future (this is why the test has power)
    af, bf = prepare(raw, ref_mode="full"), prepare(mod, ref_mode="full")
    assert not np.allclose(af.loc[k, "DeltaS"].fillna(-999), bf.loc[k, "DeltaS"].fillna(-999))


def test_optimizer_matches_brute_force_and_is_sorted():
    m = TruthModel()
    tab = Planner(m, STATE, LOSS, None).enumerate()
    assert tab["expected_delta_s"].is_monotonic_increasing
    one = tab[tab["n_stops"] == 1].iloc[0]
    bf_t, (bf_c, bf_p) = brute_force_best_one_stop(m, STATE, LOSS)
    assert one["compounds"][1] == bf_c and one["pit_laps"][0] == bf_p, (one["strategy"], bf_c, bf_p)
    assert abs(one["expected_delta_s"] - bf_t) < 1e-6
    # no strategy violates the two-compound rule, the minimum stint or the tyre-life limit
    assert all(len(set(c) | {1}) >= 2 for c in tab["compounds"])
    for _, r in tab.iterrows():
        bounds = [STATE["lap"] - 1] + list(r["pit_laps"]) + [STATE["total_laps"]]
        stints = np.diff(bounds)
        assert all(s >= C.MIN_STINT_LAPS for s in stints[1:])
    # every row re-prices to the same number
    pl = Planner(m, STATE, LOSS, None)
    for _, r in tab.sample(40, random_state=0).iterrows():
        assert abs(pl.evaluate(r["compounds"], r["pit_laps"]) - r["expected_delta_s"]) < 1e-6
    assert len(best_per_family(tab, 5)) == 5


def test_safety_car_pit_loss_wear_and_reaction():
    m = TruthModel()
    nt = Neutral("SC", 20, 4)
    pl = Planner(m, STATE, LOSS, nt)
    strat = ([1, 2], [22])
    with_loss = pl.evaluate(*strat)
    pl.pit_loss = 0.0
    assert abs((with_loss - pl.evaluate(*strat)) - LOSS * C.SC_PIT_LOSS_FACTOR) < 1e-9     # reduced pit loss in window
    # SC laps cost everyone extra time: same strategy is much slower under an SC than without one
    extra = Planner(m, STATE, LOSS, nt).evaluate([1, 2], [30]) - Planner(m, STATE, LOSS, None).evaluate([1, 2], [30])
    assert extra > 0.8 * 4 * 90.0 * (C.SC_LAP_FACTOR - 1), extra
    pl.pit_loss = LOSS
    # a stop outside the window pays the full pit loss
    pl.pit_loss = LOSS
    a = NeutralArrays.single(58, 90.0, nt)
    assert a.pit_factor[0, 22] == C.SC_PIT_LOSS_FACTOR and a.pit_factor[0, 25] == 1.0
    assert abs(a.lap_extra[0, 21] - 90.0 * (C.SC_LAP_FACTOR - 1)) < 1e-9
    # tyres wear less on SC laps: effective age after the window is lower than the lap count
    assert a.age(1.0, 11, np.arange(11, 31))[0, -1] < 20
    # reaction: SC at lap 12 while the green plan stops on ~lap 22 -> stop under the SC and gain time
    r = reactive_decision(m, STATE, Neutral("SC", 12, 3), LOSS)
    assert r["decision"].startswith("Pit") and r["gain_vs_staying_out_s"] > 3, r
    # a VSC saves less than an SC (higher pit-loss factor)
    s_sc = scenario_table(m, STATE, LOSS, [("none", None), ("sc", Neutral("SC", 15, 4)), ("vsc", Neutral("VSC", 15, 4))])
    assert s_sc.loc[1, "adaptation_gain_s"] > s_sc.loc[2, "adaptation_gain_s"] > 0
    # scenario that cannot be reacted to (after the plan's stop) leaves the plan unchanged
    late = scenario_table(m, STATE, LOSS, [("none", None), ("late", Neutral("SC", 50, 3))])
    assert abs(late.loc[1, "adaptation_gain_s"]) < 1.5


def test_planner_rejects_impossible_constraints():
    try:
        Planner(TruthModel(), dict(STATE, tyre_life=40.0), LOSS, None, Constraints(max_stops=0)).enumerate()
    except ValueError:
        return
    raise AssertionError("expected ValueError (tyres past their life limit, no stops allowed)")


def test_monte_carlo_common_random_numbers_and_calibration():
    raw = make_synthetic(n_races=6, seed=1)
    reg = regression_set(prepare(raw, ref_mode="fe"))
    m = BayesLapModel(HierarchicalTyreModel().fit(reg), max_life={0: 30, 1: 30, 2: 60})
    ls = pit_loss_samples(prepare(raw))["loss_s"].to_numpy()
    tab = Planner(m, STATE, float(np.median(ls)), None).enumerate()
    fam = best_per_family(tab, 4)
    plans = [(r.compounds, r.pit_laps, r.strategy) for r in fam.itertuples()]
    plans.append(plans[0])                                                       # duplicate of the best
    res = simulate(m, STATE, plans, ls, n_sim=4000, neutral=None, seed=3)
    assert np.allclose(res.totals[:, 0], res.totals[:, -1])                      # identical plan -> identical result
    summ = summarize(res.__class__(res.labels[:-1], res.totals[:, :-1], res.base_s, 0.0))
    assert abs(summ["p_fastest"].sum() - 1.0) < 1e-9
    # without neutralisation the simulated mean matches the deterministic planner (posterior mean = planner's model)
    assert abs(res.totals[:, 0].mean() - fam.iloc[0]["expected_delta_s"]) < 3.0
    assert summ.iloc[0]["p_fastest"] > 0.5                                       # the optimizer's pick usually wins
    c = compare(res, 0, 1)
    assert c["mean_advantage_s"] > 0 and c["p05"] <= c["mean_advantage_s"] <= c["p95"]
    # random SC/VSC worlds: some races neutralised, spread widens, nothing crashes
    r2 = simulate(m, STATE, plans[:3], ls, n_sim=2000, neutral="random",
                  stats=dict(p_sc=0.3, p_vsc=0.2, starts=[], durations=[]), seed=4)
    assert 0.4 < r2.neutral_share < 0.6
    assert r2.totals[:, 0].std() > res.totals[:, 0].std()


def test_undercut_overcut():
    m = TruthModel()
    st = dict(STATE, lap=18, tyre_life=18.0, compound=0, used={0})
    car = dict(compound=0, tyre_life=18.0, new_compound=2)
    ls = np.array([21.0])
    t, s = analyse(m, st, car, car, 1.0, ls, b_pit_lap=22, n_sim=200)
    same = t[t["a_pit_lap"] == 22].iloc[0]
    assert abs(same["net_gain_s"]) < 1e-6                                       # identical cars, same lap: no gain
    assert t[t["kind"] == "undercut"]["net_gain_s"].min() > 0                   # old soft tyres: stopping earlier gains
    assert t[t["kind"] == "overcut"]["net_gain_s"].max() < 0
    assert s["undercut"]["detected"] and not s["overcut"]["detected"]
    _, s_far = analyse(m, st, car, car, 30.0, ls, b_pit_lap=22, n_sim=200)       # 30 s back: nothing works
    assert not s_far["undercut"]["detected"]
    _, s_traffic = analyse(m, st, car, car, 1.0, ls, b_pit_lap=22, n_sim=200, traffic_cars=10)
    assert s_traffic["undercut"]["margin_s"] < s["undercut"]["margin_s"]        # traffic reduces the margin
    assert 0 <= t["p_success"].min() and t["p_success"].max() <= 1


def test_neutral_stats():
    st = neutral_stats(prepare(make_synthetic(n_races=6, seed=1)))
    assert 0 < st["p_sc"] + st["p_vsc"] <= 1.0 and len(st["starts"]) == len(st["durations"]) > 0
    assert neutral_stats(prepare(make_synthetic(n_races=3, seed=1)))["source"] == "config defaults"


def test_state_uses_no_future_information_and_backtest_runs():
    raw = make_synthetic(n_races=6, seed=1)
    race = raw[raw["RaceId"] == "SYN_R05"]
    s1 = state_from_race(race, "D02", 25)
    mod = race.copy()
    late = (mod["LapNumber"] > 25)
    mod.loc[late, ["LapTimeS", "TyreLife"]] = 9999.0
    mod.loc[late, "Compound"] = "SOFT"
    s2 = state_from_race(mod, "D02", 25)
    p1, p2 = s1.pop("past_laps"), s2.pop("past_laps")
    assert s1 == s2 and p1.equals(p2) and p1["LapNumber"].max() < 25
    bt = backtest_decision(raw, "SYN_R05", "D02", 10, "bayes", n_sim=500)          # before the driver's stop on lap 18
    assert np.isfinite(bt["model_gain_vs_actual_s"]) and "→" in bt["recommended"] and "pit L18" in bt["actual"]
    assert bt["state"]["lap"] == 10
    late = backtest_decision(raw, "SYN_R05", "D02", 25, "bayes", n_sim=500)        # already on its final tyres
    assert "no stop" in late["recommended"] and late["model_gain_vs_actual_s"] == 0.0


def test_end_to_end_recommendation_and_explanation():
    ctx = build_context(make_synthetic(n_races=6, seed=1), gbr=False)
    st = make_state(ctx, "bayes", "SYN_R05", "D03", 10, "MEDIUM", 10)
    rec = recommend(ctx, "bayes", st, n_sim=1500)
    assert rec.best["n_stops"] >= 1 and len(rec.summary) == len(rec.shortlist)
    assert rec.summary.iloc[0]["mean_s"] == rec.summary["mean_s"].min()
    for k in ("why", "changes", "risks"):
        assert rec.explanation[k], k
    assert any("SC lap" in line for line in rec.explanation["changes"])
    rp = recommend(ctx, "bayes", st, n_sim=1500, objective="p90")
    assert rp.summary.iloc[0]["p90_s"] == rp.summary["p90_s"].min()
    assert len(rec.reactive) >= 1


def main():
    slopes = test_bayes_recovers_physics_and_v2_target_was_biased()
    for f in (test_early_reference_uses_no_future_laps, test_optimizer_matches_brute_force_and_is_sorted,
              test_safety_car_pit_loss_wear_and_reaction, test_planner_rejects_impossible_constraints,
              test_monte_carlo_common_random_numbers_and_calibration, test_undercut_overcut, test_neutral_stats,
              test_state_uses_no_future_information_and_backtest_runs, test_end_to_end_recommendation_and_explanation):
        f()
        print("ok  ", f.__name__)
    print("all strategy tests passed | Bayes slopes S/M/H:", slopes.round(3))


if __name__ == "__main__":
    main()
