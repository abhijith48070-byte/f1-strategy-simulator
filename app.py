"""Streamlit dashboard:  streamlit run app.py

Race overview -> recommendation -> simulation charts -> safety-car scenarios -> undercut/overcut -> explanation.
All numbers come from src/ (tested separately); this file only lays them out.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from src import config as C
from src.data import make_synthetic
from src.engine import build_context, make_state, recommend, describe_reactive
from src.planner import NAMES, fmt_time
from src.undercut import analyse
from src import plots


@st.cache_resource(show_spinner="Fitting the Bayesian and tree models ...")
def load_context(source, csv_path):
    raw = make_synthetic(n_races=6, seed=1) if source == "Synthetic demo" else pd.read_csv(csv_path)
    return build_context(raw)


@st.cache_resource(show_spinner="Simulating races ...")
def run_recommendation(_ctx, key, model, race, driver, lap, compound, tyre_life, objective, n_sim):
    state = make_state(_ctx, model, race, driver, lap, compound, tyre_life)
    return recommend(_ctx, model, state, n_sim=n_sim, objective=objective)


def main():
    st.set_page_config(page_title="F1 Strategy Engine", layout="wide")
    st.title("F1 strategy: uncertainty-aware pit-stop optimizer")
    st.caption("Replays a race state and compares strategies. It is not a live system: nothing here receives live timing, "
               "and rivals' strategies, traffic and rain are not modelled (see the Explain tab).")

    sb = st.sidebar
    sb.header("Data and model")
    source = sb.radio("Data", ["Synthetic demo", "CSV from run_experiment.py"], index=0)
    csv_path = sb.text_input("CSV path", "data/laps_openf1.csv") if source != "Synthetic demo" else ""
    if source != "Synthetic demo" and not Path(csv_path).exists():
        st.error(f"{csv_path} not found. Run `python run_experiment.py` first, or choose the synthetic demo.")
        st.stop()
    ctx = load_context(source, csv_path)
    model_label = sb.radio("Lap-time model", ["Bayesian (recommended)", "Gradient boosting"], index=0)
    model = "bayes" if model_label.startswith("Bayesian") else "hist_gbr"
    if model == "hist_gbr" and "hist_gbr" not in ctx.models:
        st.warning("Gradient boosting needs at least 3 races; using the Bayesian model.")
        model = "bayes"
    if model == "hist_gbr":
        st.warning("Tree models cannot price stints they have never seen (e.g. a hard tyre early in the race), so "
                   "counterfactual strategy prices from them can be arbitrary.")

    sb.header("Race state")
    races = ctx.races
    race = sb.selectbox("Race", races, index=len(races) - 1)
    race_df = ctx.reg[ctx.reg["RaceId"] == race]
    total_laps = int(ctx.df[ctx.df["RaceId"] == race]["TotalLaps"].iloc[0])
    event = str(ctx.df[ctx.df["RaceId"] == race]["EventName"].iloc[0])
    drivers = sorted(race_df["Driver"].unique())
    driver = sb.selectbox("Driver", drivers, index=0)
    lap = int(sb.slider("Current lap", 2, max(3, total_laps - C.MIN_STINT_LAPS - 1), min(10, total_laps // 3)))
    names = [NAMES[c] for c in ctx.allowed]
    compound = sb.selectbox("Current tyre compound", names, index=names.index("MEDIUM") if "MEDIUM" in names else 0)
    tyre_life = float(sb.number_input("Tyre age (laps)", min_value=1.0, max_value=80.0, value=float(lap), step=1.0))
    objective = "mean" if sb.radio("Optimise for", ["Average time", "Risk-averse (90th percentile)"], index=0) == "Average time" else "p90"
    n_sim = int(sb.select_slider("Simulated races", options=[1000, 2000, 5000, 10000], value=2000)) if hasattr(sb, "select_slider") else 2000

    key = f"{source}|{csv_path}|{race}"
    rec = run_recommendation(ctx, key, model, race, driver, lap, compound, tyre_life, objective, n_sim)
    state, best = rec.state, rec.best

    st.subheader("Race overview")
    c = st.columns(5)
    c[0].metric("Circuit", event)
    c[1].metric("Driver", driver)
    c[2].metric("Lap", f"{lap} / {total_laps}")
    c[3].metric("Tyre", compound.title())
    c[4].metric("Tyre age", f"{tyre_life:.0f} laps")

    st.subheader("Strategy recommendation")
    top = rec.summary.iloc[0]
    from src.montecarlo import compare
    cmp_ = compare(rec.sim, 0, 1) if len(rec.summary) > 1 else None
    c = st.columns(4)
    c[0].metric("Recommended", best["strategy"])
    c[1].metric("Predicted finish", top["mean"], help="Mean over simulated races, including random safety cars. "
                                                      "Absolute level is approximate; ranking is what matters.")
    c[2].metric("Advantage over runner-up", f"{cmp_['mean_advantage_s']:.1f} s" if cmp_ else "n/a",
                f"90% range {cmp_['p05']:.1f} to {cmp_['p95']:.1f} s" if cmp_ else None)
    c[3].metric("P(fastest of compared)", f"{top['p_fastest']:.0%}")
    st.dataframe(rec.summary[["strategy", "mean", "p10", "p90", "mean_regret_s", "p_fastest"]]
                 .rename(columns={"mean": "average", "p10": "10th pct", "p90": "90th pct", "mean_regret_s": "avg regret (s)",
                                  "p_fastest": "P(fastest)"}).round(3), hide_index=True, width="stretch")

    t_cmp, t_deg, t_sc, t_win, t_uc, t_why = st.tabs(
        ["Strategy comparison", "Tyre degradation", "Safety car", "Pit window", "Undercut / overcut", "Explain"])

    with t_cmp:
        st.pyplot(plots.strategy_comparison(rec))
        st.caption("Distribution of the time difference to the recommended strategy in the same simulated races:")
        st.pyplot(plots.paired_distributions(rec))
        st.caption("The strategy with the best average is not always the one most likely to be fastest: compare the average "
                   "and P(fastest) columns. P(fastest) is among the strategies shown, not a race-win probability.")

    with t_deg:
        st.pyplot(plots.degradation(ctx.bayes))
        st.dataframe(ctx.bayes.degradation_summary().rename(columns={
            "deg_s_per_lap": "mean s/lap", "posterior_sd": "± posterior sd", "between_group_sd": "race-to-race sd",
            "new_group_p05": "new race 5%", "new_group_p95": "new race 95%"}), hide_index=True)
        st.caption("Hierarchical Bayesian regression; the sampler runs in NumPy. With few races the race-to-race "
                   "spread depends on the prior (src/bayes.py).")

    with t_sc:
        st.pyplot(plots.scenarios(rec))
        st.dataframe(rec.scenarios[["scenario", "best_strategy", "first_pit_lap", "adaptation_gain_s", "race_time"]]
                     .round(1), hide_index=True, width="stretch")
        st.markdown("**If it happens at that lap:**")
        for name, r in rec.reactive:
            st.write("• " + describe_reactive(name, r))
        st.caption(f"SC/VSC pit-loss factors, lap-time factors and tyre-wear factors are assumptions in src/config.py. "
                   f"Frequency in the simulation: SC {ctx.stats['p_sc']:.0%}, VSC {ctx.stats['p_vsc']:.0%} "
                   f"(from {ctx.stats['source']}).")

    with t_win:
        st.pyplot(plots.pit_window(rec))
        st.dataframe(rec.table.drop(columns=["compounds", "pit_laps", "family"]).head(15).round(2), hide_index=True)

    with t_uc:
        st.write("Our car is behind a rival; the rival's stop lap is assumed fixed and it does not react.")
        c = st.columns(4)
        gap = c[0].number_input("Gap to rival (s)", 0.0, 60.0, 2.5, 0.5)
        default_b = int(best["pit_laps"][0]) + 2 if len(best["pit_laps"]) else lap + 3
        b_pit = int(c[1].number_input("Rival pits on lap", lap, total_laps - 2, min(max(default_b, lap), total_laps - 2), 1))
        traffic = int(c[2].number_input("Cars we would rejoin behind", 0, 10, 0, 1))
        new_names = names
        new_c = c[3].selectbox("Both cars switch to", new_names, index=new_names.index("HARD") if "HARD" in new_names else 0)
        nc = {v: k for k, v in NAMES.items()}[new_c]
        car = dict(compound=state["compound"], tyre_life=state["tyre_life"], new_compound=nc)
        table, summ = analyse(ctx.models[model], state, car, dict(car), gap, ctx.pit_losses, b_pit, traffic_cars=traffic)
        for line in summ["text"]:
            st.write("• " + line)
        st.pyplot(plots.undercut(table, b_pit))
        st.dataframe(table[["a_pit_lap", "kind", "net_gain_s", "own_race_cost_s", "traffic_s", "margin_s", "p_success"]].round(2),
                     hide_index=True)

    with t_why:
        for title, k in (("Why this strategy", "why"), ("What would change the recommendation", "changes"), ("Risks", "risks")):
            st.markdown(f"**{title}**")
            for line in rec.explanation[k]:
                st.write("• " + line)


if __name__ == "__main__":
    main()
