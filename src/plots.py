"""Matplotlib figures shared by the CLI runner and the Streamlit dashboard. Every function returns a Figure."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .bayes import COMP_NAMES
from .planner import fmt_time

COLORS = {"SOFT": "#d62728", "MEDIUM": "#c9a227", "HARD": "#4a4a4a"}


def _fmt_axis(ax, axis="x"):
    f = plt.FuncFormatter(lambda v, _: fmt_time(v))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)


def strategy_comparison(rec):
    s = rec.summary
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(s) + 1.4))
    y = np.arange(len(s))[::-1]
    ax.hlines(y, s["p10_s"], s["p90_s"], color="#999", lw=6, alpha=.5, label="10th-90th percentile")
    ax.plot(s["mean_s"], y, "o", color="#1f3a5f", label="mean")
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.text(r["p90_s"], yi, f"  P(fastest) {r['p_fastest']:.0%}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(s["strategy"], fontsize=8)
    _fmt_axis(ax)
    ax.set_xlabel("Predicted finish time (includes simulated safety cars)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8, frameon=False)
    ax.margins(x=0.25)
    fig.tight_layout()
    return fig


def paired_distributions(rec):
    """Time difference to the recommended strategy, same simulated races (positive = slower than recommended)."""
    t = rec.sim.totals
    labels = rec.summary["strategy"].tolist()
    fig, ax = plt.subplots(figsize=(8, 0.55 * t.shape[1] + 1.4))
    for i, k in enumerate(range(1, t.shape[1])[::-1]):
        d = t[:, k] - t[:, 0]
        lo, q1, med, q3, hi = np.percentile(d, [5, 25, 50, 75, 95])
        ax.hlines(i, lo, hi, color="#999", lw=1.5)
        ax.hlines(i, q1, q3, color="#1f3a5f", lw=8)
        ax.plot(med, i, "o", color="white", ms=4)
    ax.set_yticks(range(t.shape[1] - 1))
    ax.set_yticklabels(labels[1:][::-1], fontsize=8)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Seconds slower than the recommended strategy (thick = 25-75%, thin = 5-95%)")
    fig.tight_layout()
    return fig


def degradation(bayes, max_age=40):
    ages = np.arange(0, max_age + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    rng = np.random.default_rng(0)
    for c, name in enumerate(COMP_NAMES):
        mu, tau = bayes.mu_[:, 2 + c], bayes.tau_[:, 2 + c]
        new = mu + tau * rng.standard_normal(len(mu))
        lo, hi = np.percentile(new[:, None] * ages[None, :], [5, 95], axis=0)
        ax.fill_between(ages, lo, hi, color=COLORS[name], alpha=.15)
        ax.plot(ages, mu.mean() * ages, color=COLORS[name], label=f"{name}: {mu.mean():.3f} ± {mu.std():.3f} s/lap")
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Lap time lost vs a fresh tyre (s)")
    ax.set_title("Bayesian tyre degradation: mean ± posterior sd; band = 90% for a new race", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def pit_window(rec):
    t = rec.table
    fam = rec.best["family"]
    s = t[(t["family"] == fam) & (t["n_stops"] == rec.best["n_stops"])]
    s = s.assign(first=[p[0] for p in s["pit_laps"]]).groupby("first")["expected_delta_s"].min()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(s.index, s.values - s.min(), color="#1f3a5f")
    ax.axvline(rec.best["pit_laps"][0] if len(rec.best["pit_laps"]) else s.idxmin(), ls="--", c="grey")
    ax.set_xlabel("First pit lap")
    ax.set_ylabel("Expected seconds lost vs best lap")
    ax.set_title(f"Pit window for {fam}", fontsize=9)
    fig.tight_layout()
    return fig


def scenarios(rec):
    s = rec.scenarios
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(s) + 1.4))
    y = np.arange(len(s))[::-1]
    ax.barh(y, s["adaptation_gain_s"], color="#2a7f62")
    for yi, (_, r) in zip(y, s.iterrows()):
        pit = "no stop" if r["first_pit_lap"] is None or r["first_pit_lap"] != r["first_pit_lap"] else f"pit lap {int(r['first_pit_lap'])}"
        ax.text(r["adaptation_gain_s"], yi, f"  {pit}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(s["scenario"], fontsize=8)
    ax.set_xlabel("Seconds gained by re-planning vs keeping the no-safety-car plan")
    ax.margins(x=0.2)
    fig.tight_layout()
    return fig


def undercut(table, b_pit_lap):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(table["a_pit_lap"], table["margin_s"], "o-", color="#1f3a5f", label="margin (s)")
    ax.fill_between(table["a_pit_lap"], table["margin_p05"], table["margin_p95"], alpha=.15, color="#1f3a5f")
    ax.axhline(0, color="k", lw=.8)
    ax.axvline(b_pit_lap, ls="--", c="grey")
    ax.text(b_pit_lap, ax.get_ylim()[1], " rival stops", fontsize=8, va="top")
    ax2 = ax.twinx()
    ax2.plot(table["a_pit_lap"], table["p_success"], "s--", color="#d62728", ms=3, label="P(success)")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("P(success)")
    ax.set_xlabel("Our pit lap")
    ax.set_ylabel("Margin over rival after the window (s)\n>0 = we come out ahead")
    fig.tight_layout()
    return fig
