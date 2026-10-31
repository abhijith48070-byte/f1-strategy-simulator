"""Deterministic strategy planner: enumerate every legal strategy from the current race state and price it.

A strategy = tyre compound for each stint + the lap at the end of which each stop happens. For a given scenario
(no neutralisation, or a known SC/VSC) the model's mean lap deltas are tabulated once, so thousands of strategies
are priced by table look-ups. Tyre-life limits come from the data (no stint longer than anything observed),
because linear and tree models cannot be trusted past the training range.

state keys: lap (next lap to be driven), total_laps, tyre_life (age on that lap), compound (0/1/2), fresh,
            track_temp, air_temp, position, used (compound codes already used in the race), ref_lap_s, elapsed_s.
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from . import config as C
from .scenarios import NeutralArrays

NAMES = {0: "SOFT", 1: "MEDIUM", 2: "HARD"}


@dataclass
class Constraints:
    max_stops: int = C.MAX_STOPS
    min_stint: int = C.MIN_STINT_LAPS
    two_compounds: bool = C.REQUIRE_TWO_COMPOUNDS
    allowed: tuple = (0, 1, 2)
    enforce_life: bool = True


def strategy_label(compounds, pit_laps):
    seq = " → ".join(NAMES[c] for c in compounds)
    return seq + (" (pit L" + ", L".join(str(p) for p in pit_laps) + ")" if pit_laps else " (no stop)")


def family(compounds):
    return ">".join(NAMES[c] for c in compounds)


class Planner:
    def __init__(self, model, state, pit_loss, neutral=None, constraints=None, deg_mult=1.0):
        self.model, self.state, self.pit_loss = model, dict(state), float(pit_loss)
        self.neutral, self.cons, self.deg_mult = neutral, constraints or Constraints(), deg_mult
        self.n, self.L = int(state["lap"]), int(state["total_laps"])
        self.ref = float(state.get("ref_lap_s", 0.0))
        self.elapsed = float(state.get("elapsed_s", (self.n - 1) * self.ref))
        self.arr = NeutralArrays.single(self.L, self.ref, neutral)
        self._build_tables()

    # -- table construction ------------------------------------------------------------------
    def _deltas(self, comp, ages, laps, fresh):
        """Mean lap deltas for many (age, lap) pairs in ONE model call, with SC laps and the degradation multiplier."""
        d = np.asarray(self.model.mean(comp, ages, laps, self.L, fresh, self.state), float)
        if self.deg_mult != 1.0:                    # scale only the tyre-age part of the lap delta
            d0 = np.asarray(self.model.mean(comp, np.ones_like(ages), laps, self.L, fresh, self.state), float)
            d = d0 + self.deg_mult * (d - d0)
        return np.where(self.arr.mask[0, laps], self.arr.lap_extra[0, laps], d)

    def _build_tables(self):
        n, L = self.n, self.L
        c0 = int(self.state["compound"])
        laps = np.arange(n, L + 1)
        self.cur_age = self.arr.age(float(self.state["tyre_life"]), n, laps)[0]      # effective age on laps n..L
        d = self._deltas(c0, self.cur_age, laps, float(self.state["fresh"]))
        self.cur_cs = np.concatenate([[0.0], np.cumsum(d)])          # cur_cs[k] = sum of laps n..n+k-1
        self.new_cs, self.new_age = {}, {}
        starts = list(range(n + 1, L + 1))
        if not starts:
            return
        lap_all = np.concatenate([np.arange(s, L + 1) for s in starts])
        age_all = np.concatenate([self.arr.age(1.0, s, np.arange(s, L + 1))[0] for s in starts])
        cuts = np.cumsum([L - s + 1 for s in starts])[:-1]
        for c in self.cons.allowed:
            d_all = self._deltas(c, age_all, lap_all, 1.0)
            for s, dd, aa in zip(starts, np.split(d_all, cuts), np.split(age_all, cuts)):
                self.new_cs[c, s] = np.concatenate([[0.0], np.cumsum(dd)])   # [k] = sum of laps s..s+k-1
                self.new_age[c, s] = aa

    def _cur(self, p):            # laps n..p
        return self.cur_cs[p - self.n + 1]

    def _new(self, c, s, e):      # laps s..e
        return self.new_cs[c, s][e - s + 1]

    def _loss(self, p):
        return self.pit_loss * self.arr.pit_factor[0, p]

    def _limit_ok(self, comp, last_age):
        return (not self.cons.enforce_life) or last_age <= self.model.life_limit(comp)

    def set_deg_mult(self, m):
        self.deg_mult = m
        self._build_tables()

    # -- pricing --------------------------------------------------------------------------------
    def evaluate(self, compounds, pit_laps):
        """Expected extra seconds (sum of lap deltas + pit losses) of a strategy; no legality checks."""
        compounds, pit_laps = tuple(compounds), tuple(pit_laps)
        n, L = self.n, self.L
        if not pit_laps:
            return float(self._cur(L))
        t = self._cur(pit_laps[0]) + self._loss(pit_laps[0])
        for i, c in enumerate(compounds[1:], start=1):
            s = pit_laps[i - 1] + 1
            e = pit_laps[i] if i < len(pit_laps) else L
            t += self._new(c, s, e) + (self._loss(pit_laps[i]) if i < len(pit_laps) else 0.0)
        return float(t)

    def _row(self, compounds, pit_laps, total):
        base = self.elapsed + (self.L - self.n + 1) * self.ref
        return dict(strategy=strategy_label(compounds, pit_laps), family=family(compounds),
                    compounds=tuple(compounds), pit_laps=tuple(pit_laps), n_stops=len(pit_laps),
                    expected_delta_s=total, race_time_s=base + total,
                    pit_time_s=sum(self._loss(p) for p in pit_laps))

    def enumerate(self):
        n, L, cons = self.n, self.L, self.cons
        c0 = int(self.state["compound"])
        used0 = set(self.state.get("used", {c0})) | {c0}
        ms = cons.min_stint
        rows = []

        def legal(comps):
            return (not cons.two_compounds) or len(used0 | set(comps)) >= 2

        if self._limit_ok(c0, self.cur_age[-1]) and legal([c0]):
            rows.append(self._row([c0], [], self._cur(L)))
        if cons.max_stops >= 1:
            for p1 in range(n, L - ms + 1):
                if not self._limit_ok(c0, self.cur_age[p1 - n]):
                    break
                for c1 in cons.allowed:
                    if not legal([c0, c1]) or not self._limit_ok(c1, self.new_age[c1, p1 + 1][-1]):
                        continue
                    rows.append(self._row([c0, c1], [p1], self.evaluate([c0, c1], [p1])))
        if cons.max_stops >= 2:
            for p1 in range(n, L - 2 * ms + 1):
                if not self._limit_ok(c0, self.cur_age[p1 - n]):
                    break
                for c1 in cons.allowed:
                    for p2 in range(p1 + ms, L - ms + 1):
                        if not self._limit_ok(c1, self.new_age[c1, p1 + 1][p2 - p1 - 1]):
                            break
                        for c2 in cons.allowed:
                            if not legal([c0, c1, c2]) or not self._limit_ok(c2, self.new_age[c2, p2 + 1][-1]):
                                continue
                            rows.append(self._row([c0, c1, c2], [p1, p2], self.evaluate([c0, c1, c2], [p1, p2])))
        if not rows:
            raise ValueError("no legal strategy: relax min stint / tyre-life limits")
        return pd.DataFrame(rows).sort_values("expected_delta_s").reset_index(drop=True)


def best_per_family(table, k=6):
    """Best pit-lap choice for each compound sequence, fastest first (this is what the dashboard compares)."""
    return table.drop_duplicates("family").head(k).reset_index(drop=True)


def fmt_time(seconds):
    m, s = divmod(float(seconds), 60)
    h, m = divmod(int(m), 60)
    return f"{h}:{m:02d}:{s:04.1f}" if h else f"{m}:{s:04.1f}"
