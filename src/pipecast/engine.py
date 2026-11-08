"""End to end: generate the CRM, backtest every method, fit the current quarter, and answer the API.

The current quarter (quarter 8) is simulated to its end like the others, but nothing here ever computes its
bookings: every forecast reads a snapshot taken before the as-of Monday, so later events are invisible.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import methods as M
from .backtest import METHODS, MID_WEEKS, Backtest, fit_inputs, forecast_week, gate, run_backtest
from .fiscal import WEEKS, as_of_day, iso, label, q_end
from .generator import CATEGORIES, STAGES, CompanyConfig, generate
from .model import training_rows
from .reps import rep_accuracy
from .risk import risk_table, stage_norms
from .snapshots import Snapshots


@dataclass
class EngineConfig:
    company: CompanyConfig = field(default_factory=CompanyConfig)
    current_week: int = 7
    backtest_from: int = 2  # first backtest quarter; the deal model needs two quarters before it
    n_sims: int = 10_000
    backtest_sims: int = 4_000
    seed: int = 0


class Forecaster:
    def __init__(self, cfg: EngineConfig | None = None):
        self.cfg = cfg or EngineConfig()
        self.timings: dict[str, float] = {}

    def run(self) -> Forecaster:
        cfg = self.cfg
        t = time.perf_counter()
        self.crm = generate(cfg.company)
        self.snaps = Snapshots(self.crm)
        self.current_q = cfg.company.quarters - 1
        history = list(range(self.current_q))
        self.timings["generate_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        self.backtest: Backtest = run_backtest(self.snaps, range(cfg.backtest_from, self.current_q), history,
                                               n_sims=cfg.backtest_sims, seed=cfg.seed)
        self.timings["backtest_s"] = round(time.perf_counter() - t, 2)

        t = time.perf_counter()
        train = training_rows(self.snaps, history)
        hist = M.history_rows(self.snaps, history)
        self.inputs = fit_inputs(self.snaps, self.current_q, train, hist, seed=cfg.seed)
        self.reps = rep_accuracy(self.snaps, self.crm.reps, history)
        self.history_bookings = [{"quarter": label(q), "bookings": self.snaps.bookings(q)} for q in history]
        self.timings["current_fit_s"] = round(time.perf_counter() - t, 2)
        self._forecasts: dict[int, dict] = {}
        return self

    # Current quarter ----------------------------------------------------------------------------------------
    def forecast(self, week: int | None = None) -> dict:
        week = week or self.cfg.current_week
        if not 1 <= week <= WEEKS:
            raise ValueError("week must be 1 to 13")
        if week not in self._forecasts:
            self._forecasts[week] = self._forecast(week)
        return self._forecasts[week]

    def _forecast(self, week: int) -> dict:
        q = self.current_q
        r = forecast_week(self.snaps, self.inputs, week, self.cfg.n_sims, seed=self.cfg.seed + 99)
        mc, roll, f = r["mc"], r["rollup"], r["forecasts"]
        bt = self.backtest.summary
        by_week = {(x["method"], x["week"]): x for x in bt["by_week"]}
        cov = next(c for c in bt["coverage"]["by_week"] if c["week"] == week)
        rows = []
        for m, lab in METHODS.items():
            h = by_week[(m, week)]
            rows.append({"method": m, "label": lab, "forecast": f[m], "backtest_mape": h["mape"],
                         "backtest_bias": h["bias"]})
        sims = r["sims"]
        o_in_q = self.snaps.open_in_quarter(q, r["day"])
        return {
            "quarter": label(q), "week": week, "weeks": WEEKS, "as_of": iso(r["day"]),
            "data_through": iso(r["day"] - 1), "quarter_end": iso(q_end(q)),
            "won_to_date": r["won_to_date"], "commit": roll["commit"], "best_case": roll["best_case"],
            "open_deals": int(len(r["open"])), "open_in_quarter": int(len(o_in_q)),
            "open_in_quarter_amount": float(o_in_q["amount"].sum()), "new_pipeline_expected": r["new_pipeline"],
            "methods": rows,
            "monte_carlo": {**mc, "p_hit_commit": float(np.mean(sims >= roll["commit"])),
                            "p_hit_best_case": float(np.mean(sims >= roll["best_case"])),
                            "backtest_coverage_this_week": cov["coverage"]},
            "_probs": r["probs"], "_open": r["open"],
        }

    def public_forecast(self, week: int | None = None) -> dict:
        return {k: v for k, v in self.forecast(week).items() if not k.startswith("_")}

    def risk(self, week: int | None = None, limit: int = 25) -> dict:
        fc = self.forecast(week)
        q = self.current_q
        day = as_of_day(q, fc["week"])
        o, p = fc["_open"], fc["_probs"]
        in_q = (o.close_day <= q_end(q)).to_numpy()
        norms = stage_norms(self.crm.stage_ev, self.crm.deals, day)
        t = risk_table(o[in_q], p[in_q], q, norms)
        names = self.crm.reps.set_index("rep_id")["name"]
        counts = {k: int(sum(k in f for f in t.flag_list)) for k in sorted({x for f in t.flag_list for x in f})}
        out = []
        for r in t.head(limit).itertuples(index=False):
            out.append({
                "deal_id": r.deal_id, "account": r.account, "rep_id": r.rep_id, "rep": names[r.rep_id],
                "segment": r.segment, "stage": STAGES[int(r.stage)], "amount": float(r.amount),
                "close_date": iso(int(r.close_day)), "category": CATEGORIES[int(r.category)],
                "p_close": round(float(r.p_close), 4), "expected_value": float(r.expected_value),
                "value_at_risk": float(r.value_at_risk), "swing": float(r.swing), "flags": list(r.flag_list),
                "pushes": int(r.pushes), "days_in_stage": int(r.days_in_stage),
                "days_since_activity": int(r.days_since_act),
            })
        return {"quarter": fc["quarter"], "week": fc["week"], "deals_in_quarter": int(in_q.sum()),
                "flag_counts": counts, "total_value_at_risk": float(t.value_at_risk.sum()),
                "flagged_deals": int((t.n_flags > 0).sum()), "deals": out}

    def backtest_report(self) -> dict:
        s = self.backtest.summary
        ok, checks = gate(s)
        return {**s, "gate": {"pass": ok, "checks": checks}}

    def summary(self) -> dict:
        return {"company": "Corvane", "reps": int(len(self.crm.reps)), "regions": 5,
                "deals": int(len(self.crm.deals)), "history": self.history_bookings,
                "current_quarter": label(self.current_q), "current_week": self.cfg.current_week,
                "timings": self.timings, "mid_weeks": list(MID_WEEKS)}

    def save(self, path: Path) -> None:
        """Pickle without the per-day snapshot cache and the cached simulations; both rebuild in milliseconds."""
        path.parent.mkdir(parents=True, exist_ok=True)
        cache, fcs = self.snaps._cache, self._forecasts
        self.snaps._cache, self._forecasts = {}, {}
        try:
            with open(path, "wb") as fh:
                pickle.dump(self, fh)
        finally:
            self.snaps._cache, self._forecasts = cache, fcs

    @staticmethod
    def load(path: Path) -> Forecaster:
        with open(path, "rb") as fh:
            return pickle.load(fh)


def backtest_table(summary: dict, weeks=(1, 3, 5, 7, 9, 11, 13)) -> str:
    bw = pd.DataFrame(summary["by_week"])
    lines = ["| Method | " + " | ".join(f"Wk {w}" for w in weeks) + " | Mid-quarter MAPE | Mid-quarter bias |",
             "|---|" + "---:|" * (len(weeks) + 2)]
    for m, lab in METHODS.items():
        g = bw[bw.method == m].set_index("week")
        o = summary["overall"][m]
        cells = " | ".join(f"{g.loc[w, 'mape']:.1%}" for w in weeks)
        lines.append(f"| {lab} | {cells} | {o['mid_mape']:.1%} | {o['mid_bias']:+.1%} |")
    return "\n".join(lines)
