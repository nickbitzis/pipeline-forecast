"""Replay past quarters week by week and score every method against what actually closed.

For a backtest quarter q, every input is rebuilt as it stood on each Monday: the snapshot only holds events logged
before that day, the deal model and calibration train on quarters before q, the stage rates and rep history come
from quarters before q, and the time-series baseline sees only past quarter totals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import methods as M
from .fiscal import WEEKS, as_of_day, label
from .model import DealModel, brier, ece, features, reliability, training_rows
from .montecarlo import simulate, summarize
from .reps import AdjustedRollup
from .snapshots import Snapshots

METHODS = {
    "rollup_commit": "Rep roll-up: commit",
    "rollup_best_case": "Rep roll-up: best case",
    "adjusted_rollup": "Adjusted roll-up",
    "stage_weighted": "Stage-weighted pipeline",
    "deal_model": "Deal model",
    "monte_carlo_p50": "Monte Carlo P50",
    "time_series": "Time-series baseline",
}
MID_WEEKS = (5, 6, 7, 8, 9)
STAGE_RATE_QUARTERS = 4


def mape(forecast, actual) -> float:
    f, a = np.asarray(forecast, float), np.asarray(actual, float)
    return float(np.mean(np.abs(f - a) / a))


def bias(forecast, actual) -> float:
    """Mean signed percentage error. Positive means the method called it too high."""
    f, a = np.asarray(forecast, float), np.asarray(actual, float)
    return float(np.mean((f - a) / a))


def coverage(lo, hi, actual) -> float:
    lo, hi, a = (np.asarray(x, float) for x in (lo, hi, actual))
    return float(np.mean((a >= lo) & (a <= hi)))


@dataclass
class QuarterInputs:
    """Everything a forecast for quarter q may use, fitted on quarters before q only."""

    q: int
    model: DealModel
    stage_rates: dict
    adjusted: AdjustedRollup
    past_bookings: list
    shape: dict
    new_pipeline: M.NewPipeline


def fit_inputs(snaps: Snapshots, q: int, train: pd.DataFrame, hist: pd.DataFrame, seed: int = 0) -> QuarterInputs:
    past = list(range(0, q))
    model = DealModel(q, past, seed=seed).fit(train)
    recent = hist[hist.quarter.isin(past[-STAGE_RATE_QUARTERS:])]
    return QuarterInputs(
        q=q, model=model, stage_rates=M.stage_conversion_rates(recent),
        adjusted=AdjustedRollup(hist[hist.quarter.isin(past)]),
        past_bookings=[snaps.bookings(p) for p in past], shape=M.linearity(snaps, past),
        new_pipeline=M.NewPipeline(snaps, past),
    )


def forecast_week(snaps: Snapshots, inp: QuarterInputs, week: int, n_sims: int, seed: int = 0) -> dict:
    q = inp.q
    day = as_of_day(q, week)
    roll = M.rep_rollup(snaps, q, day)
    o = snaps.open_deals(day)
    X = features(o, q)
    p = inp.model.predict(X)
    won = roll["won"]
    s = seed + 1000 * q + week
    new = inp.new_pipeline
    sims = simulate(o["amount"].to_numpy(), p, won, n_sims=n_sims, seed=s)
    sims += new.draw(week, n_sims, np.random.default_rng(s + 7))
    mc = summarize(sims)
    return {
        "quarter": q, "week": week, "day": day, "won_to_date": won,
        "forecasts": {
            "rollup_commit": roll["commit"],
            "rollup_best_case": roll["best_case"],
            "adjusted_rollup": inp.adjusted.forecast(snaps, q, week),
            "stage_weighted": M.stage_weighted(snaps, q, day, inp.stage_rates),
            "deal_model": won + float((o["amount"].to_numpy() * p).sum()) + new.mean[week],
            "monte_carlo_p50": mc["p50"],
            "time_series": M.time_series(snaps, q, week, inp.past_bookings, inp.shape),
        },
        "new_pipeline": new.mean[week], "mc": mc, "rollup": roll, "open": o, "probs": p, "X": X, "sims": sims,
    }


@dataclass
class Backtest:
    rows: pd.DataFrame  # quarter, week, method, forecast, actual
    intervals: pd.DataFrame  # quarter, week, p10, p50, p90, actual
    deal_preds: pd.DataFrame  # quarter, week, p_raw, p_cal, y
    quarters: list[int]
    summary: dict = field(default_factory=dict)


def run_backtest(snaps: Snapshots, quarters, history_quarters, n_sims: int = 4000, seed: int = 0) -> Backtest:
    train = training_rows(snaps, history_quarters)
    hist = M.history_rows(snaps, history_quarters)
    rows, ints, preds = [], [], []
    for q in quarters:
        inp = fit_inputs(snaps, q, train, hist, seed=seed)
        actual = snaps.bookings(q)
        won = set(snaps.won_deals(q).deal_id)
        for w in range(1, WEEKS + 1):
            r = forecast_week(snaps, inp, w, n_sims, seed)
            for m, f in r["forecasts"].items():
                rows.append({"quarter": q, "week": w, "method": m, "forecast": f, "actual": actual})
            ints.append({"quarter": q, "week": w, "p10": r["mc"]["p10"], "p50": r["mc"]["p50"],
                         "p90": r["mc"]["p90"], "actual": actual})
            y = r["open"].deal_id.isin(won).astype(int).to_numpy()
            preds.append(pd.DataFrame({"quarter": q, "week": w, "p_raw": inp.model.predict_raw(r["X"]),
                                       "p_cal": r["probs"], "y": y}))
    bt = Backtest(pd.DataFrame(rows), pd.DataFrame(ints), pd.concat(preds, ignore_index=True), list(quarters))
    bt.summary = summarize_backtest(bt)
    return bt


def summarize_backtest(bt: Backtest) -> dict:
    rows, ints, dp = bt.rows, bt.intervals, bt.deal_preds
    by_week = []
    for (m, w), g in rows.groupby(["method", "week"]):
        by_week.append({"method": m, "week": int(w), "mape": mape(g.forecast, g.actual),
                        "bias": bias(g.forecast, g.actual)})
    overall = {}
    for m, g in rows.groupby("method"):
        mid = g[g.week.isin(MID_WEEKS)]
        overall[m] = {"label": METHODS[m], "mape": mape(g.forecast, g.actual), "bias": bias(g.forecast, g.actual),
                      "mid_mape": mape(mid.forecast, mid.actual), "mid_bias": bias(mid.forecast, mid.actual)}
    cov_week = [{"week": int(w), "coverage": coverage(g.p10, g.p90, g.actual),
                 "below_p10": float(np.mean(g.actual < g.p10)), "above_p90": float(np.mean(g.actual > g.p90))}
                for w, g in ints.groupby("week")]
    paths = {m: {label(int(q)): [float(x) for x in (g.sort_values("week").forecast / g.actual - 1)]
                 for q, g in gm.groupby("quarter")} for m, gm in rows.groupby("method")}
    calib = {
        "raw": {"brier": brier(dp.p_raw, dp.y), "ece": ece(dp.p_raw, dp.y)},
        "calibrated": {"brier": brier(dp.p_cal, dp.y), "ece": ece(dp.p_cal, dp.y)},
        "reliability": reliability(dp.p_cal, dp.y),
        "n": int(len(dp)), "base_rate": float(dp.y.mean()),
    }
    actuals = {label(int(q)): float(g.actual.iloc[0]) for q, g in ints.groupby("quarter")}
    return {
        "quarters": [label(q) for q in bt.quarters], "methods": METHODS, "by_week": by_week, "overall": overall,
        "coverage": {"overall": coverage(ints.p10, ints.p90, ints.actual),
                     "mid": coverage(*(ints[ints.week.isin(MID_WEEKS)][c] for c in ("p10", "p90", "actual"))),
                     "by_week": cov_week},
        "error_paths": paths, "calibration": calib, "actuals": actuals, "mid_weeks": list(MID_WEEKS),
    }


def gate(summary: dict, band: tuple[float, float] = (0.70, 0.90)) -> tuple[bool, list[str]]:
    cov = summary["coverage"]["overall"]
    model, roll = summary["overall"]["deal_model"]["mid_mape"], summary["overall"]["rollup_commit"]["mid_mape"]
    checks = [
        (band[0] <= cov <= band[1],
         f"Monte Carlo P10 to P90 coverage {cov:.1%} (band {band[0]:.0%} to {band[1]:.0%})"),
        (model < roll, f"mid-quarter MAPE: deal model {model:.1%} vs rep roll-up {roll:.1%}"),
    ]
    return all(ok for ok, _ in checks), [("PASS " if ok else "FAIL ") + msg for ok, msg in checks]
