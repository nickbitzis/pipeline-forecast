"""The non-ML forecast methods: rep roll-up, stage-weighted pipeline, time-series baseline.

Every method answers the same question: total closed won bookings for quarter q, as known at the start of a given
week. Every method starts from closed won to date and adds an estimate for what is still open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fiscal import WEEKS, as_of_day, q_end
from .generator import BEST_CASE, COMMIT
from .snapshots import Snapshots

STAGE_IDS = (1, 2, 3, 4, 5)


def rep_rollup(snaps: Snapshots, q: int, day: int) -> dict:
    """What the reps called: commit = won + open deals marked commit; best case adds deals marked best case."""
    won = snaps.won_to_date(q, day)
    o = snaps.open_in_quarter(q, day)
    commit = float(o.loc[o.category == COMMIT, "amount"].sum())
    best = float(o.loc[o.category == BEST_CASE, "amount"].sum())
    return {"won": won, "commit": won + commit, "best_case": won + commit + best,
            "open_commit": commit, "open_best_case": best, "open_pipeline": float(o["amount"].sum())}


def in_quarter_outcomes(snaps: Snapshots, q: int, day: int) -> pd.DataFrame:
    """Open deals with a close date in quarter q as of `day`, labeled with whether they closed won inside q."""
    o = snaps.open_in_quarter(q, day)
    won = snaps.won_deals(q)
    return o.assign(won_in_q=o.deal_id.isin(set(won.deal_id)).astype(int))


def stage_conversion_rates(rows: pd.DataFrame, prior: float = 0.0) -> dict[int, float]:
    """Dollar weighted share of open in-quarter pipeline at each stage that closed won before quarter end.

    `rows` holds one line per (deal, as-of Monday) with stage, amount and won_in_q. Weighting by dollars matches
    how the method is used: rate times open amount.
    """
    rates = {}
    for s in STAGE_IDS:
        r = rows[rows.stage == s]
        tot = float(r["amount"].sum())
        rates[s] = float((r["amount"] * r["won_in_q"]).sum() / tot) if tot > 0 else prior
    return rates


def stage_weighted(snaps: Snapshots, q: int, day: int, rates: dict[int, float]) -> float:
    o = snaps.open_in_quarter(q, day)
    weights = o["stage"].map(rates).fillna(0.0)
    return snaps.won_to_date(q, day) + float((o["amount"] * weights).sum())


def history_rows(snaps: Snapshots, quarters, weeks=range(1, WEEKS + 1)) -> pd.DataFrame:
    frames = []
    for q in quarters:
        for w in weeks:
            frames.append(in_quarter_outcomes(snaps, q, as_of_day(q, w)).assign(quarter=q, week=w))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Time-series baseline -------------------------------------------------------------------------------------------
def holt(series: list[float], alpha: float = 0.5, beta: float = 0.3) -> float:
    """One step ahead forecast with Holt's linear trend. Falls back to the mean for very short series."""
    if not series:
        raise ValueError("need at least one past quarter")
    if len(series) < 3:
        return float(np.mean(series))
    level, trend = series[0], series[1] - series[0]
    for y in series[1:]:
        prev = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return float(level + trend)


def linearity(snaps: Snapshots, quarters) -> dict[int, float]:
    """Average share of a quarter's bookings already closed by the start of each week (the quarter's shape)."""
    out = {}
    for w in range(1, WEEKS + 1):
        shares = [snaps.won_to_date(q, as_of_day(q, w)) / max(snaps.bookings(q), 1.0) for q in quarters]
        out[w] = float(np.mean(shares))
    return out


def time_series(snaps: Snapshots, q: int, week: int, past: list[float], shape: dict[int, float]) -> float:
    """Trend forecast of the quarter total, minus the part the quarter's usual shape says is already in."""
    total = holt(past)
    won = snaps.won_to_date(q, as_of_day(q, week))
    return won + max(total * (1 - shape[week]), 0.0)


def quarter_end_flag(close_day: pd.Series, q: int) -> pd.Series:
    return close_day.between(q_end(q) - 6, q_end(q))


# Pipeline not created yet -----------------------------------------------------------------------------------------
def not_yet_created(snaps: Snapshots, q: int) -> dict[int, float]:
    """Bookings in quarter q from deals created on or after the start of each week.

    SMB deals can be created and closed inside the same quarter, so any method that only sums today's pipeline
    misses them. This is the history used to estimate that gap.
    """
    won = snaps.won_deals(q)
    return {w: float(won.loc[won.created_day >= as_of_day(q, w), "amount"].sum()) for w in range(1, WEEKS + 1)}


class NewPipeline:
    """Expected bookings from deals that do not exist yet, by week of quarter, with a spread for the simulation."""

    def __init__(self, snaps: Snapshots, quarters, recent: int = 4):
        hist = [not_yet_created(snaps, q) for q in list(quarters)[-recent:]]
        self.mean = {w: float(np.mean([h[w] for h in hist])) for w in range(1, WEEKS + 1)}
        self.sd = {w: max(float(np.std([h[w] for h in hist], ddof=1)) if len(hist) > 1 else 0.0,
                          0.15 * self.mean[w]) for w in range(1, WEEKS + 1)}

    def draw(self, week: int, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.maximum(rng.normal(self.mean[week], self.sd[week], size=n), 0.0)
