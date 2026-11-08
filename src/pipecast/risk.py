"""Deal risk flags and the deals that move the forecast most.

Flags are the things a sales manager asks about in a deal review:
- pushed twice: the close date has moved out at least twice
- stuck: days in the current stage beyond the 90th percentile of past stints at that stage for the segment
- no activity in 14 days: nobody has logged a call, meeting, or email in two weeks
- late amount increase: the amount went up in the last 21 days while in Proposal or Negotiation
- quarter end, early stage: close date in the last week of the quarter while still in Discovery or Qualification
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fiscal import q_end
from .generator import BEST_CASE, COMMIT, WON

FLAG_LABELS = {
    "pushed_twice": "pushed twice",
    "stuck_in_stage": "stuck in stage",
    "no_activity_14d": "no activity in 14 days",
    "late_amount_increase": "amount increased late",
    "quarter_end_early_stage": "quarter end close, early stage",
}


def stage_norms(stage_ev: pd.DataFrame, deals: pd.DataFrame, before_day: int, pct: float = 90) -> dict:
    """Percentile of completed stage stints (days from entering a stage to leaving it), by segment and stage."""
    ev = stage_ev[stage_ev.day < before_day].sort_values(["deal_id", "day"], kind="stable")
    nxt = ev.groupby("deal_id")["day"].shift(-1)
    stints = ev.assign(duration=nxt - ev.day).dropna(subset=["duration"])
    stints = stints[stints.stage < WON].merge(deals[["deal_id", "segment"]], on="deal_id")
    return {(s, int(st)): float(np.percentile(g.duration, pct))
            for (s, st), g in stints.groupby(["segment", "stage"]) if len(g) >= 5}


def flag_deals(open_deals: pd.DataFrame, q: int, norms: dict) -> pd.DataFrame:
    o = open_deals
    norm = np.array([norms.get((s, int(st)), np.inf) for s, st in zip(o.segment, o.stage, strict=True)])
    f = pd.DataFrame(index=o.index)
    f["pushed_twice"] = o.pushes >= 2
    f["stuck_in_stage"] = o.days_in_stage.to_numpy() > norm
    f["no_activity_14d"] = o.days_since_act > 14
    f["late_amount_increase"] = o.late_increase == 1
    f["quarter_end_early_stage"] = o.close_day.between(q_end(q) - 6, q_end(q)) & (o.stage <= 2)
    return f


def risk_table(open_in_q: pd.DataFrame, probs: np.ndarray, q: int, norms: dict) -> pd.DataFrame:
    """Every in-quarter open deal with its flags, model probability, and value at risk.

    Value at risk is the part of the rep's call the model does not expect: amount x (1 - p) for deals the rep
    counts (commit or best case). Swing is the deal's standard deviation contribution, amount x sqrt(p (1 - p)).
    """
    o = open_in_q.copy()
    flags = flag_deals(o, q, norms)
    o["p_close"] = probs
    o["flag_list"] = [[FLAG_LABELS[k] for k in flags.columns if row[k]] for _, row in flags.iterrows()]
    o["n_flags"] = flags.sum(axis=1).astype(int)
    counted = o.category.isin([COMMIT, BEST_CASE])
    o["value_at_risk"] = np.where(counted, o.amount * (1 - o.p_close), 0.0)
    o["swing"] = o.amount * np.sqrt(o.p_close * (1 - o.p_close))
    o["expected_value"] = o.amount * o.p_close
    return o.sort_values(["value_at_risk", "swing"], ascending=False)
