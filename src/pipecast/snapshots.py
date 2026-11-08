"""Rebuild the state of every opportunity on any day from the CRM field history.

A real CRM keeps the current row plus an audit log of field changes. Forecasting needs the pipeline as it looked on
each past Monday, so this module replays the log: the last value of each field logged before the as-of day, plus
dynamics that only the history can show (days in stage, close date pushes, amount changes, activity recency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fiscal import q_end, q_start, quarter_of
from .generator import CRM, LOST, WON

DEAL_COLS = ["deal_id", "account", "rep_id", "region", "segment", "source", "created_day"]


def _sorted(ev: pd.DataFrame) -> pd.DataFrame:
    return ev.sort_values(["deal_id", "day"], kind="stable").reset_index(drop=True)


class Snapshots:
    """Replays the field history. Results are cached per day because the backtest asks for the same Mondays often."""

    def __init__(self, crm: CRM):
        self.crm = crm
        self.deals = crm.deals[DEAL_COLS].set_index("deal_id")
        self.stage_ev = _sorted(crm.stage_ev)
        self.amount_ev = _sorted(crm.amount_ev)
        self.close_ev = _sorted(crm.close_ev)
        self.cat_ev = _sorted(crm.cat_ev)
        self.acts = crm.acts
        self._cache: dict[int, pd.DataFrame] = {}

    def at(self, day: int) -> pd.DataFrame:
        """State at the start of `day`: everything logged on earlier days, nothing from `day` onward."""
        if day not in self._cache:
            self._cache[day] = self._build(day)
        return self._cache[day]

    def _build(self, d: int) -> pd.DataFrame:
        def before(ev):
            return ev[ev.day < d]

        st = before(self.stage_ev).groupby("deal_id", sort=False).last()
        if st.empty:
            return pd.DataFrame(columns=DEAL_COLS + ["stage"])
        out = self.deals.loc[st.index].copy()
        out["stage"] = st["stage"].astype(int)
        out["stage_since"] = st["day"].astype(int)

        am = before(self.amount_ev)
        g = am.groupby("deal_id", sort=False)
        out["amount"] = g["amount"].last()
        out["amount0"] = g["amount"].first()
        inc = am[am.change == 1].groupby("deal_id")["day"].max()
        out["last_increase_day"] = inc.reindex(out.index)
        out["n_amount_changes"] = g.size() - 1

        cl = before(self.close_ev)
        gc = cl.groupby("deal_id", sort=False)
        out["close_day"] = gc["close_day"].last().astype(int)
        out["pushes"] = gc["push"].sum().astype(int)

        out["category"] = before(self.cat_ev).groupby("deal_id", sort=False)["category"].last().astype(int)

        a = self.acts[self.acts.day < d]
        ga = a.groupby("deal_id")["day"]
        out["acts_total"] = ga.size().reindex(out.index).fillna(0).astype(int)
        out["last_act_day"] = ga.max().reindex(out.index)
        out["acts_14d"] = a[a.day >= d - 14].groupby("deal_id").size().reindex(out.index).fillna(0).astype(int)

        out["as_of"] = d
        out["age"] = d - out["created_day"]
        out["days_in_stage"] = d - out["stage_since"]
        out["days_since_act"] = (d - out["last_act_day"].fillna(out["created_day"])).astype(int)
        out["amount_change_pct"] = out["amount"] / out["amount0"] - 1
        recent_inc = (d - out["last_increase_day"]).le(21).fillna(False).astype(bool)
        out["late_increase"] = (recent_inc & out["stage"].between(4, 5)).astype(int)
        out["is_open"] = out["stage"] < WON
        out["is_won"] = out["stage"] == WON
        out["is_lost"] = out["stage"] == LOST
        return out.reset_index()

    # Quarter accounting -------------------------------------------------------------------------------------
    def won_to_date(self, q: int, day: int) -> float:
        s = self.at(day)
        return float(s.loc[s.is_won & (s.close_day >= q_start(q)), "amount"].sum())

    def bookings(self, q: int) -> float:
        """Closed won amount with a close date inside quarter q. Only known once the quarter is over."""
        return self.won_to_date(q, q_end(q) + 1)

    def won_deals(self, q: int) -> pd.DataFrame:
        s = self.at(q_end(q) + 1)
        return s[s.is_won & s.close_day.between(q_start(q), q_end(q))]

    def open_in_quarter(self, q: int, day: int) -> pd.DataFrame:
        """Open deals whose close date falls in quarter q: the deals the rep roll-up counts."""
        s = self.at(day)
        return s[s.is_open & (s.close_day <= q_end(q))]

    def open_deals(self, day: int) -> pd.DataFrame:
        s = self.at(day)
        return s[s.is_open]


def replay_one(crm: CRM, deal_id: str, day: int) -> dict | None:
    """Slow, obvious reconstruction of one deal for tests: walk its events in order."""
    created = crm.deals.loc[crm.deals.deal_id == deal_id, "created_day"].iloc[0]
    if created >= day:
        return None
    state: dict = {"pushes": 0}
    for name, ev, col in (("stage", crm.stage_ev, "stage"), ("amount", crm.amount_ev, "amount"),
                          ("close_day", crm.close_ev, "close_day"), ("category", crm.cat_ev, "category")):
        rows = ev[(ev.deal_id == deal_id) & (ev.day < day)].sort_values("day", kind="stable")
        for r in rows.itertuples(index=False):
            state[name] = getattr(r, col)
            if name == "close_day":
                state["pushes"] += int(r.push)
    acts = crm.acts[(crm.acts.deal_id == deal_id) & (crm.acts.day < day)].day.to_numpy()
    state["acts_total"] = int(len(acts))
    state["acts_14d"] = int(np.sum(acts >= day - 14))
    return state


def quarter_week_days(q: int) -> list[int]:
    return [q_start(q) + 7 * w for w in range(13)]


__all__ = ["Snapshots", "replay_one", "quarter_of", "quarter_week_days"]
