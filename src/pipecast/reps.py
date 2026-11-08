"""Rep forecast accuracy: who sandbags, who has happy ears, and a roll-up corrected by each rep's history.

Evidence is deal level. For every past quarter, each in-quarter deal gets the highest call its rep made on it
between weeks 2 and 12 (pipeline, best case, commit), and the outcome is whether it closed won in that quarter.
A sandbagger's best case and pipeline deals close far more often than the team's. An over-optimistic rep's commits
close far less often. One-sided binomial tests against the team rate decide the flag, so a rep with three deals
cannot be flagged on noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from .fiscal import as_of_day, q_end
from .generator import BEST_CASE, COMMIT, PIPELINE
from .snapshots import Snapshots

CALL_WEEKS = range(2, 13)
MID_WEEK = 7
ALPHA = 0.05


def call_history(snaps: Snapshots, quarters) -> pd.DataFrame:
    """One row per (deal, quarter): the rep's highest call during the quarter and whether it closed won in it."""
    frames = []
    for q in quarters:
        won = set(snaps.won_deals(q).deal_id)
        weekly = [snaps.open_in_quarter(q, as_of_day(q, w))[["deal_id", "rep_id", "segment", "amount", "category"]]
                  for w in CALL_WEEKS]
        cur = pd.concat(weekly).assign(quarter=q)
        agg = cur.groupby(["deal_id", "rep_id", "segment", "quarter"], as_index=False).agg(
            call=("category", "max"), amount=("amount", "last"))
        agg["won_in_q"] = agg.deal_id.isin(won).astype(int)
        frames.append(agg)
    return pd.concat(frames, ignore_index=True)


def quarterly_calls(snaps: Snapshots, quarters, week: int = MID_WEEK) -> pd.DataFrame:
    """Each rep's mid-quarter commit (won so far plus open commit) against what they actually closed."""
    rows = []
    for q in quarters:
        day = as_of_day(q, week)
        s = snaps.at(day)
        won_now = s[s.is_won & (s.close_day >= q * 91)].groupby("rep_id")["amount"].sum()
        o = snaps.open_in_quarter(q, day)
        commit = o[o.category == COMMIT].groupby("rep_id")["amount"].sum()
        actual = snaps.won_deals(q).groupby("rep_id")["amount"].sum()
        reps = set(won_now.index) | set(commit.index) | set(actual.index)
        for r in reps:
            rows.append({"rep_id": r, "quarter": q, "commit": float(won_now.get(r, 0) + commit.get(r, 0)),
                         "actual": float(actual.get(r, 0))})
    return pd.DataFrame(rows)


def team_rates(calls: pd.DataFrame) -> dict:
    c = calls
    seg = c.groupby(["segment", "call"])["won_in_q"].mean()
    return {
        "commit": float(c.loc[c.call == COMMIT, "won_in_q"].mean()),
        "best_case": float(c.loc[c.call == BEST_CASE, "won_in_q"].mean()),
        "pipeline": float(c.loc[c.call == PIPELINE, "won_in_q"].mean()),
        "by_segment": {s: {"commit": float(seg.get((s, COMMIT), np.nan)),
                           "best_case": float(seg.get((s, BEST_CASE), np.nan))}
                       for s in sorted(c.segment.unique())},
    }


def rep_accuracy(snaps: Snapshots, reps: pd.DataFrame, quarters) -> dict:
    calls = call_history(snaps, quarters)
    qc = quarterly_calls(snaps, quarters)
    team = team_rates(calls)
    out = []
    for rep in reps.itertuples(index=False):
        c = calls[calls.rep_id == rep.rep_id]
        cm, bc = c[c.call == COMMIT], c[c.call == BEST_CASE]
        ref = team["by_segment"].get(rep.segment, {"commit": team["commit"], "best_case": team["best_case"]})
        qr = qc[qc.rep_id == rep.rep_id]
        q_comm, q_act = float(qr.commit.sum()), float(qr.actual.sum())
        beat = int((qr.actual > qr.commit).sum())
        p_opt = binomtest(int(cm.won_in_q.sum()), len(cm), ref["commit"], "less").pvalue if len(cm) else 1.0
        p_sand = binomtest(int(bc.won_in_q.sum()), len(bc), ref["best_case"], "greater").pvalue if len(bc) else 1.0
        flag = "accurate"
        if p_opt < ALPHA and p_opt <= p_sand:
            flag = "over-optimistic"
        elif p_sand < ALPHA:
            flag = "sandbagging"
        row = {
            "rep_id": rep.rep_id, "name": rep.name, "region": rep.region, "segment": rep.segment,
            "quarters": int(len(qr)), "commit_calls": q_comm, "actual": q_act,
            "commit_bias": (q_comm - q_act) / q_act if q_act else 0.0,
            "beat_commit_quarters": beat,
            "n_commit_deals": int(len(cm)), "commit_win_rate": float(cm.won_in_q.mean()) if len(cm) else None,
            "team_commit_rate": ref["commit"], "team_best_case_rate": ref["best_case"],
            "n_best_case_deals": int(len(bc)),
            "best_case_win_rate": float(bc.won_in_q.mean()) if len(bc) else None,
            "p_over_optimistic": float(p_opt), "p_sandbagging": float(p_sand), "flag": flag,
            "planted_tendency": rep.tendency,  # ground truth, only for checking the flags on synthetic data
        }
        row["evidence"] = evidence(row)
        out.append(row)
    return {"team": team, "reps": out, "quarters": [int(q) for q in quarters]}


def _pct(v):
    return "n/a" if v is None else f"{v:.0%}"


def evidence(r: dict) -> str:
    q = f"Mid-quarter commit {'over' if r['commit_bias'] > 0 else 'under'} called by {abs(r['commit_bias']):.0%} "
    q += f"over {r['quarters']} quarters; beat commit in {r['beat_commit_quarters']} of {r['quarters']}."
    if r["flag"] == "sandbagging":
        return (f"{r['n_best_case_deals']} deals kept at best case closed at {_pct(r['best_case_win_rate'])}, "
                f"{r['segment']} team {_pct(r['team_best_case_rate'])}. {q}")
    if r["flag"] == "over-optimistic":
        return (f"{r['n_commit_deals']} committed deals closed at {_pct(r['commit_win_rate'])}, "
                f"{r['segment']} team {_pct(r['team_commit_rate'])}. {q}")
    return (f"Commit {_pct(r['commit_win_rate'])} on {r['n_commit_deals']} deals (team {_pct(r['team_commit_rate'])}), "
            f"best case {_pct(r['best_case_win_rate'])} on {r['n_best_case_deals']} (team "
            f"{_pct(r['team_best_case_rate'])}). {q}")


# Adjusted roll-up --------------------------------------------------------------------------------------------
def week_bucket(week) -> np.ndarray:
    """Week 1 stands alone: deals that just rolled in from last quarter have not been re-categorized yet."""
    w = np.asarray(week)
    return np.select([w <= 1, w <= 4, w <= 9], [0, 1, 2], 3)


class AdjustedRollup:
    """Team conversion by forecast category and stretch of the quarter, scaled by each rep's observed/expected.

    The rep factor is shrunk toward 1 with `k` pseudo deals, so a rep with little history keeps the team rate.
    """

    def __init__(self, rows: pd.DataFrame, k: float = 25.0):
        rows = rows.assign(bucket=week_bucket(rows["week"]))
        self.rates = rows.groupby(["category", "bucket"])["won_in_q"].mean().to_dict()
        rows["expected"] = [self.rates[(c, b)] for c, b in zip(rows.category, rows.bucket, strict=True)]
        g = rows.groupby(["rep_id", "category"]).agg(wins=("won_in_q", "sum"), exp=("expected", "sum"))
        self.lift = ((g.wins + k * 1.0) / (g.exp + k)).to_dict()

    def probs(self, open_deals: pd.DataFrame, week: int) -> np.ndarray:
        b = int(week_bucket(week))
        out = []
        for r in open_deals[["rep_id", "category"]].itertuples(index=False):
            cat = int(r.category) if r.category in (PIPELINE, BEST_CASE, COMMIT) else PIPELINE
            base = self.rates.get((cat, b), 0.0)
            out.append(min(base * self.lift.get((r.rep_id, cat), 1.0), 1.0))
        return np.asarray(out, float)

    def forecast(self, snaps: Snapshots, q: int, week: int) -> float:
        day = as_of_day(q, week)
        o = snaps.open_in_quarter(q, day)
        return snaps.won_to_date(q, day) + float((o["amount"].to_numpy() * self.probs(o, week)).sum())


def label_known(q: int) -> int:
    return q_end(q) + 1
