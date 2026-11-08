import pandas as pd
import pytest

from pipecast.reps import AdjustedRollup, evidence, week_bucket


def test_planted_reps_are_detected(engine):
    reps = pd.DataFrame(engine.reps["reps"])
    flagged = reps[reps.flag != "accurate"]
    right = ((flagged.flag == "sandbagging") & (flagged.planted_tendency == "sandbagger")) | (
        (flagged.flag == "over-optimistic") & (flagged.planted_tendency == "optimistic"))
    assert right.mean() >= 0.8  # flags are mostly right
    planted = reps[reps.planted_tendency != "accurate"]
    # recall is limited by deal counts: the fixture has half the full deal flow, and small books cannot pass a test
    assert right.sum() >= 0.4 * len(planted)
    acc = reps[reps.planted_tendency == "accurate"]
    assert (acc.flag != "accurate").sum() <= 2  # few false alarms on 20+ accurate reps


def test_sandbaggers_beat_commit_and_optimists_miss(engine):
    reps = pd.DataFrame(engine.reps["reps"])
    sb = reps[reps.flag == "sandbagging"]
    oo = reps[reps.flag == "over-optimistic"]
    assert (sb.best_case_win_rate > sb.team_best_case_rate).all()
    assert (oo.commit_win_rate < oo.team_commit_rate).all()
    assert sb.commit_bias.mean() < oo.commit_bias.mean()


def test_evidence_text_names_the_numbers():
    r = {"flag": "sandbagging", "n_best_case_deals": 40, "best_case_win_rate": 0.5, "segment": "SMB",
         "team_best_case_rate": 0.2, "commit_bias": -0.4, "quarters": 8, "beat_commit_quarters": 8}
    t = evidence(r)
    assert "40 deals kept at best case closed at 50%" in t and "SMB team 20%" in t and "8 of 8" in t


def test_week_buckets():
    assert list(week_bucket([1, 2, 4, 5, 9, 10, 13])) == [0, 1, 1, 2, 2, 3, 3]


def test_adjusted_rollup_shrinks_toward_team():
    # team: commit deals in mid-quarter close 50%; rep A closes 10 of 10, rep B 0 of 2
    rows = pd.DataFrame({
        "rep_id": ["A"] * 10 + ["B"] * 2 + ["C"] * 8, "category": [2] * 20, "week": [6] * 20,
        "won_in_q": [1] * 10 + [0] * 2 + [0] * 8,
    })
    adj = AdjustedRollup(rows, k=10)
    assert adj.rates[(2, 2)] == pytest.approx(0.5)
    open_deals = pd.DataFrame({"rep_id": ["A", "B", "Z"], "category": [2, 2, 2]})
    p = adj.probs(open_deals, 6)
    assert p[0] == pytest.approx(0.5 * (10 + 10) / (5 + 10))  # A lifted
    assert p[1] == pytest.approx(0.5 * (0 + 10) / (1 + 10))  # B lowered a little, only 2 deals
    assert p[2] == pytest.approx(0.5)  # unknown rep keeps the team rate
