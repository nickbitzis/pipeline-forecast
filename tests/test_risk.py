import pandas as pd

from pipecast.risk import flag_deals, risk_table, stage_norms

from .conftest import tiny_crm


def _open(**kw):
    base = {"deal_id": "X", "segment": "SMB", "stage": 3, "pushes": 0, "days_in_stage": 5, "days_since_act": 2,
            "late_increase": 0, "close_day": 50, "amount": 10000.0, "category": 2}
    base.update(kw)
    return base


def test_each_flag_fires_alone():
    df = pd.DataFrame([_open(), _open(pushes=2), _open(days_in_stage=40), _open(days_since_act=15),
                       _open(late_increase=1), _open(stage=2, close_day=88)])
    f = flag_deals(df, 0, {("SMB", 3): 30.0, ("SMB", 2): 100.0})
    assert f.sum(axis=1).tolist() == [0, 1, 1, 1, 1, 1]
    assert f.iloc[1].pushed_twice and f.iloc[2].stuck_in_stage and f.iloc[3].no_activity_14d
    assert f.iloc[4].late_amount_increase and f.iloc[5].quarter_end_early_stage


def test_quarter_end_flag_needs_early_stage():
    df = pd.DataFrame([_open(stage=4, close_day=90), _open(stage=1, close_day=80)])
    f = flag_deals(df, 0, {})
    assert not f.quarter_end_early_stage.any()


def test_stage_norms_from_completed_stints():
    crm = tiny_crm()
    norms = stage_norms(crm.stage_ev, crm.deals, 100, pct=50)
    assert norms == {}  # fewer than 5 stints per stage: no norm, so nothing is flagged as stuck
    ev = pd.DataFrame({"deal_id": [f"D{i}" for i in range(6) for _ in range(2)],
                       "day": [x for i in range(6) for x in (0, 10 * (i + 1))], "stage": [1, 2] * 6})
    deals = pd.DataFrame({"deal_id": [f"D{i}" for i in range(6)], "segment": ["SMB"] * 6})
    assert stage_norms(ev, deals, 100, pct=50) == {("SMB", 1): 35.0}


def test_value_at_risk_only_counts_the_reps_call():
    df = pd.DataFrame([_open(category=2, amount=100.0), _open(category=1, amount=100.0),
                       _open(category=0, amount=1000.0)])
    t = risk_table(df, [0.75, 0.2, 0.5], 0, {}).sort_index()
    assert t.value_at_risk.tolist() == [25.0, 80.0, 0.0]
    assert t.expected_value.tolist() == [75.0, 20.0, 500.0]
    assert t.swing.iloc[2] == 500.0


def test_engine_risk_list(engine):
    r = engine.risk(limit=10)
    assert len(r["deals"]) == 10
    v = [d["value_at_risk"] for d in r["deals"]]
    assert v == sorted(v, reverse=True)
    assert all(d["category"] in ("commit", "best case", "pipeline") for d in r["deals"])
    assert r["flagged_deals"] <= r["deals_in_quarter"]
