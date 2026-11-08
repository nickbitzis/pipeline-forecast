import pandas as pd
import pytest

from pipecast.backtest import METHODS, MID_WEEKS, bias, coverage, gate, mape


def test_mape_bias_coverage_math():
    f, a = [110, 90, 100], [100, 100, 100]
    assert mape(f, a) == pytest.approx((0.1 + 0.1 + 0) / 3)
    assert bias(f, a) == pytest.approx(0.0)
    assert bias([120, 120], [100, 100]) == pytest.approx(0.2)
    assert mape([50], [200]) == pytest.approx(0.75)
    assert coverage([0, 0, 5], [10, 10, 6], [5, 11, 5]) == pytest.approx(2 / 3)


def test_backtest_accounting(engine, snaps):
    bt = engine.backtest
    qs = bt.quarters
    assert qs == list(range(2, 8))
    assert len(bt.rows) == len(qs) * 13 * len(METHODS)
    assert len(bt.intervals) == len(qs) * 13
    for q, g in bt.rows.groupby("quarter"):
        assert g.actual.nunique() == 1 and g.actual.iloc[0] == pytest.approx(snaps.bookings(q))
    # every forecast includes what was already closed: commit is never below won to date
    assert (bt.intervals.p10 <= bt.intervals.p50).all() and (bt.intervals.p50 <= bt.intervals.p90).all()


def test_summary_recomputes_from_rows(engine):
    bt = engine.backtest
    s = bt.summary
    r = bt.rows[(bt.rows.method == "deal_model") & bt.rows.week.isin(MID_WEEKS)]
    assert s["overall"]["deal_model"]["mid_mape"] == pytest.approx(mape(r.forecast, r.actual))
    w7 = next(x for x in s["by_week"] if x["method"] == "rollup_commit" and x["week"] == 7)
    g = bt.rows[(bt.rows.method == "rollup_commit") & (bt.rows.week == 7)]
    assert w7["bias"] == pytest.approx(bias(g.forecast, g.actual))
    assert len(s["error_paths"]["deal_model"]) == len(bt.quarters)


def test_methods_converge_and_model_beats_rollup(engine):
    s = engine.backtest.summary
    bw = pd.DataFrame(s["by_week"]).set_index(["method", "week"])
    for m in ("deal_model", "rollup_commit", "stage_weighted"):
        assert bw.loc[(m, 13), "mape"] < bw.loc[(m, 1), "mape"]
    assert s["overall"]["deal_model"]["mid_mape"] < s["overall"]["rollup_commit"]["mid_mape"]
    assert s["overall"]["rollup_commit"]["mid_bias"] < 0  # commit calls run low
    assert s["overall"]["rollup_best_case"]["mid_bias"] > 0  # best case runs high


def test_interval_coverage_near_target(engine):
    assert 0.6 <= engine.backtest.summary["coverage"]["overall"] <= 0.97


def _summary(cov, model, roll):
    return {"coverage": {"overall": cov}, "overall": {"deal_model": {"mid_mape": model},
                                                      "rollup_commit": {"mid_mape": roll}}}


def test_gate_pass_and_fail():
    assert gate(_summary(0.8, 0.05, 0.2))[0]
    assert not gate(_summary(0.5, 0.05, 0.2))[0]
    assert not gate(_summary(0.97, 0.05, 0.2))[0]
    ok, msgs = gate(_summary(0.8, 0.3, 0.2))
    assert not ok and msgs[1].startswith("FAIL")


def test_backtest_never_uses_the_current_quarter(engine):
    assert engine.current_q not in engine.backtest.quarters
    assert engine.inputs.model.train_quarters == list(range(engine.current_q))
