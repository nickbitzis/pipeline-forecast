import numpy as np
import pandas as pd
import pytest

from pipecast import methods as M
from pipecast.fiscal import as_of_day
from pipecast.snapshots import Snapshots

from .conftest import tiny_crm


def test_stage_conversion_rates_by_hand():
    rows = pd.DataFrame({"stage": [1, 1, 3, 3, 3, 5], "amount": [100, 300, 50, 50, 100, 10],
                         "won_in_q": [1, 0, 1, 0, 1, 1]})
    r = M.stage_conversion_rates(rows, prior=0.1)
    assert r[1] == pytest.approx(0.25)  # 100 of 400 dollars
    assert r[3] == pytest.approx(0.75)  # 150 of 200
    assert r[5] == 1.0 and r[2] == 0.1 and r[4] == 0.1


def test_stage_weighted_on_tiny_crm():
    s = Snapshots(tiny_crm())
    # morning of day 32: D1 stage 4 13,000 closes day 45; D2 stage 2 30,000 closes day 50 (both in quarter 0)
    f = M.stage_weighted(s, 0, 32, {1: 0, 2: 0.1, 3: 0, 4: 0.5, 5: 0})
    assert f == pytest.approx(13000 * 0.5 + 30000 * 0.1)


def test_rep_rollup_on_tiny_crm():
    s = Snapshots(tiny_crm())
    r = M.rep_rollup(s, 0, 32)
    assert r["won"] == 0 and r["commit"] == 13000 and r["best_case"] == 43000


def test_rollup_ordering(snaps):
    for w in (1, 7, 13):
        r = M.rep_rollup(snaps, 6, as_of_day(6, w))
        assert r["won"] <= r["commit"] <= r["best_case"]


def test_holt_on_a_straight_line():
    assert M.holt([10, 20, 30, 40, 50], alpha=1.0, beta=1.0) == pytest.approx(60)
    assert M.holt([5.0, 7.0]) == pytest.approx(6.0)
    with pytest.raises(ValueError):
        M.holt([])


def test_linearity_rises_through_the_quarter(snaps):
    shape = M.linearity(snaps, [3, 4, 5])
    vals = [shape[w] for w in range(1, 14)]
    assert vals[0] == pytest.approx(0.0)
    assert all(b >= a for a, b in zip(vals, vals[1:], strict=False)) and vals[-1] < 1.0


def test_stage_rates_rise_with_stage(snaps):
    rows = M.history_rows(snaps, [4, 5, 6])
    r = M.stage_conversion_rates(rows)
    assert r[1] < r[3] < r[5]


def test_not_yet_created_shrinks_through_quarter(snaps):
    h = M.not_yet_created(snaps, 5)
    assert h[1] >= h[7] >= h[13] >= 0
    npl = M.NewPipeline(snaps, [2, 3, 4, 5])
    draws = npl.draw(3, 5000, np.random.default_rng(0))
    assert (draws >= 0).all() and draws.mean() == pytest.approx(npl.mean[3], rel=0.05)
