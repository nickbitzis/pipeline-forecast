import numpy as np
import pytest

from pipecast.fiscal import q_end
from pipecast.snapshots import Snapshots, replay_one

from .conftest import tiny_crm


def test_tiny_snapshot_by_hand():
    s = Snapshots(tiny_crm())
    snap = s.at(32).set_index("deal_id")
    d1, d2 = snap.loc["D1"], snap.loc["D2"]
    assert d1.stage == 4 and d1.days_in_stage == 12 and d1.amount == 13000 and d1.amount0 == 10000
    assert d1.close_day == 45 and d1.pushes == 1 and d1.category == 2
    assert d1.acts_total == 2 and d1.acts_14d == 1 and d1.days_since_act == 14
    assert d1.late_increase == 1 and d1.is_open
    assert d2.stage == 2 and d2.pushes == 1 and d2.close_day == 50 and d2.category == 1
    assert d2.days_since_act == 18 and d2.acts_14d == 0


def test_events_on_the_as_of_day_are_not_visible():
    s = Snapshots(tiny_crm())
    assert "D2" not in set(s.at(10).deal_id)  # created on day 10, the snapshot is the morning of day 10
    assert s.at(40).set_index("deal_id").loc["D1", "stage"] == 4
    assert s.at(41).set_index("deal_id").loc["D1", "stage"] == 6


def test_closed_deals_and_bookings_by_hand():
    s = Snapshots(tiny_crm())
    after = s.at(61).set_index("deal_id")
    assert after.loc["D1", "is_won"] and after.loc["D2", "is_lost"]
    assert after.loc["D1", "amount"] == 12000 and after.loc["D1", "close_day"] == 40
    assert s.bookings(0) == 12000
    assert s.won_to_date(0, 40) == 0 and s.won_to_date(0, 41) == 12000


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_vectorized_matches_slow_replay(crm, snaps, seed):
    rng = np.random.default_rng(seed)
    for _ in range(15):
        day = int(rng.integers(0, 8 * 91))
        snap = snaps.at(day).set_index("deal_id")
        deal = rng.choice(snap.index.to_numpy())
        slow = replay_one(crm, deal, day)
        row = snap.loc[deal]
        for k in ("stage", "amount", "close_day", "category", "pushes", "acts_total", "acts_14d"):
            assert row[k] == slow[k], (deal, day, k)


def test_bookings_match_hidden_truth(crm, snaps):
    d = crm.deals
    for q in (2, 5):
        truth = d[d.won & d.resolve_day.between(q * 91, q_end(q))].final_amount.sum()
        assert snaps.bookings(q) == pytest.approx(truth)


def test_snapshot_never_exposes_hidden_columns(snaps):
    cols = set(snaps.at(300).columns)
    assert not cols & {"true_p", "won", "resolve_day", "final_amount", "tendency"}


def test_open_in_quarter_respects_close_date(snaps):
    o = snaps.open_in_quarter(5, 5 * 91 + 21)
    assert (o.close_day <= q_end(5)).all() and o.is_open.all()
    assert (o.created_day < 5 * 91 + 21).all()
