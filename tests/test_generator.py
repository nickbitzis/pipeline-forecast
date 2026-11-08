import numpy as np

from pipecast.fiscal import QUARTER_DAYS, as_of_day, label, q_end, q_start, quarter_of, to_date
from pipecast.generator import REGIONS, CompanyConfig, generate


def test_seeded_and_deterministic():
    a = generate(CompanyConfig(n_reps=10, quarters=3, seed=3))
    b = generate(CompanyConfig(n_reps=10, quarters=3, seed=3))
    assert a.deals.equals(b.deals) and a.cat_ev.equals(b.cat_ev)
    c = generate(CompanyConfig(n_reps=10, quarters=3, seed=4))
    assert not a.deals.equals(c.deals)


def test_team_shape(crm):
    r = crm.reps
    assert len(r) == 40 and set(r.region) == set(REGIONS)
    assert (r.groupby("region").size() == 8).all()
    assert set(r.tendency) == {"accurate", "optimistic", "sandbagger"}


def test_outcomes_are_plausible(crm):
    d = crm.deals
    assert 0.2 < d.won.mean() < 0.4
    # the hidden propensity actually drives outcomes
    assert d[d.true_p > 0.5].won.mean() > d[d.true_p < 0.2].won.mean() + 0.3
    assert (d.resolve_day > d.created_day).all()


def test_event_logs_are_consistent(crm):
    assert set(crm.cat_ev.category.unique()) <= {-1, 0, 1, 2, 3}
    assert set(crm.stage_ev.stage.unique()) <= {1, 2, 3, 4, 5, 6, 7}
    final = crm.stage_ev.sort_values("day", kind="stable").groupby("deal_id").stage.last()
    won = crm.deals.set_index("deal_id").won
    assert ((final == 6) == won.reindex(final.index)).all()
    # every push moves the close date later than the previous one
    ce = crm.close_ev.sort_values(["deal_id", "day"], kind="stable")
    prev = ce.groupby("deal_id").close_day.shift(1)
    pushes = ce[ce.push == 1]
    assert (pushes.close_day > prev[pushes.index]).all()


def test_activity_dates_inside_deal_life(crm):
    a = crm.acts.merge(crm.deals[["deal_id", "created_day", "resolve_day"]])
    assert (a.day >= a.created_day).all() and (a.day < a.resolve_day).all()


def test_fiscal_calendar():
    assert q_start(2) == 2 * QUARTER_DAYS and q_end(2) == 3 * QUARTER_DAYS - 1
    assert quarter_of(q_end(4)) == 4 and quarter_of(q_end(4) + 1) == 5
    assert as_of_day(3, 1) == q_start(3) and as_of_day(3, 13) == q_start(3) + 84
    assert label(8) == "FY26 Q4" and label(0) == "FY24 Q4"
    assert to_date(as_of_day(8, 1)).isoformat() == "2026-09-28"
    assert to_date(0).weekday() == 0
    try:
        as_of_day(1, 14)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_growth_shows_in_pipeline(crm):
    created = crm.deals.created_day
    early = np.sum((created >= 0) & (created < 2 * QUARTER_DAYS))
    late = np.sum((created >= 6 * QUARTER_DAYS) & (created < 8 * QUARTER_DAYS))
    assert late > early
