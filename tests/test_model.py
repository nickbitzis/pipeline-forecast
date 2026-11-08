import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from pipecast.fiscal import as_of_day
from pipecast.generator import CRM
from pipecast.model import FEATURES, DealModel, brier, ece, features, reliability, training_rows
from pipecast.snapshots import Snapshots


@pytest.fixture(scope="module")
def rows(snaps):
    return training_rows(snaps, range(0, 6))


def test_training_rows_shape(rows):
    assert set(FEATURES) <= set(rows.columns)
    assert set(rows.quarter.unique()) == set(range(6))
    assert 0.05 < rows.y.mean() < 0.4
    assert (rows.label_known_day == (rows.quarter + 1) * 91).all()


def test_time_split_refuses_future_quarters(rows):
    with pytest.raises(ValueError):
        DealModel(forecast_quarter=3, train_quarters=[0, 1, 2, 3]).fit(rows)
    with pytest.raises(ValueError):
        DealModel(forecast_quarter=2, train_quarters=[1]).fit(rows)
    m = DealModel(forecast_quarter=4, train_quarters=[0, 1, 2, 3]).fit(rows)
    assert m.n_train == int((rows.quarter < 3).sum()) and m.n_cal == int((rows.quarter == 3).sum())


def test_features_ignore_everything_after_the_as_of_day(crm):
    """Rewrite every event after day d: the snapshot features on day d must not change."""
    d = as_of_day(5, 6)
    base = features(Snapshots(crm).at(d).query("is_open"), 5)

    def scramble(ev, col):
        ev = ev.copy()
        late = ev.day >= d
        ev.loc[late, col] = ev.loc[late, col].to_numpy()[::-1]
        return ev

    alt = CRM(crm.reps, crm.deals.assign(won=~crm.deals.won), scramble(crm.stage_ev, "stage"),
              scramble(crm.amount_ev, "amount"), scramble(crm.close_ev, "close_day"),
              scramble(crm.cat_ev, "category"), crm.acts[crm.acts.day < d], crm.config)
    other = features(Snapshots(alt).at(d).query("is_open"), 5)
    pd.testing.assert_frame_equal(base, other)


def test_model_beats_base_rate_out_of_time(snaps, rows):
    m = DealModel(forecast_quarter=6, train_quarters=list(range(6))).fit(rows)
    test = training_rows(snaps, [6])
    p = m.predict(test)
    base = np.full(len(test), rows.y.mean())
    assert brier(p, test.y) < 0.85 * brier(base, test.y)
    assert ece(p, test.y) < 0.03
    assert ((p >= 0) & (p <= 1)).all()


def test_ece_and_brier_by_hand():
    p = np.array([0.1, 0.1, 0.9, 0.9])
    y = np.array([0, 0, 1, 1])
    assert brier(p, y) == pytest.approx(0.01)
    assert ece(p, y) == pytest.approx(0.1)
    assert ece(np.array([0.5, 0.5]), np.array([0, 1])) == pytest.approx(0.0)
    rel = reliability(p, y)
    assert [r["n"] for r in rel] == [2, 2] and rel[1]["observed"] == 1.0


def test_isotonic_fixes_an_overconfident_score():
    rng = np.random.default_rng(0)
    true = rng.uniform(0, 1, 20000)
    y = (rng.random(20000) < true).astype(int)
    raw = true**0.4  # systematically too high
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw[:10000], y[:10000])
    assert ece(iso.predict(raw[10000:]), y[10000:]) < 0.3 * ece(raw[10000:], y[10000:])


def test_backtest_calibration_improves_or_holds(engine):
    c = engine.backtest.summary["calibration"]
    assert c["calibrated"]["ece"] <= c["raw"]["ece"] + 0.005
    assert c["calibrated"]["ece"] < 0.03
