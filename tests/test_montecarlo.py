import numpy as np
import pytest
from scipy.stats import binom

from pipecast.montecarlo import prob_at_least, simulate, summarize


def test_identical_deals_match_binomial_percentiles():
    n, p, a = 50, 0.3, 10_000.0
    sims = simulate(np.full(n, a), np.full(n, p), won_to_date=5_000, n_sims=40_000, seed=1)
    s = summarize(sims)
    for q, key in ((0.1, "p10"), (0.5, "p50"), (0.9, "p90")):
        assert s[key] == pytest.approx(5_000 + a * binom.ppf(q, n, p), abs=a)
    assert s["mean"] == pytest.approx(5_000 + a * n * p, rel=0.01)
    assert s["std"] == pytest.approx(a * np.sqrt(n * p * (1 - p)), rel=0.03)


def test_mixed_deals_mean_and_variance():
    amounts = np.array([100.0, 50.0, 20.0, 5.0])
    probs = np.array([0.9, 0.5, 0.2, 0.7])
    sims = simulate(amounts, probs, n_sims=100_000, seed=2)
    assert sims.mean() == pytest.approx((amounts * probs).sum(), rel=0.01)
    assert sims.var() == pytest.approx((amounts**2 * probs * (1 - probs)).sum(), rel=0.03)


def test_certain_deals_have_no_spread():
    sims = simulate([10, 20, 30], [1, 0, 1], won_to_date=7, n_sims=500)
    assert (sims == 47).all()
    s = summarize(sims)
    assert s["p10"] == s["p90"] == 47 and sum(s["histogram"]["counts"]) == 500


def test_seeded_and_chunked():
    a = simulate(np.ones(30), np.full(30, 0.4), n_sims=5000, seed=9, chunk=777)
    b = simulate(np.ones(30), np.full(30, 0.4), n_sims=5000, seed=9, chunk=777)
    assert (a == b).all() and len(a) == 5000


def test_prob_at_least():
    assert prob_at_least(np.array([1, 2, 3, 4]), 3) == 0.5
    assert prob_at_least(np.array([1, 2]), 9) == 0.0
