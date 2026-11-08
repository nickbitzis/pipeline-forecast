"""Monte Carlo over deal probabilities: a distribution of quarter bookings instead of one number.

Each simulated quarter flips every open deal with its calibrated close probability and adds up the winners on top
of what is already closed. Deals are treated as independent, which is the main reason real intervals come out too
narrow; the backtest measures how often the P10 to P90 range actually contained the result.
"""

from __future__ import annotations

import numpy as np


def simulate(amounts, probs, won_to_date: float = 0.0, n_sims: int = 10_000, seed: int = 0,
             chunk: int = 2_000) -> np.ndarray:
    amounts = np.asarray(amounts, float)
    probs = np.clip(np.asarray(probs, float), 0.0, 1.0)
    rng = np.random.default_rng(seed)
    out = np.empty(n_sims)
    for start in range(0, n_sims, chunk):
        k = min(chunk, n_sims - start)
        wins = rng.random((k, len(amounts))) < probs
        out[start:start + k] = wins @ amounts
    return out + won_to_date


def summarize(sims: np.ndarray, bins: int = 40) -> dict:
    p10, p50, p90 = np.percentile(sims, [10, 50, 90])
    lo, hi = float(sims.min()), float(sims.max())
    if hi <= lo:
        hi = lo + 1.0
    counts, edges = np.histogram(sims, bins=bins, range=(lo, hi))
    return {
        "p10": float(p10), "p50": float(p50), "p90": float(p90), "mean": float(sims.mean()),
        "std": float(sims.std()), "n_sims": int(len(sims)),
        "histogram": {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]},
    }


def prob_at_least(sims: np.ndarray, target: float) -> float:
    return float(np.mean(sims >= target))
