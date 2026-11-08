"""Deal-level model: probability that an open deal closes won inside the forecast quarter.

One training row per (open deal, Monday of a past quarter). The label is whether that deal closed won before that
quarter ended. Features come only from the snapshot on that Monday, so the model sees exactly what a forecaster
would have seen. The time split is strict: a model used for quarter q trains on quarters before q, and isotonic
calibration is fitted on the most recent of those quarters, never on the quarter being forecast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from .fiscal import WEEKS, as_of_day, q_end, quarter_of
from .generator import SEGMENTS, SOURCES
from .snapshots import Snapshots

FEATURES = [
    "stage", "log_amount", "segment_code", "source_code", "category", "days_in_stage", "age", "pushes",
    "amount_change_pct", "late_increase", "n_amount_changes", "acts_14d", "days_since_act", "act_rate",
    "close_in_q", "days_to_close", "days_left", "week",
]
CATEGORICAL = ["segment_code", "source_code"]


def features(snap: pd.DataFrame, q: int) -> pd.DataFrame:
    """Model inputs for open deals in a snapshot, relative to forecast quarter q."""
    d = snap["as_of"].iloc[0] if len(snap) else 0
    X = pd.DataFrame(index=snap.index)
    X["stage"] = snap["stage"]
    X["log_amount"] = np.log(snap["amount"])
    X["segment_code"] = snap["segment"].map({s: i for i, s in enumerate(SEGMENTS)})
    X["source_code"] = snap["source"].map({s: i for i, s in enumerate(SOURCES)})
    X["category"] = snap["category"]
    X["days_in_stage"] = snap["days_in_stage"]
    X["age"] = snap["age"]
    X["pushes"] = snap["pushes"]
    X["amount_change_pct"] = snap["amount_change_pct"]
    X["late_increase"] = snap["late_increase"]
    X["n_amount_changes"] = snap["n_amount_changes"]
    X["acts_14d"] = snap["acts_14d"]
    X["days_since_act"] = snap["days_since_act"]
    X["act_rate"] = snap["acts_total"] / np.maximum(snap["age"], 7) * 7
    X["close_in_q"] = (snap["close_day"] <= q_end(q)).astype(int)
    X["days_to_close"] = snap["close_day"] - d
    X["days_left"] = q_end(q) + 1 - d
    X["week"] = (d - q * 91) // 7 + 1
    return X[FEATURES]


def training_rows(snaps: Snapshots, quarters) -> pd.DataFrame:
    """Open deals on every Monday of each past quarter, with the in-quarter outcome as the label."""
    frames = []
    for q in quarters:
        won = set(snaps.won_deals(q).deal_id)
        for w in range(1, WEEKS + 1):
            day = as_of_day(q, w)
            o = snaps.open_deals(day)
            X = features(o, q)
            X["y"] = o.deal_id.isin(won).astype(int).to_numpy()
            X["quarter"] = q
            X["label_known_day"] = q_end(q) + 1
            frames.append(X)
    return pd.concat(frames, ignore_index=True)


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error with equal width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(err)


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability(p, y, bins: int = 10) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.any():
            out.append({"bin": f"{b / bins:.1f}-{(b + 1) / bins:.1f}", "n": int(m.sum()),
                        "predicted": round(float(p[m].mean()), 4), "observed": round(float(y[m].mean()), 4)})
    return out


@dataclass
class DealModel:
    """Gradient boosting plus isotonic calibration, trained for one forecast quarter."""

    forecast_quarter: int
    train_quarters: list[int]
    seed: int = 0

    def fit(self, rows: pd.DataFrame) -> DealModel:
        rows = rows[rows.quarter.isin(self.train_quarters)]
        if rows["label_known_day"].max() > self.forecast_quarter * 91:
            raise ValueError("training labels would not be known when the forecast quarter starts")
        cal_q = max(self.train_quarters)
        fit_rows, cal_rows = rows[rows.quarter < cal_q], rows[rows.quarter == cal_q]
        if fit_rows.empty:
            raise ValueError("need at least two training quarters (one to fit, one to calibrate)")
        cat_mask = [f in CATEGORICAL for f in FEATURES]
        self.clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=0.1,
            categorical_features=cat_mask, random_state=self.seed)
        self.clf.fit(fit_rows[FEATURES], fit_rows["y"])
        raw = self.clf.predict_proba(cal_rows[FEATURES])[:, 1]
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(raw, cal_rows["y"])
        self.n_train = int(len(fit_rows))
        self.n_cal = int(len(cal_rows))
        return self

    def predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return self.clf.predict_proba(X[FEATURES])[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return self.iso.predict(self.predict_raw(X))

    def importances(self, X: pd.DataFrame, y: np.ndarray, n_repeats: int = 3) -> list[dict]:
        from sklearn.inspection import permutation_importance

        r = permutation_importance(self.clf, X[FEATURES], y, n_repeats=n_repeats, random_state=0,
                                   scoring="neg_log_loss")
        order = np.argsort(-r.importances_mean)
        return [{"feature": FEATURES[i], "importance": round(float(r.importances_mean[i]), 5)} for i in order]


def quarter_for(day: int) -> int:
    return quarter_of(day)
