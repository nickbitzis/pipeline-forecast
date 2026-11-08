import pandas as pd
import pytest

from pipecast.engine import EngineConfig, Forecaster
from pipecast.generator import CRM, CompanyConfig


@pytest.fixture(scope="session")
def engine():
    """A smaller company (half the deal flow) so the whole suite stays fast. Same code path as the server."""
    cfg = EngineConfig(company=CompanyConfig(rate_scale=0.5, seed=11), n_sims=2000, backtest_sims=1000)
    return Forecaster(cfg).run()


@pytest.fixture(scope="session")
def crm(engine):
    return engine.crm


@pytest.fixture(scope="session")
def snaps(engine):
    return engine.snaps


def tiny_crm() -> CRM:
    """Two hand-written deals with a known history."""
    reps = pd.DataFrame([{"rep_id": "R01", "name": "Test R.", "region": "EMEA", "segment": "SMB",
                          "tendency": "accurate", "skill": 0.0}])
    deals = pd.DataFrame([
        {"deal_id": "D1", "account": "A", "rep_id": "R01", "region": "EMEA", "segment": "SMB", "source": "inbound",
         "created_day": 0, "true_p": 0.5, "won": True, "resolve_day": 40, "final_amount": 12000.0},
        {"deal_id": "D2", "account": "B", "rep_id": "R01", "region": "EMEA", "segment": "SMB", "source": "outbound",
         "created_day": 10, "true_p": 0.2, "won": False, "resolve_day": 60, "final_amount": 30000.0},
    ])
    stage_ev = pd.DataFrame([("D1", 0, 1), ("D1", 5, 2), ("D1", 20, 4), ("D1", 40, 6),
                             ("D2", 10, 1), ("D2", 15, 2), ("D2", 60, 7)], columns=["deal_id", "day", "stage"])
    amount_ev = pd.DataFrame([("D1", 0, 10000.0, 0), ("D1", 30, 13000.0, 1), ("D1", 40, 12000.0, -1),
                              ("D2", 10, 30000.0, 0)], columns=["deal_id", "day", "amount", "change"])
    close_ev = pd.DataFrame([("D1", 0, 25, 0), ("D1", 26, 45, 1), ("D1", 40, 40, 0),
                             ("D2", 10, 30, 0), ("D2", 31, 50, 1), ("D2", 51, 80, 1), ("D2", 60, 60, 0)],
                            columns=["deal_id", "day", "close_day", "push"])
    cat_ev = pd.DataFrame([("D1", 0, 0), ("D1", 21, 2), ("D1", 40, 3), ("D2", 10, 0), ("D2", 17, 1), ("D2", 60, -1)],
                          columns=["deal_id", "day", "category"])
    acts = pd.DataFrame([("D1", 2), ("D1", 18), ("D1", 33), ("D2", 12), ("D2", 14)], columns=["deal_id", "day"])
    return CRM(reps, deals, stage_ev, amount_ev, close_ev, cat_ev, acts, CompanyConfig(n_reps=1))
