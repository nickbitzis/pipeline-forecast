import json

import pytest
from fastapi.testclient import TestClient

from pipecast.app import create_app

from .test_commentary import FakeClient


@pytest.fixture(scope="module")
def client(engine):
    return TestClient(create_app(engine, llm_client=FakeClient("Commit what the model says.")))


def test_health_and_index(client):
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["week"] == 7
    assert "x-request-id" in r.headers
    page = client.get("/")
    assert page.status_code == 200 and "Pipeline Forecast" in page.text


def test_forecast_all_methods(client):
    f = client.get("/api/forecast").json()
    assert f["quarter"] == "FY26 Q4" and f["week"] == 7
    assert {m["method"] for m in f["methods"]} == {"rollup_commit", "rollup_best_case", "adjusted_rollup",
                                                   "stage_weighted", "deal_model", "monte_carlo_p50", "time_series"}
    mc = f["monte_carlo"]
    assert f["won_to_date"] <= mc["p10"] <= mc["p50"] <= mc["p90"]
    assert sum(mc["histogram"]["counts"]) == mc["n_sims"]
    assert not any(k.startswith("_") for k in f)
    json.dumps(f)


def test_other_weeks_and_validation(client):
    early, late = client.get("/api/forecast?week=2").json(), client.get("/api/forecast?week=12").json()
    assert early["won_to_date"] <= late["won_to_date"]
    spread = lambda f: f["monte_carlo"]["p90"] - f["monte_carlo"]["p10"]  # noqa: E731
    assert spread(late) < spread(early)
    assert client.get("/api/forecast?week=14").status_code == 422
    assert client.get("/api/forecast?week=0").status_code == 422


def test_backtest_reps_risk(client):
    b = client.get("/api/backtest").json()
    assert "gate" in b and len(b["by_week"]) == 7 * 13
    reps = client.get("/api/reps").json()
    assert len(reps["reps"]) == 40
    risk = client.get("/api/risk?limit=5").json()
    assert len(risk["deals"]) == 5


def test_deal_history_stops_at_as_of(client, engine):
    deal = engine.risk(limit=1)["deals"][0]["deal_id"]
    h = client.get(f"/api/deals/{deal}").json()
    cutoff = engine.forecast()["_open"]["as_of"].iloc[0]
    assert h["history"] and all(e["day"] < cutoff for e in h["history"])
    assert client.get("/api/deals/NOPE").status_code == 404


def test_commentary_providers(client):
    t = client.get("/api/commentary").json()
    assert t["provider"] == "template" and t["valid"] and t["facts"]["quarter"] == "FY26 Q4"
    c = client.get("/api/commentary?provider=claude").json()
    assert c["provider"] == "claude" and c["valid"]


def test_metrics(client):
    client.get("/api/commentary")
    m = client.get("/metrics").text
    assert "pipecast_http_requests_total" in m and "pipecast_backtest_interval_coverage" in m
    assert 'pipecast_commentary_total{provider="template",valid="true"}' in m
    assert "pipecast_http_request_duration_seconds_count" in m


def test_summary_endpoint(client):
    s = client.get("/api/summary").json()
    assert s["reps"] == 40 and s["regions"] == 5 and len(s["history"]) == 8
    assert all(h["bookings"] > 0 for h in s["history"])


def test_unknown_provider_falls_back_to_template(client):
    assert client.get("/api/commentary?provider=nope").json()["provider"] == "template"
