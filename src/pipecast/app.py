"""Forecast API and UI.

The engine (CRM, backtests, current quarter model) is built once at startup or loaded from a cached pickle.
Requests for another week of the current quarter re-run only the snapshot, scoring, and simulation for that week.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse

from .commentary import build_facts, get_writer, write_commentary
from .engine import EngineConfig, Forecaster
from .generator import CompanyConfig
from .logs import get_logger, log_event
from .metrics import Metrics

STATIC = Path(__file__).parent / "static"


def load_engine() -> Forecaster:
    path = Path(os.environ.get("PIPECAST_CACHE", "data/forecaster.pkl"))
    log = get_logger()
    if path.exists():
        try:
            eng = Forecaster.load(path)
            log_event(log, "engine loaded", path=str(path))
            return eng
        except Exception as exc:  # stale or incompatible pickle: rebuild
            log_event(log, "engine cache unreadable, rebuilding", path=str(path), error=str(exc))
    reps = int(os.environ.get("PIPECAST_REPS", "40"))
    scale = float(os.environ.get("PIPECAST_RATE_SCALE", "1.0"))
    cfg = EngineConfig(company=CompanyConfig(n_reps=reps, rate_scale=scale),
                       current_week=int(os.environ.get("PIPECAST_WEEK", "7")))
    eng = Forecaster(cfg).run()
    eng.save(path)
    log_event(log, "engine built", path=str(path), **eng.timings)
    return eng


def create_app(engine: Forecaster | None = None, llm_client=None) -> FastAPI:
    eng = engine or load_engine()
    metrics = Metrics()
    log = get_logger()
    bt = eng.backtest_report()
    fc0 = eng.public_forecast()
    metrics.set_gauge("pipecast_backtest_interval_coverage", "Monte Carlo P10 to P90 coverage on backtests.",
                      bt["coverage"]["overall"])
    for m, o in bt["overall"].items():
        metrics.set_gauge(f"pipecast_backtest_mid_mape_{m}", f"Mid-quarter MAPE on backtests: {o['label']}.",
                          round(o["mid_mape"], 6))
    metrics.set_gauge("pipecast_forecast_p50", "Current quarter Monte Carlo P50.", round(fc0["monte_carlo"]["p50"]))
    metrics.set_gauge("pipecast_forecast_commit", "Current quarter rep commit.", round(fc0["commit"]))
    metrics.set_gauge("pipecast_gate_pass", "1 when the backtest gate passes.", int(bt["gate"]["pass"]))

    app = FastAPI(title="Pipeline Forecast", version="1.0.0")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        dt = time.perf_counter() - t0
        path = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.observe_request(path, response.status_code, dt)
        response.headers["x-request-id"] = rid
        if path not in ("/healthz", "/metrics"):
            log_event(log, "request", request_id=rid, method=request.method, path=request.url.path,
                      status=response.status_code, ms=round(dt * 1000, 1))
        return response

    def week_or_400(week):
        try:
            return eng.public_forecast(week)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "quarter": fc0["quarter"], "week": fc0["week"], "deals": len(eng.crm.deals)}

    @app.get("/metrics")
    def prom():
        return PlainTextResponse(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/summary")
    def summary():
        return eng.summary()

    @app.get("/api/forecast")
    def forecast(week: int | None = Query(default=None, ge=1, le=13)):
        return week_or_400(week)

    @app.get("/api/backtest")
    def backtest():
        return bt

    @app.get("/api/reps")
    def reps():
        return eng.reps

    @app.get("/api/risk")
    def risk(week: int | None = Query(default=None, ge=1, le=13), limit: int = Query(default=25, ge=1, le=200)):
        return eng.risk(week, limit)

    @app.get("/api/deals/{deal_id}")
    def deal(deal_id: str):
        d = eng.crm.deals[eng.crm.deals.deal_id == deal_id.strip().upper()]
        if d.empty:
            raise HTTPException(404, f"deal {deal_id} not found")
        row = d.iloc[0]
        cutoff = eng.forecast()["_open"]["as_of"].iloc[0]
        hist = [e for e in eng.crm.deal_history(row.deal_id) if e["day"] < cutoff]
        return {"deal_id": row.deal_id, "account": row.account, "rep_id": row.rep_id, "segment": row.segment,
                "source": row.source, "history": hist, "note": "History up to the current as-of Monday only."}

    @app.get("/api/commentary")
    def commentary(week: int | None = Query(default=None, ge=1, le=13), provider: str | None = None):
        fc = week_or_400(week)
        facts = build_facts(fc, eng.risk(week, 10), bt, eng.reps)
        try:
            writer = get_writer(provider, client=llm_client)
        except Exception as exc:  # e.g. the anthropic extra is not installed
            raise HTTPException(503, f"provider unavailable: {exc}") from exc
        out = write_commentary(facts, writer)
        metrics.commentary_written(out["provider"], out["valid"])
        return {**out, "facts": facts}

    return app
