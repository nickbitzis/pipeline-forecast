"""Command line: serve, eval.

    uv run pipecast serve               # API and UI on http://127.0.0.1:8854 (builds on first start, then cached)
    uv run pipecast eval                # full backtest, writes reports/backtest.json and reports/summary.md
    uv run pipecast eval --gate         # same, exit 1 if interval coverage or the model vs roll-up check fails
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float):
        return round(o, 6)
    return o


def summary_markdown(eng) -> str:
    from .commentary import money
    from .engine import backtest_table

    bt = eng.backtest_report()
    fc = eng.public_forecast()
    lines = ["# Backtest summary", "", f"Quarters replayed: {', '.join(bt['quarters'])}. Error is MAPE against "
             "closed won bookings, by week of quarter.", "", backtest_table(bt), ""]
    c = bt["calibration"]
    lines += [f"Monte Carlo P10 to P90 coverage: {bt['coverage']['overall']:.1%} overall, "
              f"{bt['coverage']['mid']:.1%} mid-quarter.",
              f"Deal model calibration on {c['n']:,} out-of-time deal weeks: Brier {c['raw']['brier']:.4f} raw, "
              f"{c['calibrated']['brier']:.4f} calibrated; ECE {c['raw']['ece']:.4f} raw, "
              f"{c['calibrated']['ece']:.4f} calibrated.", "", "## Gate", ""]
    lines += [f"- {x}" for x in bt["gate"]["checks"]]
    mc = fc["monte_carlo"]
    lines += ["", f"## Current quarter: {fc['quarter']}, week {fc['week']}", "",
              f"Won to date {money(fc['won_to_date'])}. Rep commit {money(fc['commit'])}, best case "
              f"{money(fc['best_case'])}. Monte Carlo P10 {money(mc['p10'])}, P50 {money(mc['p50'])}, "
              f"P90 {money(mc['p90'])}.", "", "| Method | Forecast | Backtest MAPE this week |", "|---|---:|---:|"]
    lines += [f"| {r['label']} | {money(r['forecast'])} | {r['backtest_mape']:.1%} |" for r in fc["methods"]]
    reps = eng.reps["reps"]
    lines += ["", "## Rep flags", "", "| Rep | Segment | Flag | Evidence |", "|---|---|---|---|"]
    lines += [f"| {r['name']} | {r['segment']} | {r['flag']} | {r['evidence']} |" for r in reps
              if r["flag"] != "accurate"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pipecast")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8854")))
    e = sub.add_parser("eval")
    e.add_argument("--reps", type=int, default=40)
    e.add_argument("--rate-scale", type=float, default=1.0)
    e.add_argument("--seed", type=int, default=7)
    e.add_argument("--out", default="reports")
    e.add_argument("--gate", action="store_true", help="exit 1 if the backtest gate fails")
    e.add_argument("--save", default=None, help="also pickle the engine here (the server loads it)")
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        uvicorn.run("pipecast.app:create_app", factory=True, host=args.host, port=args.port, access_log=False)
        return 0

    from .engine import EngineConfig, Forecaster
    from .generator import CompanyConfig

    cfg = EngineConfig(company=CompanyConfig(n_reps=args.reps, rate_scale=args.rate_scale, seed=args.seed))
    eng = Forecaster(cfg).run()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bt = eng.backtest_report()
    report = {"summary": eng.summary(), "backtest": bt, "forecast": eng.public_forecast(),
              "reps": eng.reps, "risk": eng.risk(limit=15)}
    (out / "backtest.json").write_text(json.dumps(_clean(report), indent=1))
    md = summary_markdown(eng)
    (out / "summary.md").write_text(md)
    if args.save:
        eng.save(Path(args.save))
    print(md)
    if args.gate and not bt["gate"]["pass"]:
        print("GATE FAILED")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
