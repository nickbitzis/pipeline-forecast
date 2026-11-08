"""Forecast call commentary for a CRO, written only from computed numbers.

The facts are built from the forecast, risk list, backtest, and rep accuracy. A writer (offline template by default,
Claude when PIPECAST_LLM_PROVIDER=claude) turns them into a short call. A validator then checks that every number in
the text matches a computed fact, allowing only the rounding the text shows ("$4.2M" matches 4,213,500). If Claude's
text fails, the service returns the template text instead and reports what failed.
"""

from __future__ import annotations

import json
import os
import re

NUM = re.compile(r"(?<![A-Za-z0-9.])(\$)?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(\s?%|[KMB]\b)?")
UNITS = {"K": 1e3, "M": 1e6, "B": 1e9}


def money(v: float) -> str:
    v = float(v)
    if abs(v) >= 1e6:
        return f"${v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"${v / 1e3:.0f}K"
    return f"${v:.0f}"


def pct(v: float) -> str:
    return f"{100 * float(v):.0f}%"


def build_facts(fc: dict, risk: dict, backtest: dict, reps: dict) -> dict:
    """Plain dict of display strings. Everything the commentary may say comes from here."""
    mc = fc["monte_carlo"]
    m = {r["method"]: r for r in fc["methods"]}
    ov = backtest["overall"]
    top = risk["deals"][:3]
    flagged = [r for r in reps["reps"] if r["flag"] != "accurate"]
    return {
        "quarter": fc["quarter"], "week": str(fc["week"]), "weeks_in_quarter": str(fc["weeks"]),
        "quarter_end": fc["quarter_end"],
        "won_to_date": money(fc["won_to_date"]),
        "rep_commit": money(fc["commit"]), "rep_best_case": money(fc["best_case"]),
        "deal_model": money(m["deal_model"]["forecast"]),
        "adjusted_rollup": money(m["adjusted_rollup"]["forecast"]),
        "p10": money(mc["p10"]), "p50": money(mc["p50"]), "p90": money(mc["p90"]),
        "p_hit_commit": pct(mc["p_hit_commit"]), "p_hit_best_case": pct(mc["p_hit_best_case"]),
        "model_mid_mape": pct(ov["deal_model"]["mid_mape"]), "commit_mid_mape": pct(ov["rollup_commit"]["mid_mape"]),
        "commit_mid_bias": pct(ov["rollup_commit"]["mid_bias"]),
        "interval_coverage": pct(backtest["coverage"]["overall"]),
        "backtest_quarters": str(len(backtest["quarters"])),
        "value_at_risk": money(risk["total_value_at_risk"]), "flagged_deals": str(risk["flagged_deals"]),
        "top_risks": [{"account": d["account"], "amount": money(d["amount"]), "p_close": pct(d["p_close"]),
                       "category": d["category"], "flags": d["flags"]} for d in top],
        "sandbaggers": [r["name"] for r in flagged if r["flag"] == "sandbagging"],
        "over_optimistic": [r["name"] for r in flagged if r["flag"] == "over-optimistic"],
    }


def _numbers_in(obj) -> list[str]:
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _numbers_in(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _numbers_in(v)]
    return [str(obj)]


def _parse(text: str) -> list[dict]:
    out = []
    for m in NUM.finditer(text):
        whole, frac, unit = m.group(2).replace(",", ""), m.group(3) or "", (m.group(4) or "").strip()
        value = float(f"{whole}.{frac}" if frac else whole)
        out.append({"token": m.group(0).strip(), "value": value, "decimals": len(frac),
                    "scale": UNITS.get(unit, 1.0), "pct": unit == "%"})
    return out


def fact_numbers(facts: dict) -> list[dict]:
    """Every number that appears in a fact, in raw units (dollars, percent points, counts)."""
    nums = []
    for s in _numbers_in(facts):
        for t in _parse(s):
            nums.append({"value": t["value"] * t["scale"], "pct": t["pct"]})
    return nums


def validate(text: str, facts: dict) -> dict:
    allowed = fact_numbers(facts)
    bad = []
    for t in _parse(text):
        ok = False
        for f in allowed:
            if f["pct"] != t["pct"]:
                continue
            if round(f["value"] / t["scale"], t["decimals"]) == round(t["value"], t["decimals"]):
                ok = True
                break
        if not ok:
            bad.append(t["token"])
    return {"valid": not bad, "violations": bad, "numbers_checked": len(_parse(text))}


SYSTEM = (
    "You write the weekly forecast call for a CRO of a B2B SaaS company, for a board update. Use only the facts "
    "given, and copy every number exactly as it is written in the facts; do not compute new numbers, ranges, or "
    "differences. Four short paragraphs, plain text, no headings, no bullet points, under 180 words: the call "
    "(what to commit and the range), how it compares with the reps' roll-up, the deals that move it most, and "
    "which reps' calls to discount and why."
)


class TemplateWriter:
    name = "template"

    def write(self, facts: dict) -> dict:
        f = facts
        p1 = (f"{f['quarter']}, week {f['week']} of {f['weeks_in_quarter']}. Closed won so far is {f['won_to_date']}. "
              f"The simulation puts the quarter at {f['p50']} most likely, with a P10 to P90 range of {f['p10']} to "
              f"{f['p90']}. Commit {f['p10']} to the board: that is the P10, beaten in all but the worst tenth of "
              "simulated quarters.")
        p2 = (f"The reps' roll-up is {f['rep_commit']} commit and {f['rep_best_case']} best case. The simulation "
              f"reaches the commit {f['p_hit_commit']} of the time and best case {f['p_hit_best_case']}. Over "
              f"{f['backtest_quarters']} backtested quarters the deal model missed by {f['model_mid_mape']} at "
              f"mid-quarter, the rep commit by {f['commit_mid_mape']}, and the P10 to P90 range held the result "
              f"{f['interval_coverage']} of the time.")
        risks = "; ".join(f"{d['account']} ({d['amount']}, {d['category']}, {d['p_close']} to close"
                          + (f", {', '.join(d['flags'])}" if d["flags"] else "") + ")" for d in f["top_risks"])
        p3 = (f"{f['flagged_deals']} in-quarter deals carry at least one risk flag, and {f['value_at_risk']} of the "
              f"reps' call is value the model does not expect. The biggest: {risks}." if risks else "")
        sb, oo = f["sandbaggers"], f["over_optimistic"]
        p4 = []
        if oo:
            p4.append(f"Discount commits from {', '.join(oo)}: their committed deals close well below the team rate.")
        if sb:
            p4.append(f"Expect upside from {', '.join(sb)}: their best case deals close well above the team rate.")
        text = "\n\n".join(x for x in (p1, p2, p3, " ".join(p4)) if x)
        return {"provider": self.name, "text": text, "usage": {"input_tokens": 0, "output_tokens": 0}}


class ClaudeWriter:
    name = "claude"

    def __init__(self, client=None, model: str | None = None):
        self.model = model or os.environ.get("PIPECAST_CLAUDE_MODEL", "claude-opus-5-5")
        if client is None:
            import anthropic  # lazy: only needed when this provider is chosen

            client = anthropic.Anthropic()
        self.client = client

    def write(self, facts: dict) -> dict:
        resp = self.client.messages.create(
            model=self.model, max_tokens=600, system=SYSTEM,
            messages=[{"role": "user", "content": "Facts:\n" + json.dumps(facts, indent=1)}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        return {"provider": self.name, "model": self.model, "text": text,
                "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}}


def get_writer(name: str | None = None, client=None):
    name = (name or os.environ.get("PIPECAST_LLM_PROVIDER", "template")).lower()
    if name == "claude":
        return ClaudeWriter(client=client)
    return TemplateWriter()


def write_commentary(facts: dict, writer) -> dict:
    out = writer.write(facts)
    check = validate(out["text"], facts)
    result = {**out, **check, "fallback": False}
    if not check["valid"] and writer.name != "template":
        fb = TemplateWriter().write(facts)
        result = {**fb, **validate(fb["text"], facts), "fallback": True, "rejected_provider": writer.name,
                  "rejected_violations": check["violations"]}
    return result
