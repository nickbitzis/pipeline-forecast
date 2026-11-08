"""Seeded synthetic CRM for Corvane, a made-up B2B SaaS company.

40 account executives in 5 regions sell to three segments. Every opportunity has a hidden true propensity that
decides whether it is won, and a hidden resolution day. The simulator walks each deal week by week and writes what a
CRM would log: stage changes, close date pushes, amount changes, the forecast category the rep sets, and sales
activities. Reps have hidden tendencies: optimistic reps commit too early, sandbaggers keep deals they will win in
best case, accurate reps call it straight.

Nothing in the forecasting code reads the hidden columns (true_p, won, resolve_day, tendency). They exist so tests
and the README can check the methods against the truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .fiscal import QUARTER_DAYS, q_end, quarter_of

REGIONS = ("NA East", "NA West", "EMEA", "APAC", "LATAM")
SEGMENTS = ("SMB", "Mid-Market", "Enterprise")
SOURCES = ("inbound", "outbound", "partner", "expansion")
TENDENCIES = ("accurate", "optimistic", "sandbagger")
STAGES = {1: "Discovery", 2: "Qualification", 3: "Solution", 4: "Proposal", 5: "Negotiation", 6: "Closed Won",
          7: "Closed Lost"}
WON, LOST = 6, 7
# Forecast categories as the rep sets them. Closed and omitted are set by the system when the deal resolves.
PIPELINE, BEST_CASE, COMMIT, CLOSED, OMITTED = 0, 1, 2, 3, -1
CATEGORIES = {PIPELINE: "pipeline", BEST_CASE: "best case", COMMIT: "commit", CLOSED: "closed", OMITTED: "omitted"}

SEG = {
    # median amount, median cycle days, opportunities per rep per week, base win logit, weekly activities
    "SMB": dict(amount=18_000, cycle=40, rate=1.3, logit=-1.15, acts=2.0),
    "Mid-Market": dict(amount=55_000, cycle=75, rate=0.7, logit=-1.15, acts=3.0),
    "Enterprise": dict(amount=160_000, cycle=130, rate=0.3, logit=-1.45, acts=4.0),
}
SOURCE_MIX = {"SMB": (0.5, 0.3, 0.1, 0.1), "Mid-Market": (0.35, 0.35, 0.15, 0.15),
              "Enterprise": (0.2, 0.4, 0.2, 0.2)}
SOURCE_LOGIT = {"inbound": 0.3, "outbound": -0.4, "partner": 0.2, "expansion": 0.9}
# (bias on the commit decision, bias on the best case decision). Sandbaggers hold back commits only.
TENDENCY_BIAS = {"accurate": (0.0, 0.0), "optimistic": (1.3, 1.0), "sandbagger": (-1.8, 0.3)}

FIRST = ("Avery", "Jordan", "Riley", "Morgan", "Casey", "Taylor", "Quinn", "Reese", "Rowan", "Sasha", "Devon",
         "Emery", "Harper", "Jules", "Kai", "Logan", "Marlo", "Noel", "Parker", "Remy", "Sage", "Tatum", "Blair",
         "Cameron", "Dana", "Elliot", "Finley", "Gray", "Hollis", "Indy", "Jessie", "Kendall", "Lane", "Milan",
         "Nico", "Oakley", "Peyton", "Robin", "Shay", "Toby")
ACCT_A = ("Ardent", "Bluefen", "Cobalt Ridge", "Driftwood", "Emberline", "Foxglove", "Granite Bay", "Harborview",
          "Ironleaf", "Juniper", "Kestrel", "Lumen", "Marlowe", "Northgate", "Oakhurst", "Pinecrest", "Quarry Hill",
          "Redwater", "Silverline", "Tidewell", "Upland", "Valemont", "Westbrook", "Yarrow")
ACCT_B = ("Logistics", "Health", "Foods", "Freight", "Clinics", "Energy", "Manufacturing", "Retail", "Labs",
          "Insurance", "Builders", "Hospitality", "Utilities", "Dental", "Robotics", "Packaging")


@dataclass
class CompanyConfig:
    n_reps: int = 40
    quarters: int = 9  # quarters 0..7 are history, quarter 8 is the current one
    warmup_days: int = 182  # deals are created this long before quarter 0 so it starts with a full pipeline
    rate_scale: float = 1.0
    growth: float = 0.035  # pipeline creation grows this much per quarter
    quarter_shock: float = 0.12  # sd of a shared win logit shock per quarter (soft and strong quarters)
    tendency_mix: tuple[float, float, float] = (0.55, 0.25, 0.20)
    seed: int = 7


@dataclass
class CRM:
    reps: pd.DataFrame
    deals: pd.DataFrame
    stage_ev: pd.DataFrame  # deal_id, day, stage
    amount_ev: pd.DataFrame  # deal_id, day, amount, change (-1 down, 0 initial, 1 up)
    close_ev: pd.DataFrame  # deal_id, day, close_day, push (1 when the close date moved later)
    cat_ev: pd.DataFrame  # deal_id, day, category
    acts: pd.DataFrame  # deal_id, day
    config: CompanyConfig = field(default_factory=CompanyConfig)

    HIDDEN = ("true_p", "won", "resolve_day")

    def deal_history(self, deal_id: str) -> list[dict]:
        """Field history in the shape a CRM audit log shows it."""
        out = []
        for name, df, col in (("stage", self.stage_ev, "stage"), ("amount", self.amount_ev, "amount"),
                              ("close_day", self.close_ev, "close_day"), ("category", self.cat_ev, "category")):
            for r in df[df.deal_id == deal_id].itertuples(index=False):
                out.append({"day": int(r.day), "field": name, "value": float(getattr(r, col))})
        return sorted(out, key=lambda e: e["day"])


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_reps(cfg: CompanyConfig, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    per_region = max(cfg.n_reps // len(REGIONS), 1)
    seg_cycle = ["SMB", "SMB", "Mid-Market", "Mid-Market", "SMB", "Mid-Market", "Enterprise", "Enterprise"]
    tendencies = rng.choice(TENDENCIES, size=cfg.n_reps, p=cfg.tendency_mix)
    for i in range(cfg.n_reps):
        region = REGIONS[(i // per_region) % len(REGIONS)]
        rows.append({
            "rep_id": f"R{i + 1:02d}",
            "name": f"{FIRST[i % len(FIRST)]} {chr(65 + (i * 7) % 26)}.",
            "region": region,
            "segment": seg_cycle[i % per_region % len(seg_cycle)],
            "tendency": str(tendencies[i]),
            "skill": float(rng.normal(0, 0.25)),
        })
    return pd.DataFrame(rows)


def generate(cfg: CompanyConfig | None = None) -> CRM:
    cfg = cfg or CompanyConfig()
    rng = np.random.default_rng(cfg.seed)
    reps = make_reps(cfg, rng)
    shocks = rng.normal(0, cfg.quarter_shock, size=cfg.quarters + 20)

    deals, stage_ev, amount_ev, close_ev, cat_ev, acts = [], [], [], [], [], []
    horizon = cfg.quarters * QUARTER_DAYS
    n = 0
    for rep in reps.itertuples(index=False):
        seg = SEG[rep.segment]
        for week_start in range(-cfg.warmup_days, horizon, 7):
            q = quarter_of(max(week_start, 0))
            lam = seg["rate"] * cfg.rate_scale * (1 + cfg.growth) ** q
            for _ in range(rng.poisson(lam)):
                n += 1
                created = week_start + int(rng.integers(0, 7))
                d = _simulate_deal(f"D{n:05d}", created, rep, seg, shocks, rng)
                deals.append(d["deal"])
                stage_ev += d["stage"]
                amount_ev += d["amount"]
                close_ev += d["close"]
                cat_ev += d["cat"]
                acts += d["acts"]

    deals_df = pd.DataFrame(deals).sort_values(["created_day", "deal_id"]).reset_index(drop=True)
    return CRM(
        reps=reps,
        deals=deals_df,
        stage_ev=pd.DataFrame(stage_ev, columns=["deal_id", "day", "stage"]),
        amount_ev=pd.DataFrame(amount_ev, columns=["deal_id", "day", "amount", "change"]),
        close_ev=pd.DataFrame(close_ev, columns=["deal_id", "day", "close_day", "push"]),
        cat_ev=pd.DataFrame(cat_ev, columns=["deal_id", "day", "category"]),
        acts=pd.DataFrame(acts, columns=["deal_id", "day"]).sort_values(["day", "deal_id"]).reset_index(drop=True),
        config=cfg,
    )


def _round_amount(x: float) -> float:
    return float(max(round(x / 500) * 500, 2_000))


def _simulate_deal(deal_id, created, rep, seg, shocks, rng) -> dict:
    segment = rep.segment
    source = SOURCES[int(rng.choice(4, p=SOURCE_MIX[segment]))]
    amount = seg["amount"] * float(np.exp(rng.normal(0, 0.5))) * (0.6 if source == "expansion" else 1.0)
    amount = _round_amount(amount)
    shock = shocks[max(quarter_of(created), 0) + 1]
    p = float(_sigmoid(seg["logit"] + SOURCE_LOGIT[source] + rep.skill + shock + rng.normal(0, 0.8)))
    won = bool(rng.random() < p)

    cycle = seg["cycle"] * float(np.exp(rng.normal(0, 0.35)))
    if not won:
        cycle *= float(rng.uniform(0.7, 1.5))
    resolve = created + max(int(cycle), 10)
    if won:
        # Quarter end hockey stick: buyers sign at the deadline, reps pull deals in with a discount.
        qe = q_end(quarter_of(resolve))
        pos = resolve % QUARTER_DAYS
        if pos >= 63 and rng.random() < 0.6:
            resolve = max(qe - int(rng.integers(0, 7)), created + 10)
        elif pos <= 9 and rng.random() < 0.35 and qe - QUARTER_DAYS - created > 14:
            resolve = qe - QUARTER_DAYS - int(rng.integers(0, 5))
    span = resolve - created

    # Stage schedule: won deals walk through 1 to 5; lost deals stall at some stage and sit there.
    if won:
        fr = np.sort(np.clip(np.array([0.12, 0.3, 0.55, 0.78]) + rng.normal(0, 0.05, 4), 0.02, 0.95))
        top = 5
    else:
        top = int(rng.choice([1, 2, 3, 4, 5], p=[0.22, 0.26, 0.24, 0.17, 0.11]))
        stall = float(rng.uniform(0.25, 0.65))
        fr = np.sort(rng.uniform(0.03, stall, size=4))[: top - 1]
    stage_days = [created + max(int(f * span), 1) for f in fr]

    # Rep's close date: reps underestimate cycles; optimists more so. Many park it at quarter end.
    optimism = (0.4, 0.75) if rep.tendency == "optimistic" else (0.55, 0.95)
    close = created + int(seg["cycle"] * rng.uniform(*optimism))
    if rng.random() < 0.35:
        close = q_end(quarter_of(close))
    close = max(close, created + 7)

    bias_commit, bias_best = TENDENCY_BIAS[rep.tendency]
    min_commit_stage = 2 if rep.tendency == "optimistic" else (4 if rep.tendency == "sandbagger" else 3)
    dark = int(rng.integers(21, 36))  # lost deals go quiet this many days before they are marked lost
    rate = seg["acts"]
    deal_noise = float(rng.normal(0, 0.5))

    ev_stage = [(deal_id, created, 1)]
    ev_amount = [(deal_id, created, amount, 0)]
    ev_close = [(deal_id, created, close, 0)]
    cat = PIPELINE
    ev_cat = [(deal_id, created, cat)]
    ev_acts = []
    stage = 1
    t = created
    while t < resolve:
        # stage changes that happened during the last week
        while stage < len(stage_days) + 1 and stage_days[stage - 1] <= t:
            stage += 1
            ev_stage.append((deal_id, stage_days[stage - 2], stage))
        # activities this week
        r = rate * (0.1 if (not won and resolve - t < dark) else (1.5 if won and resolve - t < 14 else 1.0))
        for _ in range(rng.poisson(r)):
            day = t + int(rng.integers(0, 7))
            if day < resolve:
                ev_acts.append((deal_id, day))
        if t > created:
            # close date pushes: it is passing and the deal is still open
            if close <= t + 3 or (not won and quarter_of(close) == quarter_of(t) and rng.random() < 0.05):
                if rng.random() < 0.35:
                    new = q_end(quarter_of(close) + 1)
                else:
                    new = max(close + int(rng.integers(14, 43)), t + 7)
                close = int(new)
                ev_close.append((deal_id, t, close, 1))
            # amount changes
            u = rng.random()
            late = resolve - t < 42 and stage >= 4
            if not won and late and u < 0.15:
                amount = _round_amount(amount * rng.uniform(1.15, 1.4))  # happy ears: scope grows, deal dies
                ev_amount.append((deal_id, t, amount, 1))
            elif won and late and u < 0.04:
                amount = _round_amount(amount * rng.uniform(1.1, 1.3))
                ev_amount.append((deal_id, t, amount, 1))
            elif u > 0.97:
                new = _round_amount(amount * rng.uniform(0.8, 1.2))
                if new != amount:
                    ev_amount.append((deal_id, t, new, 1 if new > amount else -1))
                    amount = new
        # forecast category, updated before the Monday call
        # Reps update calls before the Monday call, so in the last week of a quarter they already call next quarter.
        if quarter_of(close) not in (quarter_of(t), quarter_of(t + 7)):
            new_cat = PIPELINE
        else:
            progress = (t - created) / max(span, 1)
            belief = (0.6 * np.log(p / (1 - p)) + (1 if won else -1) * 3.0 * progress**2 + 0.4 * (stage - 3)
                      + deal_noise + rng.normal(0, 0.3))
            if belief + bias_commit > 1.0 and stage >= min_commit_stage:
                new_cat = COMMIT
            elif belief + bias_best > -0.4:
                new_cat = BEST_CASE
            else:
                new_cat = PIPELINE
        if new_cat != cat:
            cat = new_cat
            ev_cat.append((deal_id, t, cat))
        t += 7

    while stage < len(stage_days) + 1:  # stage changes between the last weekly update and resolution
        stage += 1
        ev_stage.append((deal_id, stage_days[stage - 2], stage))

    # resolution
    if won and rng.random() < 0.5:
        new = _round_amount(amount * rng.uniform(0.85, 1.0))  # discount to get it signed
        if new != amount:
            ev_amount.append((deal_id, resolve, new, -1))
            amount = new
    ev_close.append((deal_id, resolve, resolve, 0))
    ev_stage.append((deal_id, resolve, WON if won else LOST))
    ev_cat.append((deal_id, resolve, CLOSED if won else OMITTED))
    acct = f"{ACCT_A[int(rng.integers(len(ACCT_A)))]} {ACCT_B[int(rng.integers(len(ACCT_B)))]}"
    deal = {
        "deal_id": deal_id, "account": acct, "rep_id": rep.rep_id, "region": rep.region, "segment": segment,
        "source": source, "created_day": created, "true_p": round(p, 4), "won": won, "resolve_day": resolve,
        "final_amount": amount,
    }
    return {"deal": deal, "stage": ev_stage, "amount": ev_amount, "close": ev_close, "cat": ev_cat,
            "acts": ev_acts}
