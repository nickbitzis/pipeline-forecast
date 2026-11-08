"""Fiscal calendar: 13 week quarters, integer days.

Day 0 is the first day of quarter 0. Quarter q covers days [91q, 91q + 91). A forecast "as of week w" is made on the
Monday that starts week w, using everything logged before that day.
"""

from __future__ import annotations

from datetime import date, timedelta

QUARTER_DAYS = 91
WEEKS = 13
# Quarter 0 starts on a Monday; quarter 8 (the current one) starts on Monday 28 Sep 2026.
DAY0 = date(2026, 9, 28) - timedelta(days=8 * QUARTER_DAYS)


def q_start(q: int) -> int:
    return q * QUARTER_DAYS


def q_end(q: int) -> int:
    """Last day of quarter q (inclusive)."""
    return q * QUARTER_DAYS + QUARTER_DAYS - 1


def quarter_of(day: int) -> int:
    return day // QUARTER_DAYS


def as_of_day(q: int, week: int) -> int:
    """Morning of the Monday that starts week `week` (1 to 13) of quarter q."""
    if not 1 <= week <= WEEKS:
        raise ValueError("week must be 1 to 13")
    return q_start(q) + 7 * (week - 1)


def label(q: int) -> str:
    n = 24 * 4 + 3 + q  # quarter 0 is FY24 Q4
    return f"FY{n // 4} Q{n % 4 + 1}"


def to_date(day: int) -> date:
    return DAY0 + timedelta(days=int(day))


def iso(day: int) -> str:
    return to_date(day).isoformat()
