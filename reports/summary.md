# Backtest summary

Quarters replayed: FY25 Q2, FY25 Q3, FY25 Q4, FY26 Q1, FY26 Q2, FY26 Q3. Error is MAPE against closed won bookings, by week of quarter.

| Method | Wk 1 | Wk 3 | Wk 5 | Wk 7 | Wk 9 | Wk 11 | Wk 13 | Mid-quarter MAPE | Mid-quarter bias |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rep roll-up: commit | 62.8% | 49.9% | 40.4% | 24.3% | 14.1% | 9.8% | 7.3% | 26.3% | -26.3% |
| Rep roll-up: best case | 12.5% | 17.6% | 19.8% | 30.6% | 22.3% | 17.8% | 7.5% | 25.1% | +25.1% |
| Adjusted roll-up | 22.3% | 12.8% | 14.1% | 5.5% | 4.9% | 9.9% | 10.3% | 7.5% | -4.8% |
| Stage-weighted pipeline | 37.2% | 26.5% | 19.3% | 10.6% | 6.6% | 6.4% | 8.0% | 12.0% | -12.0% |
| Deal model | 5.6% | 5.8% | 4.9% | 6.2% | 5.3% | 3.7% | 3.0% | 5.0% | +3.7% |
| Monte Carlo P50 | 5.7% | 5.7% | 4.8% | 6.1% | 5.2% | 3.5% | 3.2% | 4.9% | +3.4% |
| Time-series baseline | 13.4% | 14.7% | 13.6% | 10.0% | 6.8% | 8.4% | 6.4% | 10.3% | +5.8% |

Monte Carlo P10 to P90 coverage: 84.6% overall, 86.7% mid-quarter.
Deal model calibration on 27,964 out-of-time deal weeks: Brier 0.0698 raw, 0.0695 calibrated; ECE 0.0102 raw, 0.0070 calibrated.

## Gate

- PASS Monte Carlo P10 to P90 coverage 84.6% (band 70% to 90%)
- PASS mid-quarter MAPE: deal model 5.0% vs rep roll-up 26.3%

## Current quarter: FY26 Q4, week 7

Won to date $3.01M. Rep commit $4.75M, best case $9.40M. Monte Carlo P10 $6.53M, P50 $7.03M, P90 $7.59M.

| Method | Forecast | Backtest MAPE this week |
|---|---:|---:|
| Rep roll-up: commit | $4.75M | 24.3% |
| Rep roll-up: best case | $9.40M | 30.6% |
| Adjusted roll-up | $6.38M | 5.5% |
| Stage-weighted pipeline | $5.95M | 10.6% |
| Deal model | $7.05M | 6.2% |
| Monte Carlo P50 | $7.03M | 6.1% |
| Time-series baseline | $6.71M | 10.0% |

## Rep flags

| Rep | Segment | Flag | Evidence |
|---|---|---|---|
| Avery A. | SMB | over-optimistic | 60 committed deals closed at 67%, SMB team 81%. Mid-quarter commit under called by 31% over 8 quarters; beat commit in 8 of 8. |
| Jordan H. | SMB | sandbagging | 68 deals kept at best case closed at 59%, SMB team 23%. Mid-quarter commit under called by 48% over 8 quarters; beat commit in 7 of 8. |
| Riley O. | Mid-Market | over-optimistic | 40 committed deals closed at 60%, Mid-Market team 77%. Mid-quarter commit under called by 11% over 8 quarters; beat commit in 4 of 8. |
| Taylor J. | Mid-Market | sandbagging | 36 deals kept at best case closed at 28%, Mid-Market team 13%. Mid-quarter commit under called by 47% over 8 quarters; beat commit in 8 of 8. |
| Rowan E. | SMB | over-optimistic | 56 committed deals closed at 62%, SMB team 81%. Mid-quarter commit under called by 25% over 8 quarters; beat commit in 6 of 8. |
| Marlo I. | SMB | sandbagging | 73 deals kept at best case closed at 37%, SMB team 23%. Mid-quarter commit under called by 52% over 8 quarters; beat commit in 7 of 8. |
| Noel P. | SMB | over-optimistic | 43 committed deals closed at 67%, SMB team 81%. Mid-quarter commit under called by 39% over 8 quarters; beat commit in 7 of 8. |
| Parker W. | Mid-Market | over-optimistic | 45 committed deals closed at 51%, Mid-Market team 77%. Mid-quarter commit under called by 3% over 8 quarters; beat commit in 4 of 8. |
| Remy D. | Mid-Market | sandbagging | 44 deals kept at best case closed at 25%, Mid-Market team 13%. Mid-quarter commit under called by 40% over 8 quarters; beat commit in 5 of 8. |
| Blair Y. | Enterprise | over-optimistic | 13 committed deals closed at 31%, Enterprise team 66%. Mid-quarter commit over called by 100% over 7 quarters; beat commit in 1 of 7. |
| Elliot T. | SMB | sandbagging | 31 deals kept at best case closed at 39%, SMB team 23%. Mid-quarter commit under called by 60% over 7 quarters; beat commit in 7 of 7. |
| Gray H. | Mid-Market | sandbagging | 41 deals kept at best case closed at 29%, Mid-Market team 13%. Mid-quarter commit under called by 48% over 8 quarters; beat commit in 6 of 8. |
| Hollis O. | SMB | over-optimistic | 61 committed deals closed at 67%, SMB team 81%. Mid-quarter commit under called by 24% over 8 quarters; beat commit in 6 of 8. |
| Nico E. | Mid-Market | over-optimistic | 30 committed deals closed at 57%, Mid-Market team 77%. Mid-quarter commit under called by 5% over 8 quarters; beat commit in 3 of 8. |
