Outside week 1, the sim's per-player tails are near calibrated. Over weeks 2–8, 6.2% (0.5) of real DK scores beat the run's own p95 (par 5%) and 11.0% (0.9) fell below its p10 (par 10%). The remaining misses: RB and TE bust more than modeled (below p10 14.9% (2.0) and 16.4% (2.7)); QB floors are set too low (6.0% (1.7)). Kicker and DST upside tails are not too thin: prime-time Showdown K beat p95 2.0% (2.0) (n 50) and DST 0.0% (0.0) (n 50), so the books' K/DST shortfall is not a thin-tail problem in the marginals. No per-position spread multiplier fitted on weeks 1–6 improves weeks 7–8.

# Vale unit 1103CL-B (11/3/2026): tail calibration on the public p10/p95, weeks 1–8

Reduced version, as agreed: each run's own published p10 (`floor`), p95 (`ceiling`) and mean (`avgPts`), not a full percentile (PIT) analysis. Public data only, aggregates only. The pre-rewrite git objects were not used. No site code was changed and nothing was released.

- **Data repo:** `chrisspags/splashplaynba` `origin/main` @ `65a59ddc52`.
- **Inputs:**
  - all 145 `nfl-dk-{main,early-only,afternoon-only,showdown}-<dg>.json` (`results.playerResults`);
  - real points from `nfl-dk-results/<date>.json` `contests[].playerActuals[].fpts`;
  - who played, from `nfl-recap/2026-w01..w08.json`;
  - kickoff times from `nfl-schedule.json`.
- **Week 9** is not played yet.

## Matching and populations

**Rule:**
- A player-slate row is a run's player with projection > 0 who took part: snaps > 0, or a box row for K; DST always.
- Real points come from a contest of the **same draft group**.
- When that draft group has no ingested contest, a Classic player falls back to the same date's Classic contest points, which are identical DK scoring for the same game.
- Kickers need their own Showdown draft group.

**Counts:**
- 141 of 145 runs have rows: 7364 player-slate rows, 2956 player-games.
- **Classic:** 24 runs (16 with same-draft-group results).
- **Showdown prime-time:** 25 runs (Thu, Sun night, Mon, and the opener; kickoff 7 pm ET or later, or not a Sunday). All 25 have same-draft-group results, so K and DST are scored there.
- **Showdown Sunday day:** 92 runs, 0 with same-draft-group results. Their non-K players use same-date Classic points; their kickers are not scored.
- **Not scored:** 4 Showdown runs (2026-10-04, 2026-10-11, 2026-10-18, 2026-10-25) are morning international games. They are on no Classic slate and have no ingested results.

**Showdown points are FLEX points:**
- For 554 same-draft-group Showdown rows, DK `fpts` equal the box-score DK points exactly in 526, and equal 1.5× the box in 0.
- Across all rows, DK points matched the box in 6289 of 6712. The rest differ by fumbles or 2-point conversions.

**Tests:**
- Above p95 is strict (real > p95).
- Below p10 is strict and is **tested only where p10 > 0, or for DST**. A run with p10 = 0 puts a point mass at zero, so a strict "below" can't happen; those rows are counted in the "p10 = 0" column.

**Standard errors:**
- Player-game Poisson bootstrap, 1,000 replicates: each team-week-player is resampled as one unit, with all its slate rows.
- This doesn't capture week-to-week swings, which are large (see "By week").
- † marks a share more than 2 se from par.

## Weeks 1–8: by population, position and salary band

| Cell | Rows | Player-games | Above own p95 (par 5%) | Rows tested for p10 | Below own p10 (par 10%) | p10 = 0 (not tested) | Mean bias, actual − mean (se) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **All runs** | 7364 | 2956 | 6.6% (0.5) † | 4372 | 13.8% (0.8) † | 40.6% | 0.02 (0.12) |
| **Classic** | 4410 | 2223 | 6.6% (0.5) † | 2614 | 13.8% (0.9) † | 40.7% | -0.03 (0.13) |
| **Showdown prime-time** | 654 | 654 | 4.6% (0.8) | 421 | 13.8% (1.7) † | 35.6% | 0.21 (0.22) |
| **Showdown Sunday day** | 2300 | 2300 | 7.1% (0.5) † | 1337 | 13.7% (0.9) † | 41.9% | 0.05 (0.13) |
| QB | 600 | 232 | 5.3% (1.6) | 600 | 6.0% (1.5) † | 0.0% | 0.29 (0.55) |
| RB | 1672 | 665 | 7.1% (1.0) | 1102 | 16.1% (1.8) † | 34.1% | -0.33 (0.26) |
| WR | 2698 | 1064 | 6.5% (0.8) | 1348 | 14.8% (1.5) † | 50.0% | 0.21 (0.20) |
| TE | 1742 | 711 | 7.4% (1.0) † | 670 | 21.5% (2.6) † | 61.5% | 0.10 (0.20) |
| K | 50 | 50 | 2.0% (2.0) | 50 | 12.0% (4.8) | 0.0% | -0.89 (0.68) |
| DST | 602 | 234 | 5.3% (1.6) | 602 | 6.6% (1.8) | 0.0% | -0.29 (0.38) |
| under $4,000 | 3605 | 1695 | 7.2% (0.7) † | 1135 | 20.9% (1.9) † | 68.5% | 0.04 (0.12) |
| $4,000-$6,999 | 2686 | 1423 | 6.3% (0.7) | 2166 | 12.2% (1.0) † | 19.4% | 0.01 (0.20) |
| $7,000+ | 1073 | 855 | 5.4% (0.9) | 1071 | 9.4% (1.1) | 0.2% | -0.01 (0.33) |

### By position and salary band, within each population

| Cell | Rows | Player-games | Above own p95 (par 5%) | Rows tested for p10 | Below own p10 (par 10%) | p10 = 0 (not tested) | Mean bias, actual − mean (se) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Classic: QB | 368 | 184 | 5.4% (1.7) | 368 | 6.3% (1.7) † | 0.0% | 0.21 (0.58) |
| Classic: RB | 1008 | 506 | 7.0% (1.1) | 673 | 16.6% (2.0) † | 33.2% | -0.43 (0.28) |
| Classic: WR | 1635 | 825 | 6.3% (0.8) | 815 | 14.6% (1.7) † | 50.2% | 0.14 (0.22) |
| Classic: TE | 1031 | 524 | 7.5% (1.2) † | 390 | 21.3% (2.9) † | 62.2% | 0.09 (0.22) |
| Classic: DST | 368 | 184 | 6.0% (1.8) | 368 | 6.5% (1.9) | 0.0% | -0.22 (0.41) |
| Showdown prime-time: QB | 48 | 48 | 4.2% (2.9) | 48 | 8.3% (4.3) | 0.0% | 1.25 (1.21) |
| Showdown prime-time: RB | 135 | 135 | 7.4% (2.3) | 92 | 12.0% (3.3) | 31.9% | 0.59 (0.51) |
| Showdown prime-time: WR | 216 | 216 | 4.6% (1.4) | 118 | 17.8% (3.5) † | 45.4% | 0.50 (0.44) |
| Showdown prime-time: TE | 155 | 155 | 4.5% (1.7) | 63 | 19.0% (5.0) | 59.4% | -0.07 (0.37) |
| Showdown prime-time: K | 50 | 50 | 2.0% (2.0) | 50 | 12.0% (4.8) | 0.0% | -0.89 (0.68) |
| Showdown prime-time: DST | 50 | 50 | 0.0% (0.0) | 50 | 8.0% (3.8) | 0.0% | -1.02 (0.65) |
| Showdown Sunday day: QB | 184 | 184 | 5.4% (1.7) | 184 | 4.9% (1.6) † | 0.0% | 0.20 (0.58) |
| Showdown Sunday day: RB | 529 | 529 | 7.0% (1.1) | 337 | 16.0% (2.0) † | 36.3% | -0.36 (0.27) |
| Showdown Sunday day: WR | 847 | 847 | 7.3% (0.9) † | 415 | 14.2% (1.7) † | 51.0% | 0.26 (0.21) |
| Showdown Sunday day: TE | 556 | 556 | 8.1% (1.1) † | 217 | 22.6% (2.8) † | 61.0% | 0.17 (0.21) |
| Showdown Sunday day: DST | 184 | 184 | 5.4% (1.7) | 184 | 6.5% (1.9) | 0.0% | -0.22 (0.41) |
| Classic: under $4,000 | 2132 | 1082 | 6.8% (0.8) † | 753 | 17.8% (1.9) † | 64.7% | -0.06 (0.14) |
| Classic: $4,000-$6,999 | 2060 | 1032 | 6.6% (0.7) † | 1643 | 12.6% (1.1) † | 20.2% | 0.07 (0.22) |
| Classic: $7,000+ | 218 | 109 | 5.5% (2.2) | 218 | 9.2% (2.6) | 0.0% | -0.58 (0.92) |
| Showdown prime-time: under $4,000 | 307 | 307 | 5.5% (1.3) | 93 | 31.2% (4.8) † | 69.7% | -0.05 (0.22) |
| Showdown prime-time: $4,000-$6,999 | 170 | 170 | 1.8% (1.0) † | 151 | 11.9% (2.6) | 11.2% | -0.62 (0.41) |
| Showdown prime-time: $7,000+ | 177 | 177 | 5.6% (1.7) | 177 | 6.2% (1.9) † | 0.0% | 1.48 (0.61) |
| Showdown Sunday day: under $4,000 | 1166 | 1166 | 8.3% (0.8) † | 289 | 25.6% (2.6) † | 75.2% | 0.23 (0.12) |
| Showdown Sunday day: $4,000-$6,999 | 456 | 456 | 6.8% (1.2) | 372 | 10.5% (1.6) | 18.4% | 0.01 (0.31) |
| Showdown Sunday day: $7,000+ | 678 | 678 | 5.3% (0.9) | 676 | 10.4% (1.2) | 0.3% | -0.22 (0.30) |

## K and DST on Showdown (the main question)

**Kickers** (prime-time Showdown only; kickers need their own draft group's results), 50 player-games:
- 2.0% (2.0) beat their p95 (par 5%).
- 12.0% (4.8) fell below their p10 (par 10%).
- Mean bias -0.89 (se 0.68).

**DST:**
- Prime-time Showdown, 50 player-games: 0.0% (0.0) above p95, 8.0% (3.8) below p10, mean bias -1.02 (se 0.65).
- Classic: 6.0% (1.8) above p95 and 6.5% (1.9) below p10.

**What this means:** the sim's marginals do not give K or DST too little upside; if anything, prime-time DST upside is a little generous. So the top-1% lineups' heavy K/DST use (11/1 SNF: K in 96.5% of top-1% lineups vs 29.6% of ours) is not explained by a thin K/DST tail. The place to look is how K/DST outcomes co-move with the rest of a lineup (game script, the kicker/DST loads in the copula) and how the ranking scores them. One-player tail checks can't see that.

**Sample size:** the K and DST cells are small (50 and 50). One more kicker beating p95 would move the share by 2.0 points.

## By week (all rows; RB/WR/TE in the last two columns)

| Week | Rows | Above p95 | Below p10 (tested rows) | RB/WR/TE above p95 | RB/WR/TE below p10 (tested rows) |
|---:|---:|---:|---:|---:|---:|
| 1 | 945 | 9.4% (1.6) | 24.3% (2.3) (920) | 10.3% (1.8) | 28.5% (2.7) (752) |
| 2 | 1026 | 5.0% (1.1) | 10.5% (1.9) (649) | 5.0% (1.3) | 12.0% (2.3) (475) |
| 3 | 1031 | 8.3% (1.4) | 8.7% (2.0) (553) | 9.3% (1.6) | 9.0% (2.4) (379) |
| 4 | 968 | 5.6% (1.2) | 13.1% (2.5) (526) | 6.6% (1.4) | 17.9% (3.4) (364) |
| 5 | 894 | 7.4% (1.5) | 14.0% (2.6) (477) | 6.5% (1.5) | 16.8% (3.5) (327) |
| 6 | 818 | 5.6% (1.3) | 9.6% (2.4) (425) | 6.3% (1.5) | 13.6% (3.4) (287) |
| 7 | 797 | 5.4% (1.3) | 7.3% (2.0) (384) | 6.1% (1.5) | 6.5% (2.3) (246) |
| 8 | 885 | 5.9% (1.2) | 13.0% (2.6) (438) | 4.7% (1.2) | 13.8% (3.3) (290) |

Week 1 carries most of the miss: it ran on prior-season data only. From week 2 on, the below-p10 share is close to par and the above-p95 share is 5–9%.

## Weeks 2–8 (week 1 excluded)

| Cell | Rows | Player-games | Above own p95 (par 5%) | Rows tested for p10 | Below own p10 (par 10%) | p10 = 0 (not tested) | Mean bias, actual − mean (se) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **All runs** | 6419 | 2561 | 6.2% (0.5) † | 3452 | 11.0% (0.9) | 46.2% | 0.00 (0.13) |
| **Classic** | 3859 | 1943 | 6.2% (0.5) † | 2072 | 11.1% (1.0) | 46.3% | -0.08 (0.14) |
| **Showdown prime-time** | 546 | 546 | 4.2% (0.9) | 316 | 9.8% (1.7) | 42.1% | 0.51 (0.24) |
| **Showdown Sunday day** | 2014 | 2014 | 6.7% (0.5) † | 1064 | 11.0% (0.9) | 47.2% | 0.01 (0.13) |
| QB | 520 | 200 | 4.4% (1.6) | 520 | 6.0% (1.7) † | 0.0% | -0.06 (0.56) |
| RB | 1464 | 578 | 7.0% (1.1) | 897 | 14.9% (2.0) † | 38.7% | -0.36 (0.28) |
| WR | 2345 | 918 | 5.5% (0.8) | 1013 | 9.6% (1.5) | 56.8% | 0.20 (0.21) |
| TE | 1526 | 621 | 7.1% (1.1) † | 458 | 16.4% (2.7) † | 70.0% | 0.12 (0.21) |
| K | 42 | 42 | 2.4% (2.4) | 42 | 9.5% (4.7) | 0.0% | -0.35 (0.74) |
| DST | 522 | 202 | 6.1% (1.8) | 522 | 7.1% (1.9) | 0.0% | -0.14 (0.42) |
| Showdown prime-time: QB | 40 | 40 | 5.0% (3.5) | 40 | 5.0% (3.5) | 0.0% | 2.00 (1.29) |
| Showdown prime-time: RB | 111 | 111 | 5.4% (2.2) | 70 | 8.6% (3.3) | 36.9% | 0.79 (0.56) |
| Showdown prime-time: WR | 179 | 179 | 5.0% (1.6) | 81 | 9.9% (3.4) | 54.7% | 0.97 (0.50) |
| Showdown prime-time: TE | 132 | 132 | 3.8% (1.7) | 41 | 17.1% (6.1) | 68.9% | -0.03 (0.38) |
| Showdown prime-time: K | 42 | 42 | 2.4% (2.4) | 42 | 9.5% (4.7) | 0.0% | -0.35 (0.74) |
| Showdown prime-time: DST | 42 | 42 | 0.0% (0.0) | 42 | 9.5% (4.5) | 0.0% | -1.10 (0.70) |

## Spread multiplier per position

**Method:**
- Scale each run's tails as `mean + m·(q − mean)` for q = p10 and p95.
- Pick one m per position on the training weeks to bring both tail shares to par: minimize (share95 − .05)²/(.05·.95) + (share10 − .10)²/(.10·.90) over a grid of m = 0.40…3.00.
- se of m from a 200-replicate player-game bootstrap of the fit.
- Held out on weeks 7–8. A share "moves toward par" if |share(m) − par| < |share(1) − par|.

| Fit | m (se) | Training rows | Held-out rows | Held-out above p95: m=1 → m | p95 | Held-out below p10: m=1 → m | p10 |
|---|---:|---:|---:|---:|---|---:|---|
| **Specified fit: weeks 1–6** | | | | | | | |
| QB (all runs) | 0.89 (0.12) | 464 | 136 | 4.4% (3.1) → 5.1% (3.1) | toward par | 11.0% (4.4) → 16.2% (5.4) | away from par |
| RB (all runs) | 1.14 (0.02) | 1306 | 366 | 11.2% (2.7) → 7.7% (2.2) | toward par | 10.7% (3.4) → 6.7% (2.8) | away from par |
| WR (all runs) | 1.20 (0.02) | 2085 | 613 | 2.8% (1.1) → 2.3% (0.9) | away from par | 10.4% (3.2) → 4.3% (2.2) | away from par |
| TE (all runs) | 1.34 (0.01) | 1325 | 417 | 4.1% (1.5) → 3.1% (1.3) | away from par | 9.9% (4.4) → 0.0% (0.0) | away from par |
| K (all runs) | 1.00 (0.19) | 38 | 12 | 0.0% (0.0) → 0.0% (0.0) | unchanged | 8.3% (8.8) → 8.3% (8.8) | unchanged |
| DST (all runs) | 0.93 (0.05) | 464 | 138 | 10.1% (4.4) → 10.9% (4.7) | away from par | 9.4% (4.3) → 13.8% (5.3) | away from par |
| QB (Showdown) | 0.91 (0.09) | 180 | 52 | 3.8% (2.7) → 3.8% (2.7) | unchanged | 7.7% (3.7) → 11.5% (4.4) | toward par |
| RB (Showdown) | 1.14 (0.02) | 517 | 147 | 8.8% (2.4) → 8.2% (2.3) | toward par | 9.5% (3.2) → 6.0% (2.6) | away from par |
| WR (Showdown) | 1.20 (0.01) | 825 | 238 | 2.9% (1.1) → 2.9% (1.1) | unchanged | 10.9% (3.2) → 4.3% (2.1) | away from par |
| TE (Showdown) | 1.35 (0.01) | 542 | 169 | 5.3% (1.7) → 3.6% (1.4) | away from par | 13.0% (5.0) → 0.0% (0.0) | away from par |
| K (Showdown) | 1.00 (0.19) | 38 | 12 | 0.0% (0.0) → 0.0% (0.0) | unchanged | 8.3% (8.8) → 8.3% (8.8) | unchanged |
| DST (Showdown) | 0.92 (0.06) | 180 | 54 | 7.4% (3.6) → 9.3% (4.0) | away from par | 9.3% (4.0) → 13.0% (4.8) | away from par |
| QB (Classic) | 0.84 (0.12) | 284 | 84 | 4.8% (3.3) → 8.3% (4.1) | away from par | 13.1% (5.1) → 19.0% (6.0) | away from par |
| RB (Classic) | 1.15 (0.02) | 789 | 219 | 12.8% (3.1) → 6.4% (2.3) | toward par | 11.4% (3.9) → 7.1% (3.1) | away from par |
| WR (Classic) | 1.19 (0.02) | 1260 | 375 | 2.7% (1.1) → 2.1% (1.0) | away from par | 10.1% (3.5) → 5.0% (2.4) | away from par |
| TE (Classic) | 1.33 (0.02) | 783 | 248 | 3.2% (1.6) → 3.2% (1.6) | unchanged | 5.7% (5.6) → 0.0% (0.0) | away from par |
| DST (Classic) | 0.94 (0.05) | 284 | 84 | 11.9% (5.1) → 11.9% (5.1) | unchanged | 9.5% (4.6) → 14.3% (5.7) | away from par |
| **Sensitivity: weeks 2–6 (week 1 dropped)** | | | | | | | |
| QB (all runs, wk2-6) | 0.85 (0.07) | 384 | 136 | 4.4% (3.1) → 5.9% (3.2) | away from par | 11.0% (4.4) → 17.6% (5.5) | away from par |
| RB (all runs, wk2-6) | 1.14 (0.04) | 1098 | 366 | 11.2% (2.7) → 7.7% (2.2) | toward par | 10.7% (3.4) → 6.7% (2.8) | away from par |
| WR (all runs, wk2-6) | 1.00 (0.07) | 1732 | 613 | 2.8% (1.1) → 2.8% (1.1) | unchanged | 10.4% (3.2) → 10.4% (3.2) | unchanged |
| TE (all runs, wk2-6) | 1.15 (0.05) | 1109 | 417 | 4.1% (1.5) → 3.4% (1.4) | away from par | 9.9% (4.4) → 3.7% (2.1) | away from par |
| K (all runs, wk2-6) | 1.00 (0.19) | 30 | 12 | 0.0% (0.0) → 0.0% (0.0) | unchanged | 8.3% (8.8) → 8.3% (8.8) | unchanged |
| DST (all runs, wk2-6) | 0.97 (0.08) | 384 | 138 | 10.1% (4.4) → 10.9% (4.7) | away from par | 9.4% (4.3) → 13.0% (5.0) | away from par |
| QB (Showdown, wk2-6) | 0.91 (0.08) | 148 | 52 | 3.8% (2.7) → 3.8% (2.7) | unchanged | 7.7% (3.7) → 11.5% (4.4) | toward par |
| RB (Showdown, wk2-6) | 1.12 (0.04) | 431 | 147 | 8.8% (2.4) → 8.2% (2.3) | toward par | 9.5% (3.2) → 6.0% (2.6) | away from par |
| WR (Showdown, wk2-6) | 1.06 (0.07) | 679 | 238 | 2.9% (1.1) → 2.9% (1.1) | unchanged | 10.9% (3.2) → 7.6% (2.7) | away from par |
| TE (Showdown, wk2-6) | 1.15 (0.07) | 452 | 169 | 5.3% (1.7) → 3.6% (1.4) | away from par | 13.0% (5.0) → 6.5% (3.7) | away from par |
| K (Showdown, wk2-6) | 1.00 (0.19) | 30 | 12 | 0.0% (0.0) → 0.0% (0.0) | unchanged | 8.3% (8.8) → 8.3% (8.8) | unchanged |
| DST (Showdown, wk2-6) | 0.94 (0.08) | 148 | 54 | 7.4% (3.6) → 9.3% (4.0) | away from par | 9.3% (4.0) → 13.0% (4.8) | away from par |

**Reading:**
- On weeks 1–6 the fit widens RB, WR and TE (m 1.14, 1.20, 1.34) and narrows QB and DST (0.89, 0.93).
- On weeks 7–8 it over-corrects: the RB, WR and TE below-p10 shares fall well under par. Only RB's above-p95 share moves toward par (11.2% (2.7) → 7.7% (2.2)).
- Dropping week 1 shrinks the WR and TE multipliers toward 1, but the held-out check doesn't improve.
- So no static per-position multiplier qualifies. Weeks 7–8 at m = 1 are already near par at p10 for RB, WR and TE.
- K: 38 training rows; the fit stays at m = 1.00.

## Caveats

- These are per-player marginal tails only. Lineup tails also depend on the dependence between players, which public summaries don't carry.
- p90, p99 and the PIT histogram are not available from public data; see `RESULT.md`.
- Week-to-week variation is larger than the player-game bootstrap se suggests, so read single cells loosely.
- Sunday-day Showdown rows and Classic rows without same-draft-group results use same-date Classic points. Week-6+ runs are published only in the summary schema, the same as weeks 1–5 now.

**Reproduce:** `node scripts/calib-b.cjs <dir with the repo's nfl-dk-*.json, nfl-recap/, nfl-dk-results/, nfl-schedule.json> calib-b.json`, then `node scripts/render-result-b.cjs calib-b.json RESULT-B.md`.

RESULT: tails near-calibrated outside week 1 (weeks 2–8: 6.2% above p95, 11.0% below p10). RB/TE bust more and QB floors are too low. K/DST upside tails on Showdown are not thin (prime-time K 2.0%, DST 0.0% above p95). No weeks 1–6 spread multiplier helps weeks 7–8.
