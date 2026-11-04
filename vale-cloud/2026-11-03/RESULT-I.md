The NBA top-1% line is set by how the field's owned players score that night. The ownership-weighted chalk score (the average field lineup, on actuals) predicts the line held out with R² 0.81 (0.07), and the line sits 62.1 (3.2) points above it. No field-structure feature predicts the line or that gap held out: concentration, HHI, 30%+ count, games, contest size and fee give a best held-out R² of 0.03 for the raw line and 0.01 for the gap. Top-1% lineups beat the average field lineup by 71.3 (3.4) points. About 49 of those come from players owned 5–30%, 23.7 (10.4) from 30%+ players, and −1.5 (3.0) from players under 5%. Sample: 62 contests on 13 nights.

# Vale unit 1103CL-I (11/4/2026): what sets the height of the real NBA top-1% line

Public data only (`chrisspags/splashplaynba` `origin/main`), aggregates only: no lineups, user ids or entry ids. No site code was changed.

## Data

- **Results:** `dk-results/<night>.json`, every NBA night from 2026-10-20 to 2026-11-02 with standings (13 nights; there is no 11/1 file).
- **Contests:** 62 Classic contests, one draft group per night. No Showdown contests are in these files.
- **Fields used:** `summary.topThresholds.top1pct`, `summary.fieldOwnership`, `playerActuals` (DK points and eligible slots), `summary.totalEntries`, `structure.entryFee`, `gameCount`.
- **Top-1% lineups:** `topFieldLineups` at or above the line, or all 250 when the line sits deeper (contests over 25,000 entries). That gives 10987 lineups, 0 unmatched.
- **Salaries are not in the results files.** They are public for 7 nights through the `nba-tonight/<night>.json` slate rows: 2026-10-20, 2026-10-21, 2026-10-22, 2026-10-23, 2026-10-28, 2026-10-29, 2026-10-31. For the other nights, no public file carries that draft group's salaries. The salary-capped chalk lineup and the salary-weighted ownership are therefore limited to those 7 nights.
- **se = night bootstrap** (1,000 replicates; nights are the clusters, since contests on one night share players).

## 1. Top-1% line minus the chalk score

- **(a)** The best salary-legal 8-player Classic lineup from players owned 20% or more, scored on actuals ($50,000 cap; PG, SG, SF, PF, C, G, F, UTIL). It is shown with the cap (7 nights) and without the cap (all nights).
- **(b)** Ownership-weighted actual points × 8. Ownership sums to 795.4 (0.7)%, so this is the average field lineup's score. Check: (b) − field median = −0.3 (0.4) points.

| Measure | Contests | Points |
|---|---:|---:|
| Top-1% line | 62 | 318.7 (7.3) |
| Chalk (a), cap | 33 | 311.2 (16.4) |
| Line − chalk (a), cap | 33 | 6.6 (9.4) |
| Chalk (a), no cap | 62 | 328.2 (10.1) |
| Line − chalk (a), no cap | 62 | −9.5 (7.9) |
| What the cap costs chalk (a) on salary nights | 33 | 12.9 (3.7) |
| Chalk (b) | 62 | 256.6 (5.1) |
| Line − chalk (b) | 62 | 62.1 (3.2) |
| Players owned 20%+ (pool for chalk (a)) | 62 | 13.4 (0.7) |

**By night** (means over that night's contests; "line sd" is the spread of the line across contests that night):

| Night | Contests | Games | Top-1% line | Line sd | Chalk (a), cap | Chalk (a), no cap | Chalk (b) | Players 20%+ | Players 30%+ | Top-decile own share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-10-20 | 6 | 3 | 290.7 | 2.1 | 289.5 | 310.3 | 241.2 | 16.0 | 7.3 | 53.2% |
| 2026-10-21 | 4 | 11 | 358.9 | 0.7 | 346.3 | 375.4 | 288.6 | 10.0 | 5.0 | 66.7% |
| 2026-10-22 | 4 | 2 | 284.8 | 1.3 | 307.3 | 329.5 | 226.4 | 19.5 | 9.3 | 37.2% |
| 2026-10-23 | 5 | 6 | 292.1 | 5.5 | 237.9 | 239.3 | 227.2 | 11.6 | 6.0 | 54.2% |
| 2026-10-24 | 5 | 5 | 318.8 | 1.1 | — | 338.4 | 263.0 | 14.4 | 6.8 | 54.7% |
| 2026-10-25 | 4 | 4 | 306.8 | 1.2 | — | 345.9 | 247.6 | 16.5 | 7.3 | 49.9% |
| 2026-10-26 | 5 | 7 | 345.6 | 2.5 | — | 338.3 | 277.8 | 10.6 | 6.6 | 58.3% |
| 2026-10-27 | 6 | 4 | 301.1 | 1.1 | — | 343.7 | 252.2 | 13.3 | 6.3 | 49.5% |
| 2026-10-28 | 4 | 9 | 314.5 | 1.0 | 293.8 | 294.0 | 261.4 | 10.3 | 6.0 | 62.2% |
| 2026-10-29 | 6 | 3 | 356.9 | 2.6 | 372.9 | 384.9 | 269.4 | 15.3 | 9.5 | 40.5% |
| 2026-10-30 | 5 | 8 | 306.1 | 2.8 | — | 321.0 | 248.6 | 12.8 | 3.6 | 54.5% |
| 2026-10-31 | 4 | 5 | 326.8 | 2.7 | 329.0 | 332.8 | 254.8 | 12.5 | 6.5 | 54.9% |
| 2026-11-02 | 4 | 11 | 347.3 | 2.1 | — | 305.3 | 283.2 | 10.5 | 8.0 | 73.1% |

**Reading:**
- **The line is a night property.** Contests on one night land within a few points of each other.
- **The line sits about 62 points above the average field lineup.** The gap moves by night (about 12 points around its mean), and section 3 finds nothing that predicts that movement.
- **The best capped chalk lineup lands near the line on average** (6.6 (9.4)), but swings by night. On some nights pure chalk would have cleared the line; on others it fell well short.

## 2. Field features (from `fieldOwnership`)

| Feature | Contests | Mean | sd | Min | Max |
|---|---:|---:|---:|---:|---:|
| Top-decile ownership share (top 10% of owned players) | 62 | 0.539 | 0.093 | 0.370 | 0.756 |
| Ownership HHI (shares of total ownership) | 62 | 0.0321 | 0.0062 | 0.0208 | 0.0434 |
| Players at 30%+ | 62 | 6.8 | 1.7 | 2.0 | 10.0 |
| Salary-weighted ownership ($ per roster slot) | 33 | 6235.5 | 4.6 | 6226.6 | 6247.5 |
| Slate size (games) | 62 | 5.8 | 2.8 | 2.0 | 11.0 |
| Contest size (log10 entries) | 62 | 4.3 | 0.4 | 3.1 | 5.0 |
| Entry fee ($) | 62 | 4.7 | 5.3 | 0.5 | 20.0 |

Salary-weighted ownership barely moves (sd 4.6 around 6236 per slot, about $49884 per lineup): the field spends essentially the whole cap every night. It can't explain anything.

## 3. Leave-one-night-out: what predicts the line

**Method:**
- OLS fit on all other nights, predicting each held-out night's contests.
- R² is against the training mean.
- se is a night bootstrap of the held-out predictions.
- "Sign" is the share of folds whose slope has the full-sample sign.
- Chalk (b) uses actuals, so it is an after-the-fact explainer, not a pre-lock predictor.

| Target | Features | Contests | Nights | Held-out R² | Held-out RMSE | Baseline RMSE | Full-sample slopes | Sign |
|---|---|---:|---:|---:|---:|---:|---|---|
| top-1% line | topDecileShare | 62 | 13 | −0.26 (0.40) | 30.5 | 27.2 | top-decile share 87.8 | 100% |
| top-1% line | hhi | 62 | 13 | −0.07 (0.19) | 28.1 | 27.2 | HHI −1231.6 | 100% |
| top-1% line | n30 | 62 | 13 | −0.23 (0.10) | 30.1 | 27.2 | players 30%+ 1.9 | 92% |
| top-1% line | salaryWeightedOwn | 33 | 7 | −0.39 (0.42) | 41.0 | 34.8 | salary-weighted own 1.0 | 86% |
| top-1% line | games | 62 | 13 | 0.03 (0.29) | 26.8 | 27.2 | games 4.1 | 100% |
| top-1% line | logEntries | 62 | 13 | −0.03 (0.01) | 27.6 | 27.2 | log10 entries 0.874 | 77% |
| top-1% line | entryFee | 62 | 13 | −0.01 (0.00) | 27.3 | 27.2 | entry fee −0.112 | 85% |
| top-1% line | games + logEntries + entryFee | 62 | 13 | 0.02 (0.31) | 26.9 | 27.2 | games 4.3; log10 entries −4.1; entry fee 0.169 | 100%, 100%, 100% |
| top-1% line | ownership trio (topDecile, HHI, n30) | 62 | 13 | −0.08 (0.34) | 28.3 | 27.2 | top-decile share 70.2; HHI −3308.0; players 30%+ 12.3 | 100%, 100%, 100% |
| top-1% line | all but salary-weighted | 62 | 13 | −0.30 (0.40) | 31.0 | 27.2 | top-decile share −28.4; HHI −2531.6; players 30%+ 11.4; games 4.6; log10 entries −4.8; entry fee 0.529 | 85%, 92%, 100%, 100%, 85%, 100% |
| top-1% line | chalk (b) only | 62 | 13 | 0.81 (0.07) | 11.8 | 27.2 | chalk (b) 1.2 | 100% |
| top-1% line | chalk (b) + games | 62 | 13 | 0.72 (0.10) | 14.4 | 27.2 | chalk (b) 1.4; games −1.2 | 100%, 92% |
| line − chalk (a, cap) | topDecileShare | 33 | 7 | 0.26 (0.25) | 25.7 | 29.8 | top-decile share 148.7 | 100% |
| line − chalk (a, cap) | hhi | 33 | 7 | 0.22 (0.21) | 26.4 | 29.8 | HHI −2152.2 | 100% |
| line − chalk (a, cap) | n30 | 33 | 7 | 0.39 (0.26) | 23.3 | 29.8 | players 30%+ −9.8 | 100% |
| line − chalk (a, cap) | salaryWeightedOwn | 33 | 7 | 0.35 (0.19) | 24.1 | 29.8 | salary-weighted own 3.9 | 100% |
| line − chalk (a, cap) | games | 33 | 7 | 0.03 (0.61) | 29.4 | 29.8 | games 4.4 | 100% |
| line − chalk (a, cap) | logEntries | 33 | 7 | −0.01 (0.01) | 30.0 | 29.8 | log10 entries 0.338 | 43% |
| line − chalk (a, cap) | entryFee | 33 | 7 | 0.03 (0.02) | 29.4 | 29.8 | entry fee −0.893 | 100% |
| line − chalk (a, cap) | games + logEntries + entryFee | 33 | 7 | 0.05 (0.55) | 29.0 | 29.8 | games 4.6; log10 entries −10.7; entry fee −0.418 | 100%, 100%, 100% |
| line − chalk (a, cap) | ownership trio (topDecile, HHI, n30) | 33 | 7 | 0.24 (0.32) | 26.1 | 29.8 | top-decile share 10.2; HHI −634.3; players 30%+ −7.6 | 43%, 86%, 100% |
| line − chalk (a, cap) | all but salary-weighted | 33 | 7 | 0.13 (0.45) | 27.9 | 29.8 | top-decile share 44.1; HHI −2101.5; players 30%+ −9.6; games −5.1; log10 entries −18.0; entry fee −0.718 | 100%, 100%, 100%, 86%, 100%, 100% |
| line − chalk (a, no cap) | topDecileShare | 62 | 13 | 0.29 (0.18) | 28.4 | 33.7 | top-decile share 195.6 | 100% |
| line − chalk (a, no cap) | hhi | 62 | 13 | 0.08 (0.18) | 32.3 | 33.7 | HHI −2057.4 | 100% |
| line − chalk (a, no cap) | n30 | 62 | 13 | −0.03 (0.12) | 34.2 | 33.7 | players 30%+ −4.2 | 100% |
| line − chalk (a, no cap) | salaryWeightedOwn | 33 | 7 | 0.27 (0.34) | 31.2 | 36.6 | salary-weighted own 4.6 | 100% |
| line − chalk (a, no cap) | games | 62 | 13 | 0.21 (0.28) | 30.0 | 33.7 | games 6.3 | 100% |
| line − chalk (a, no cap) | logEntries | 62 | 13 | 0.00 (0.02) | 33.7 | 33.7 | log10 entries 6.7 | 100% |
| line − chalk (a, no cap) | entryFee | 62 | 13 | 0.03 (0.02) | 33.1 | 33.7 | entry fee −1.2 | 100% |
| line − chalk (a, no cap) | games + logEntries + entryFee | 62 | 13 | 0.19 (0.29) | 30.2 | 33.7 | games 6.2; log10 entries −3.0; entry fee −0.729 | 100%, 92%, 100% |
| line − chalk (a, no cap) | ownership trio (topDecile, HHI, n30) | 62 | 13 | 0.25 (0.21) | 29.2 | 33.7 | top-decile share 170.6; HHI −1597.6; players 30%+ 3.7 | 100%, 100%, 100% |
| line − chalk (a, no cap) | all but salary-weighted | 62 | 13 | 0.11 (0.25) | 31.8 | 33.7 | top-decile share 176.1; HHI −1377.1; players 30%+ 3.3; games −0.154; log10 entries −0.013; entry fee −0.667 | 100%, 92%, 100%, 62%, 54%, 100% |
| line − chalk (b) | topDecileShare | 62 | 13 | −0.31 (0.12) | 14.0 | 12.2 | top-decile share −19.9 | 92% |
| line − chalk (b) | hhi | 62 | 13 | −0.09 (0.08) | 12.8 | 12.2 | HHI −225.6 | 92% |
| line − chalk (b) | n30 | 62 | 13 | −0.20 (0.15) | 13.4 | 12.2 | players 30%+ 1.9 | 92% |
| line − chalk (b) | salaryWeightedOwn | 33 | 7 | −0.13 (0.14) | 16.7 | 15.7 | salary-weighted own 0.328 | 86% |
| line − chalk (b) | games | 62 | 13 | −0.21 (0.06) | 13.4 | 12.2 | games 0.164 | 77% |
| line − chalk (b) | logEntries | 62 | 13 | −0.03 (0.01) | 12.4 | 12.2 | log10 entries −0.683 | 85% |
| line − chalk (b) | entryFee | 62 | 13 | 0.00 (0.01) | 12.2 | 12.2 | entry fee −0.069 | 100% |
| line − chalk (b) | games + logEntries + entryFee | 62 | 13 | −0.23 (0.05) | 13.6 | 12.2 | games 0.177; log10 entries −1.1; entry fee −0.074 | 77%, 85%, 92% |
| line − chalk (b) | ownership trio (topDecile, HHI, n30) | 62 | 13 | 0.01 (0.28) | 12.1 | 12.2 | top-decile share −31.9; HHI −1668.1; players 30%+ 5.8 | 92%, 100%, 100% |
| line − chalk (b) | all but salary-weighted | 62 | 13 | −0.09 (0.40) | 12.8 | 12.2 | top-decile share −61.4; HHI −1526.3; players 30%+ 5.5; games 1.3; log10 entries −3.4; entry fee 0.174 | 100%, 100%, 100%, 100%, 100%, 100% |

**Reading:**
- **Raw line:** only chalk (b) predicts it (R² 0.81 (0.07), slope 1.24). Each extra point the average field lineup scores lifts the line by about 1.2 points. Slate size, contest size, fee and ownership shape do not predict it held out (best: games, R² 0.03 (0.29)).
- **Line − chalk (b):** nothing predicts the gap (best: ownership trio (topDecile, HHI, n30), R² 0.01 (0.28)).
- **Line − chalk (a):** ownership shape does carry some held-out signal, with fewer 30%+ players meaning a bigger gap over the best chalk lineup. But it rests on 7 nights with a se of 0.26, so it is not established.

## 4. Which ownership band carries the excess points

**Comparison:** top-1% lineups vs the ownership-weighted field, which is the average field lineup.

**Pooled over 62 contests:**

| Ownership band | Share of points, top 1% | Share of points, field | Top − field | Points per lineup, top 1% | Points per lineup, field | Excess points |
|---|---:|---:|---:|---:|---:|---:|
| under 5% | 6.3% (1.1) | 8.4% (1.6) | −2.1% (1.1) | 20.6 (3.7) | 22.0 (4.6) | −1.5 (3.0) |
| 5–15% | 23.5% (2.6) | 22.3% (1.7) | 1.3% (1.4) | 76.8 (7.9) | 57.4 (4.5) | 19.4 (5.0) |
| 15–30% | 34.0% (2.9) | 31.6% (2.4) | 2.4% (1.3) | 109.8 (8.3) | 80.1 (5.1) | 29.7 (4.5) |
| 30%+ | 36.2% (5.1) | 37.7% (3.6) | −1.6% (2.4) | 120.8 (17.9) | 97.1 (9.6) | 23.7 (10.4) |
| total | | | | | | 71.3 (3.4) |

**By night.** Excess points per top-1% lineup over the field, by band. The share columns use the night's contests; se is over contests within the night, so it is rough.

| Night | Contests | Top-1% lineups | Excess under 5% | Excess 5–15% | Excess 15–30% | Excess 30%+ | Share top − field, under 5% | Share top − field, 30%+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026-10-20 | 6 | 1185 | 4.5 | 16.8 | 42.0 | −5.8 | 1.0% (0.4) | −10.5% (0.7) |
| 2026-10-21 | 4 | 909 | −29.1 | 0.7 | 11.8 | 98.0 | −12.3% (0.7) | 19.2% (0.4) |
| 2026-10-22 | 4 | 761 | 0.1 | −1.0 | 25.9 | 41.0 | −0.2% (0.1) | −0.2% (2.6) |
| 2026-10-23 | 5 | 921 | 17.1 | 47.1 | 31.1 | −17.2 | 2.8% (0.5) | −11.3% (1.2) |
| 2026-10-24 | 5 | 806 | 10.9 | 39.7 | 22.4 | −8.2 | 2.0% (0.6) | −9.1% (2.2) |
| 2026-10-25 | 4 | 475 | −3.7 | 7.7 | 63.1 | −0.9 | −2.2% (0.2) | −8.7% (1.9) |
| 2026-10-26 | 5 | 814 | −6.8 | 8.0 | 41.0 | 34.2 | −4.1% (0.6) | 1.0% (1.2) |
| 2026-10-27 | 6 | 1037 | 5.7 | 27.2 | 17.7 | 6.3 | 1.2% (0.6) | −3.8% (0.6) |
| 2026-10-28 | 4 | 857 | −5.2 | 41.7 | −4.5 | 31.9 | −4.7% (1.5) | 3.9% (1.2) |
| 2026-10-29 | 6 | 879 | −4.1 | −10.1 | 26.8 | 83.2 | −1.7% (0.1) | 8.0% (1.1) |
| 2026-10-30 | 5 | 921 | 4.3 | 27.1 | 55.4 | −19.4 | −1.5% (0.9) | −8.5% (0.5) |
| 2026-10-31 | 4 | 658 | −9.1 | 9.9 | 22.3 | 58.2 | −4.6% (0.2) | 7.2% (1.0) |
| 2026-11-02 | 4 | 764 | −16.4 | 37.9 | 23.9 | 27.2 | −8.3% (0.6) | −1.3% (1.1) |

**Reading:**
- **Top-1% lineups look like the field in ownership mix.** Point shares by band are within a few points of the field's.
- **The extra points come mostly from the 5–30% band:** 19.4 (5.0) + 29.7 (4.5) points.
- **Sub-5% players add nothing on average** (−1.5 (3.0)).
- **The 30%+ band swings by night** (23.7 (10.4)). It adds a lot when the chalk hits and subtracts when it doesn't.

## What is and isn't proven

**Proven on 62 contests, 13 nights:**
- The line tracks the field's average score on actuals, about 62 points above it (night-to-night spread about 12).
- Pre-lock field structure (concentration, 30%+ count, slate size, contest size, fee) doesn't predict the line held out.
- Top-1% excess comes from mid-owned players, not sub-5% plays.

**Not proven:**
- Anything from salaries beyond 7 nights (the other nights' salaries aren't public).
- The ownership-shape signal on line − chalk (a).
- Showdown (no Showdown contests in these files).
- Whether a pre-lock projection of chalk (b) predicts the line as well as the realized chalk (b). That needs pre-lock projections for the Main slates, which aren't public for these nights (see RESULT-F).

**Reproduce:** `node scripts/unit-i.cjs <data dir> unit-i.json`, then `node scripts/render-result-i.cjs unit-i.json RESULT-I.md`. The data dir holds `dk-results/` and `nba-tonight/` from `origin/main`.

RESULT: The NBA top-1% line = the average field lineup's actual score + 62.1 (3.2) (held-out R² 0.81 (0.07)). No field-structure feature predicts the line or the gap. The excess comes from 5–30% owned players, not sub-5% plays (62 contests, 13 nights).
