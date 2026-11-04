Our published NBA projections run slightly high: −0.62 (0.32) DK points per player overall. The miss is concentrated in expensive players (top salary tertile −1.55 (0.62)) and at C and PG. The ranges are calibrated: 4.1% (0.7) of actuals land above the p95 ceiling and 10.7% (1.1) below the p10 floor. Only small corrections hold out of sample: a 3.3% shrink of projections above about 23 points (actual = 0.75 + 0.967 × proj; held-out MSE gain 1.12 (0.40) on 89). The bias was about 0 through 10/27 and −1.24 (0.43) from 10/28, so most of it appeared late and no early fit catches it. Unit I's 5–30%-owned excess is not our projections running low on those players: their bias was −0.75 (0.64) (5–15%) and −2.53 (1.42) (15–30%), at or below projection. Projected minutes are not public, so the minutes bands were skipped.

# Vale unit 1103CL-R (11/4/2026): are the published NBA projections biased?

Public data only (`chrisspags/splashplaynba` `origin/main`), aggregates only: no lineups, user ids, entry ids or player-level rows. Stokastic fields were not read. No site code was changed.

## Data

**Runs:** every NBA run in the window with `simUpdated` before the lock in its `slateLabel` (unit F's rule) and no in-game fields.
- 53 runs in the window, 41 pre-lock, 12 excluded as post-lock.
- One run per game (the latest pre-lock run carrying both teams): 30 Showdown and 14 Turbo games.
- There is no pre-lock Main or Night run for these nights (see RESULT-F and RESULT-L).

**Coverage:** 44 games on 11 nights (2026-10-21 to 2026-10-31). That is 778 player-games that played and 15 projected but out; 2 had no box-score match.
- `nba-box` ends with the 11/1 games, and 11/1 has no pre-lock run.

**Actuals:** DK points from `nba-box/2026-27` (98.5% exact against DK in unit F).

**Floor and ceiling:** the run code is not public; `tools/nba-ingest` has no floor or ceiling definition. Unit F checked them against the runs' own `hit30`–`hit60`: floor = 10th percentile, median = 50th, ceiling = 95th.

**Minutes:** not public. The runs carry no minutes, `nba-tonight` rosters carry none, and `nba-proj` has `modelMin` only for 10/20. So the projected-minutes bands and the minutes miss are skipped. Projection tier stands in for a pre-lock role split.

**Game script:** spread and total from `nba-tonight/<night>.json` (DraftKings line). 32 of 44 games have a line read before tip; a second split includes lines read after the first tip (44 games).

**Tiers:**
- Salary tertiles within slate type: Showdown cuts $4400 / $7000, Turbo $4700 / $6200.
- Projection tertiles: 19.2 / 29.5 DK points.

**se** = game-cluster bootstrap.

## 1 and 3. Mean bias (actual − proj) and range calibration

| Split | Group | Player-games | Mean proj | Bias | Above ceiling (par 5%) | Below floor (par 10%) | Below median (par 50%) |
|---|---|---:|---:|---:|---:|---:|---:|
| overall | all | 778 | 25.3 | −0.62 (0.32) | 4.1% (0.7) | 10.7% (1.1) | 47.6% (1.8) |
| position (first listed) | C | 129 | 23.8 | −1.50 (0.86) | 1.6% (1.1) | 12.4% (2.6) | 55.8% (4.4) |
| position (first listed) | PF | 132 | 27.2 | 0.28 (0.96) | 5.3% (2.1) | 7.6% (2.5) | 39.4% (4.9) |
| position (first listed) | PG | 219 | 28.0 | −1.31 (0.66) | 5.0% (1.4) | 12.8% (2.3) | 50.2% (3.4) |
| position (first listed) | SF | 149 | 24.8 | −0.92 (0.64) | 3.4% (1.4) | 10.7% (2.4) | 49.0% (3.2) |
| position (first listed) | SG | 149 | 21.6 | 0.65 (0.60) | 4.7% (1.9) | 8.7% (2.5) | 42.3% (3.4) |
| salary tier | high | 270 | 37.4 | −1.55 (0.62) | 4.8% (1.2) | 10.7% (1.7) | 47.4% (3.4) |
| salary tier | low | 251 | 14.9 | 0.17 (0.44) | 4.0% (1.3) | 8.0% (1.7) | 50.2% (2.9) |
| salary tier | mid | 257 | 22.8 | −0.42 (0.52) | 3.5% (1.1) | 13.2% (2.0) | 45.1% (3.0) |
| projection tier | high | 260 | 38.2 | −1.33 (0.64) | 4.2% (1.1) | 11.2% (1.9) | 44.2% (3.3) |
| projection tier | low | 259 | 13.6 | 0.17 (0.45) | 3.9% (1.3) | 9.7% (1.9) | 48.6% (3.3) |
| projection tier | mid | 259 | 24.1 | −0.70 (0.60) | 4.2% (1.1) | 11.2% (2.2) | 49.8% (3.3) |
| slate type | Showdown | 569 | 24.3 | −0.58 (0.33) | 4.0% (0.8) | 11.4% (1.3) | 47.1% (1.8) |
| slate type | Turbo/Classic | 209 | 28.0 | −0.73 (0.71) | 4.3% (1.2) | 8.6% (1.2) | 48.8% (4.7) |
| team spread (line read before tip) | favorite by 8+ | 61 | 27.2 | −0.80 (0.84) | 3.3% (1.8) | 6.6% (2.9) | 47.5% (5.3) |
| team spread (line read before tip) | favorite by <8 | 223 | 25.9 | −0.50 (0.66) | 3.6% (1.3) | 12.1% (2.5) | 45.7% (3.1) |
| team spread (line read before tip) | underdog by 8+ | 60 | 23.9 | 0.27 (1.33) | 8.3% (1.7) | 13.3% (1.6) | 45.0% (6.7) |
| team spread (line read before tip) | underdog by <8 | 226 | 24.1 | −0.44 (0.47) | 4.9% (1.3) | 10.6% (2.4) | 49.1% (3.3) |
| total (line read before tip) | total 225–235 | 289 | 24.9 | −0.51 (0.50) | 4.2% (1.1) | 11.8% (1.9) | 45.0% (2.3) |
| total (line read before tip) | total < 225 | 126 | 23.9 | 0.30 (0.79) | 6.3% (1.5) | 9.5% (1.9) | 49.2% (5.0) |
| total (line read before tip) | total > 235 | 155 | 26.4 | −0.85 (0.69) | 3.9% (1.4) | 11.0% (3.1) | 49.7% (3.9) |
| team spread (incl. lines read after tip) | favorite by 8+ | 77 | 27.7 | −0.60 (0.71) | 2.6% (1.5) | 5.2% (2.5) | 48.1% (4.9) |
| team spread (incl. lines read after tip) | favorite by <8 | 308 | 26.2 | −0.94 (0.49) | 3.2% (1.0) | 13.0% (2.0) | 48.1% (2.8) |
| team spread (incl. lines read after tip) | underdog by 8+ | 77 | 23.9 | −0.32 (1.24) | 6.5% (1.8) | 10.4% (2.3) | 46.8% (6.8) |
| team spread (incl. lines read after tip) | underdog by <8 | 316 | 24.2 | −0.39 (0.49) | 4.7% (1.2) | 9.8% (1.8) | 47.2% (2.8) |
| total (incl. lines read after tip) | total 225–235 | 408 | 25.3 | −0.68 (0.44) | 3.9% (0.9) | 10.8% (1.5) | 44.4% (2.6) |
| total (incl. lines read after tip) | total < 225 | 163 | 23.9 | −0.16 (0.67) | 6.1% (1.5) | 11.7% (1.9) | 50.9% (4.3) |
| total (incl. lines read after tip) | total > 235 | 207 | 26.4 | −0.88 (0.58) | 2.9% (1.1) | 9.7% (2.3) | 51.2% (3.3) |

**Reading:**
- **The bias rises with price and projection.** Low tiers are unbiased; the top tertile is 1.3–1.6 points high.
- **C and PG run high; SG and PF run slightly low.** None is more than about 2 se from 0 on its own.
- **Floors and ceilings hit close to par in every split.** The one exception is big underdogs (8+ points), whose players beat the ceiling 8.3% (1.7) of the time on 60 player-games, a small sample.
- **No total or spread band carries a bias** beyond its se.

## 2. Played vs projected but out

- **15 projected players did not play** (1.9% of projected players; 0.34 per game). Their mean projection was 9.0 points, and 14 of 15 were in the low projection tier.
- **Counting them at 0 moves the overall bias** from −0.62 (0.32) to −0.78 (0.29).
- **Minutes miss:** not measurable (no public projected minutes).

## 4. Hold-out: fit on nights through 10/27, check on 10/28 onward

**Split:** train 2026-10-21, 2026-10-22, 2026-10-23, 2026-10-24, 2026-10-25, 2026-10-26, 2026-10-27 (405 player-games); hold-out 2026-10-28, 2026-10-29, 2026-10-30, 2026-10-31 (373).

**Gain** = held-out MSE of the published projection − MSE after the correction (positive = better); held-out baseline MSE is 88.9.

| Correction | Fitted on train | Held-out MSE gain | Held-out bias before | After | Holds (gain > 2 se) |
|---|---|---:|---:|---:|---|
| overall shift | shift −0.054 | 0.13 (0.05) | −1.24 (0.42) | −1.19 (0.43) | yes |
| shift by position | C −1.790; PF 2.599; PG −1.584; SF −0.532; SG 1.507 | −3.19 (2.02) | −1.24 (0.42) | −1.06 (0.45) | no |
| shift by salary tier | high −0.354; low 0.102; mid 0.112 | 0.59 (0.23) | −1.24 (0.42) | −1.19 (0.43) | yes |
| shift by projection tier | high 0.326; low 0.228; mid −0.707 | −0.56 (0.37) | −1.24 (0.43) | −1.19 (0.44) | no |
| linear in proj (a + b·proj) | intercept 0.753; slope 0.967 | 1.12 (0.40) | −1.24 (0.43) | −1.14 (0.43) | yes |
| shift by slate type | Showdown 0.100; Turbo/Classic −0.902 | −0.06 (0.58) | −1.24 (0.43) | −0.94 (0.45) | no |
| Range scale (floor/ceiling distance × s) | s = 0.99 | — | above ceiling 3.8% (0.9), below floor 11.0% (1.5) | 3.8% (0.9), 11.3% (1.7) | not needed |

**Corrections that hold:** overall shift (gain 0.13 (0.05)); shift by salary tier (gain 0.59 (0.23)); linear in proj (a + b·proj) (gain 1.12 (0.40)). The overall shift passes the test but is negligible (−0.05 points).
- **Linear in proj:** actual = 0.75 + 0.967 × proj. That shrinks projections above about 23 points (a 40-point projection becomes 39.4) and leaves low ones about unchanged.
- **Salary-tier shift:** −0.35 for the top tertile.
- **Both are small,** about 1.3% of MSE.
- **The bigger fact is drift:** the train nights were unbiased, and the hold-out nights run −1.24 (0.43). No correction fitted on the first seven nights removes that. Position shifts and the projection-tier step fail held out.

## 5. Unit I's 5–30% band: are we low on those players?

**Placement:** players are placed by their mean `fieldOwnership` in that night's Main contests (`dk-results`). Projections and actuals are from the Showdown and Turbo runs above. "Not in Main pool" = the player's game was not on the Main slate.

| Main-contest ownership | Player-games | Mean proj | Bias | Above ceiling | Below floor |
|---|---:|---:|---:|---:|---:|
| under 5% | 348 | 21.0 | −0.18 (0.52) | 5.2% (1.2) | 9.2% (1.4) |
| 5–15% | 114 | 31.7 | −0.75 (0.64) | 2.6% (1.4) | 7.9% (2.3) |
| 15–30% | 67 | 35.3 | −2.53 (1.42) | 4.5% (2.4) | 17.9% (5.4) |
| 30%+ | 39 | 35.1 | −1.81 (1.37) | 0.0% (0.0) | 10.3% (4.6) |
| not in Main pool | 210 | 24.0 | −0.46 (0.57) | 3.8% (1.2) | 12.4% (2.5) |

**Reading:**
- **We are not low on those players.** The 5–30% owned ran at or below projection, like the other priced players.
- **So unit I's top-lineup excess from that band is outcome variance** (which of them hit), not a mean the projections miss.
- **It is not something a projection shift would capture.**

## What is and isn't proven

**Proven on 44 games:**
- A small high bias, larger for expensive players.
- Calibrated p10/p95 ranges.
- A small shrink that holds out of sample.
- The 5–30%-owned players are not under-projected.

**Not proven:**
- Anything about Main-slate projections (none public pre-lock).
- Minutes bands (no public projected minutes).
- Whether the late-October drift persists.
- Game-script effects beyond the small big-underdog ceiling note.

**Reproduce:** `node scripts/unit-r.cjs <data dir> unit-r.json`, then `node scripts/render-result-r.cjs unit-r.json RESULT-R.md`. The data dir holds `nba-box`, `nba-tonight`, `dk-results` and `nba-dk-*.json` from `origin/main`.

RESULT: NBA projections run 0.62 (0.32) points high, mostly in expensive players (−1.55 (0.62)), and the ranges are calibrated (4.1% (0.7) above ceiling, 10.7% (1.1) below floor). Only a small shrink holds out of sample (actual = 0.75 + 0.967 × proj, 1.3% MSE). The bias grew after 10/27, and 5–30%-owned players are not under-projected. Minutes not public.
