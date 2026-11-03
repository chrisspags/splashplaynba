Pooled over 25 prime-time Showdowns, top-1% lineups do not carry more K or DST than the field, and the sim's winner rates are calibrated on average (FLEX slope 1.05, CPT 0.92). K/DST use in top lineups is game-specific: it follows the team that wins (DST FLEX share 25.5% (6.9) on the winner vs 2.1% (1.3) on the loser; K 28.3% (5.7) vs 8.6% (2.9)). The sim spreads K/DST flat across games, and no pre-lock field predicts which games. So the 11/1 SNF gap is one of those games, not a standing bias in the sim's winner rates. Whether our book's picking makes it worse is not measurable from public data.

# Vale unit 1103CL-C (11/3/2026): what wins prime-time Showdowns vs what our sim calls winners

Public data only, aggregates only: no lineups, user ids, anonIds or entry ids. No site code was changed and the pre-rewrite git objects were not used.

**Data:** `chrisspags/splashplaynba` `origin/main`.
- **Results:** `nfl-dk-results/<date>.json` (topFieldLineups, summary.topThresholds, fieldOwnership, captainOwnership, dupes).
- **Pre-lock runs:** `nfl-dk-showdown-<dg>.json` (`results.playerResults`, `results.publicForecast`).
- **Scores:** `nfl-schedule.json`.
- nflverse lines were not fetched. Game script uses real scores; pre-lock lines come from each run's own `publicForecast` (implied, total, spread).

## Sample

- 25 prime-time Showdown games, weeks 1–8. Every Showdown draft group with ingested results is prime-time; week 9 is unplayed.
- 152 contests and 35191 top lineups. 161 of 35352 were dropped for an unmatched name.
- **Top lineups:** per contest, the `topFieldLineups` at or above the contest's top-1% line. When that line sits deeper than the 250 published lineups, all 250 are used, so in large contests "top" is the top 250 (top 0.2–0.4%).
- Each game's value is the mean over its contests. **se = game bootstrap** (2,000 replicates over the 25 games).
- **Field-implied rates** come from each contest's own `fieldOwnership` (total own %) and `captainOwnership` (CPT own %); FLEX own = total − CPT.
- **Small samples:** 25 games overall, 50 kicker and 50 DST player-games, and only 6 held-out games (weeks 7–8).

## 1. Anatomy of top lineups vs the field

| Feature | Top lineups | Field-implied | Top − field |
|---|---:|---:|---:|
| cptPos QB | 18.4% (4.1) | 22.2% (1.5) | -3.8% (3.9) |
| cptPos RB | 29.5% (6.3) | 28.3% (1.7) | 1.1% (6.7) |
| cptPos WR | 42.9% (7.2) | 34.3% (1.7) | 8.7% (6.5) |
| cptPos TE | 5.4% (2.3) | 10.1% (0.7) | -4.7% (2.3) |
| cptPos K | 1.7% (1.2) | 1.8% (0.2) | -0.0% (1.2) |
| cptPos DST | 2.0% (1.3) | 3.3% (0.3) | -1.3% (1.4) |
| cpt on game winner | 72.5% (6.2) | 48.2% (2.6) | 24.2% (6.0) |
| cpt tier under $4,000 | 5.7% (2.6) | 4.9% (0.6) | 0.9% (2.3) |
| cpt tier $4,000-$7,999 | 13.2% (3.6) | 22.3% (1.3) | -9.1% (3.6) |
| cpt tier $8,000+ | 81.1% (4.4) | 72.8% (1.2) | 8.2% (4.4) |
| cpt own rank 1 | 23.4% (5.6) | 24.3% (1.4) | -0.9% (5.6) |
| cpt own rank 2-3 | 34.3% (6.2) | 27.8% (0.8) | 6.5% (6.4) |
| cpt own rank 4-6 | 23.6% (4.4) | 24.9% (0.9) | -1.4% (4.2) |
| cpt own rank 7+ | 18.7% (4.3) | 22.9% (0.7) | -4.2% (4.4) |
| K in FLEX (count) | 0.383 (0.064) | 0.429 (0.014) | -0.046 (0.064) |
| DST in FLEX (count) | 0.283 (0.071) | 0.265 (0.016) | 0.019 (0.067) |
| K or DST anywhere (count) | 0.704 (0.106) | 0.745 (0.031) | -0.040 (0.098) |
| K in FLEX on captain team (count) | 0.242 (0.048) | not derivable from marginals | — |
| K in FLEX on other team (count) | 0.141 (0.035) | not derivable from marginals | — |
| DST in FLEX on captain team (count) | 0.237 (0.067) | not derivable from marginals | — |
| DST in FLEX on other team (count) | 0.046 (0.019) | not derivable from marginals | — |
| K in FLEX on game winner (count) | 0.283 (0.056) | 0.215 (0.008) | 0.068 (0.054) |
| DST in FLEX on game winner (count) | 0.255 (0.068) | 0.133 (0.010) | 0.122 (0.065) |
| share of slots on game winner | 58.0% (3.4) | 49.3% (1.2) | 8.7% (3.3) |
| split 5-1 | 26.1% (6.4) | not derivable from marginals | — |
| split 4-2 | 39.7% (4.8) | not derivable from marginals | — |
| split 3-3 | 34.2% (6.5) | not derivable from marginals | — |
| split 5-1 toward captain team | 23.8% (6.4) | not derivable from marginals | — |
| salary left ($) | $584 (66) | $1059 (48) | $-475 (43) |
| salary left <= $500 | 65.2% (3.9) | not derivable from marginals | — |
| sum of field own (%) | 176.619 (3.074) | 173.300 (1.842) | 3.319 (2.248) |
| entries in a roster repeated within the top set | 96.7% (0.6) | 95.5% (0.2) | 1.2% (0.5) |

**Notes on the anatomy:**
- Shares are per top lineup, and counts are per lineup.
- "Salary left" uses each run's FLEX and CPT salaries.
- The field-implied salary and own sums are expectations under the ownership marginals.
- Team splits and K/DST team pairing need the field's joint lineups, which aren't published, so they have no field column.
- Duplication compares top entries whose roster repeats within the top set against the field's share of entries in repeated rosters (`summary.dupes`).

**What stands out (hindsight features, not pre-lock signals):**
- Captain on the game winner: 72.5% (6.2) vs 48.2% (2.6).
- $8,000+ captain: 81.1% (4.4) vs 72.8% (1.2).
- Less salary left: $584 (66) vs $1059 (48).
- DST rides with the captain's team: 0.237 (0.067) vs 0.046 (0.019) on the other team.

## 2. Sim winner rates vs real top-lineup rates

**Unit:** one row per player-game (734 rows).
- **Real:** share of top lineups holding the player as CPT, or as FLEX.
- **Sim:** the pre-lock run's `cptWinnerExposure` and `flexWinnerExposure` (share of simulated winning lineups, by role).
- **Field:** that player's CPT or FLEX own.
- se = game-cluster bootstrap.

| Role | Real on sim: intercept | slope | Real on field own: intercept | slope |
|---|---:|---:|---:|---:|
| CPT | 0.003 (0.004) | 0.919 (0.105) | -0.000 (0.004) | 1.012 (0.119) |
| FLEX | -0.009 (0.005) | 1.053 (0.030) | -0.005 (0.006) | 1.038 (0.033) |

The sim's winner rates track real top-lineup rates about as well as the field's own ownership does: slopes are near 1, intercepts near 0.

### Bias by position, salary band and game script (mean per player-game; real − sim)

| Role | Group | n | Real | Sim winner rate | Field own | Real − sim | Real − field |
|---|---|---:|---:|---:|---:|---:|---:|
| FLEX | pos QB | 50 | 54.5% (3.6) | 49.6% (0.8) | 49.6% (1.2) | 4.9% (3.6) | 5.0% (3.5) |
| FLEX | pos RB | 160 | 18.4% (1.9) | 16.2% (0.5) | 15.8% (0.6) | 2.2% (1.8) | 2.6% (1.8) |
| FLEX | pos WR | 250 | 13.3% (1.5) | 14.7% (0.3) | 15.4% (0.4) | -1.4% (1.4) | -2.1% (1.4) |
| FLEX | pos TE | 174 | 10.7% (1.7) | 9.2% (0.4) | 10.5% (0.3) | 1.4% (1.7) | 0.1% (1.7) |
| FLEX | pos K | 50 | 19.2% (3.3) | 25.2% (0.7) | 21.5% (0.7) | -6.1% (3.5) | -2.3% (3.2) |
| FLEX | pos DST | 50 | 14.2% (3.6) | 17.9% (1.0) | 13.2% (0.8) | -3.7% (3.3) | 0.9% (3.4) |
| FLEX | salary under $4,000 | 374 | 6.0% (0.5) | 5.5% (0.2) | 5.6% (0.3) | 0.5% (0.5) | 0.4% (0.5) |
| FLEX | salary $4,000-$7,999 | 214 | 18.7% (1.3) | 21.4% (0.4) | 20.4% (0.5) | -2.7% (1.2) | -1.6% (1.2) |
| FLEX | salary $8,000+ | 146 | 42.9% (1.6) | 40.1% (0.6) | 40.9% (0.8) | 2.7% (1.4) | 2.0% (1.4) |
| FLEX | game winner team | 354 | 18.7% (1.2) | 16.9% (0.3) | 16.7% (0.3) | 1.8% (1.1) | 2.0% (1.2) |
| FLEX | game loser team | 350 | 15.4% (1.2) | 17.2% (0.3) | 17.2% (0.4) | -1.8% (1.2) | -1.8% (1.2) |
| CPT | pos QB | 50 | 9.2% (2.1) | 11.4% (0.7) | 11.1% (0.7) | -2.2% (2.2) | -1.9% (2.0) |
| CPT | pos RB | 160 | 4.6% (1.0) | 4.8% (0.3) | 4.4% (0.3) | -0.2% (1.1) | 0.2% (1.1) |
| CPT | pos WR | 250 | 4.3% (0.8) | 3.1% (0.2) | 3.4% (0.2) | 1.2% (0.7) | 0.9% (0.7) |
| CPT | pos TE | 174 | 0.8% (0.3) | 1.3% (0.1) | 1.5% (0.1) | -0.6% (0.4) | -0.7% (0.3) |
| CPT | pos K | 50 | 0.9% (0.6) | 1.1% (0.1) | 0.9% (0.1) | -0.3% (0.6) | -0.0% (0.6) |
| CPT | pos DST | 50 | 1.0% (0.7) | 1.7% (0.2) | 1.6% (0.1) | -0.7% (0.7) | -0.6% (0.7) |
| CPT | salary under $4,000 | 374 | 0.4% (0.2) | 0.3% (0.0) | 0.3% (0.0) | 0.1% (0.2) | 0.1% (0.2) |
| CPT | salary $4,000-$7,999 | 214 | 1.5% (0.4) | 2.7% (0.1) | 2.6% (0.1) | -1.1% (0.4) | -1.1% (0.4) |
| CPT | salary $8,000+ | 146 | 13.9% (0.8) | 12.4% (0.3) | 12.5% (0.2) | 1.5% (0.7) | 1.4% (0.8) |
| CPT | game winner team | 354 | 4.9% (0.4) | 3.2% (0.1) | 3.3% (0.2) | 1.7% (0.4) | 1.6% (0.4) |
| CPT | game loser team | 350 | 1.9% (0.4) | 3.6% (0.1) | 3.5% (0.2) | -1.7% (0.4) | -1.7% (0.4) |

### Game script inside each position (real game result; real-total terciles cut at 37 and 53 points)

Inside a game the player shares sum to a fixed total, so game script only means something within a position.

| Role | Position | Group | Real | Sim | Real − sim |
|---|---|---|---:|---:|---:|
| FLEX | QB | game winner team | 61.1% (7.0) | 49.8% (0.8) | 11.3% (7.0) |
| FLEX | QB | game loser team | 49.6% (8.0) | 49.6% (1.0) | -0.1% (7.7) |
| FLEX | QB | low total | 50.0% (7.8) | 49.5% (1.5) | 0.5% (7.5) |
| FLEX | QB | mid total | 52.0% (4.7) | 51.5% (0.8) | 0.5% (4.3) |
| FLEX | QB | high total | 60.7% (6.2) | 48.0% (1.4) | 12.7% (6.0) |
| FLEX | RB | game winner team | 22.5% (2.8) | 16.8% (0.8) | 5.7% (2.8) |
| FLEX | RB | game loser team | 15.8% (2.7) | 16.0% (0.8) | -0.3% (2.6) |
| FLEX | RB | low total | 18.0% (3.3) | 14.9% (0.8) | 3.1% (2.9) |
| FLEX | RB | mid total | 17.3% (3.5) | 15.6% (0.7) | 1.6% (3.5) |
| FLEX | RB | high total | 19.9% (3.6) | 18.2% (0.9) | 1.7% (3.6) |
| FLEX | WR | game winner team | 10.4% (1.7) | 13.8% (0.6) | -3.4% (1.9) |
| FLEX | WR | game loser team | 15.0% (2.6) | 15.5% (0.5) | -0.4% (2.3) |
| FLEX | WR | low total | 14.5% (3.7) | 15.1% (0.5) | -0.6% (3.6) |
| FLEX | WR | mid total | 13.8% (2.4) | 14.2% (0.5) | -0.5% (2.2) |
| FLEX | WR | high total | 11.7% (1.8) | 14.7% (0.6) | -3.0% (1.8) |
| FLEX | TE | game winner team | 10.4% (2.2) | 9.1% (0.5) | 1.3% (2.1) |
| FLEX | TE | game loser team | 11.5% (2.0) | 9.2% (0.4) | 2.3% (1.9) |
| FLEX | TE | low total | 7.1% (3.4) | 9.0% (0.6) | -1.9% (3.4) |
| FLEX | TE | mid total | 11.4% (3.3) | 8.4% (0.6) | 3.0% (3.0) |
| FLEX | TE | high total | 12.9% (2.7) | 10.1% (0.5) | 2.8% (2.7) |
| FLEX | K | game winner team | 28.3% (5.7) | 25.6% (1.0) | 2.7% (5.9) |
| FLEX | K | game loser team | 8.6% (2.9) | 24.8% (0.9) | -16.2% (3.1) |
| FLEX | K | low total | 16.4% (4.6) | 25.4% (1.3) | -9.0% (4.3) |
| FLEX | K | mid total | 15.0% (5.6) | 25.3% (0.9) | -10.2% (5.9) |
| FLEX | K | high total | 25.3% (6.3) | 25.0% (1.7) | 0.3% (7.1) |
| FLEX | DST | game winner team | 25.5% (6.9) | 18.4% (1.2) | 7.1% (6.3) |
| FLEX | DST | game loser team | 2.1% (1.3) | 17.3% (1.3) | -15.2% (1.9) |
| FLEX | DST | low total | 26.1% (7.0) | 19.6% (1.8) | 6.5% (6.4) |
| FLEX | DST | mid total | 14.5% (7.0) | 18.2% (1.7) | -3.7% (6.6) |
| FLEX | DST | high total | 3.3% (1.7) | 16.1% (1.7) | -12.8% (2.3) |
| CPT | QB | game winner team | 14.0% (3.9) | 12.3% (1.2) | 1.7% (4.2) |
| CPT | QB | game loser team | 5.2% (2.2) | 10.5% (0.9) | -5.3% (2.1) |
| CPT | QB | low total | 8.4% (4.4) | 10.4% (1.0) | -2.0% (4.6) |
| CPT | QB | mid total | 7.3% (2.7) | 13.6% (1.5) | -6.3% (2.6) |
| CPT | QB | high total | 11.7% (3.6) | 10.4% (1.2) | 1.3% (3.7) |
| CPT | RB | game winner team | 7.1% (2.0) | 4.2% (0.6) | 2.9% (1.9) |
| CPT | RB | game loser team | 1.6% (0.6) | 5.7% (0.6) | -4.2% (1.0) |
| CPT | RB | low total | 1.6% (1.3) | 4.2% (0.6) | -2.6% (1.6) |
| CPT | RB | mid total | 8.7% (1.3) | 4.6% (0.5) | 4.1% (1.4) |
| CPT | RB | high total | 3.7% (1.8) | 5.7% (0.5) | -2.1% (1.8) |
| CPT | WR | game winner team | 5.5% (1.7) | 2.8% (0.3) | 2.7% (1.5) |
| CPT | WR | game loser team | 3.2% (1.1) | 3.3% (0.3) | -0.1% (1.0) |
| CPT | WR | low total | 7.3% (1.3) | 3.7% (0.3) | 3.6% (1.3) |
| CPT | WR | mid total | 1.6% (0.7) | 2.6% (0.3) | -1.0% (0.7) |
| CPT | WR | high total | 4.2% (1.2) | 3.1% (0.2) | 1.0% (1.2) |
| CPT | TE | game winner team | 1.3% (0.7) | 1.3% (0.2) | -0.0% (0.7) |
| CPT | TE | game loser team | 0.3% (0.2) | 1.3% (0.1) | -1.0% (0.2) |
| CPT | TE | low total | 0.0% (0.0) | 1.1% (0.2) | -1.1% (0.2) |
| CPT | TE | mid total | 0.9% (0.7) | 1.4% (0.3) | -0.5% (0.8) |
| CPT | TE | high total | 1.3% (0.8) | 1.5% (0.1) | -0.2% (0.8) |
| CPT | K | game winner team | 1.6% (1.1) | 1.2% (0.1) | 0.4% (1.1) |
| CPT | K | game loser team | 0.3% (0.1) | 1.1% (0.1) | -0.9% (0.2) |
| CPT | K | low total | 0.0% (0.0) | 1.2% (0.2) | -1.1% (0.1) |
| CPT | K | mid total | 0.4% (0.3) | 1.2% (0.2) | -0.8% (0.4) |
| CPT | K | high total | 2.1% (1.7) | 1.1% (0.3) | 1.0% (1.7) |
| CPT | DST | game winner team | 2.0% (1.4) | 1.8% (0.3) | 0.2% (1.5) |
| CPT | DST | game loser team | 0.1% (0.1) | 1.7% (0.3) | -1.6% (0.3) |
| CPT | DST | low total | 0.4% (0.3) | 1.9% (0.3) | -1.5% (0.3) |
| CPT | DST | mid total | 2.3% (2.0) | 2.0% (0.5) | 0.3% (2.3) |
| CPT | DST | high total | 0.4% (0.3) | 1.4% (0.3) | -1.0% (0.3) |

### K and DST per game (count per lineup: top lineups vs field-implied vs the sim's FLEX winner rate summed over that position)

| Date | Draft group | K top | K field | K sim | DST top | DST field | DST sim |
|---|---|---:|---:|---:|---:|---:|---:|
| 2026-09-09 | 151820 | 0.52 | 0.39 | 0.46 | 1.23 | 0.30 | 0.44 |
| 2026-09-10 | 153072 | 0.66 | 0.47 | 0.67 | 0.40 | 0.29 | 0.46 |
| 2026-09-13 | 153085 | 0.00 | 0.43 | 0.53 | 0.00 | 0.23 | 0.35 |
| 2026-09-14 | 153086 | 0.16 | 0.47 | 0.48 | 0.88 | 0.28 | 0.49 |
| 2026-09-17 | 153434 | 0.01 | 0.43 | 0.75 | 0.00 | 0.22 | 0.45 |
| 2026-09-20 | 153448 | 0.95 | 0.47 | 0.50 | 0.00 | 0.39 | 0.45 |
| 2026-09-21 | 153449 | 0.19 | 0.29 | 0.45 | 0.47 | 0.18 | 0.34 |
| 2026-09-24 | 153775 | 0.56 | 0.47 | 0.54 | 0.20 | 0.32 | 0.46 |
| 2026-09-27 | 153789 | 0.17 | 0.40 | 0.43 | 0.18 | 0.24 | 0.36 |
| 2026-09-28 | 153790 | 0.22 | 0.44 | 0.47 | 0.25 | 0.32 | 0.51 |
| 2026-10-01 | 154084 | 0.24 | 0.49 | 0.44 | 0.01 | 0.40 | 0.48 |
| 2026-10-04 | 154098 | 0.80 | 0.40 | 0.45 | 0.05 | 0.20 | 0.25 |
| 2026-10-05 | 154099 | 0.16 | 0.39 | 0.48 | 0.04 | 0.22 | 0.33 |
| 2026-10-08 | 154474 | 0.36 | 0.52 | 0.44 | 0.09 | 0.22 | 0.24 |
| 2026-10-11 | 154487 | 0.96 | 0.49 | 0.51 | 0.30 | 0.31 | 0.34 |
| 2026-10-12 | 154488 | 0.61 | 0.33 | 0.44 | 0.01 | 0.13 | 0.17 |
| 2026-10-15 | 154947 | 0.14 | 0.57 | 0.45 | 0.40 | 0.36 | 0.37 |
| 2026-10-18 | 154959 | 0.10 | 0.40 | 0.55 | 0.00 | 0.13 | 0.18 |
| 2026-10-19 | 154960 | 0.02 | 0.38 | 0.58 | 0.05 | 0.21 | 0.24 |
| 2026-10-22 | 155204 | 0.16 | 0.31 | 0.50 | 0.00 | 0.17 | 0.22 |
| 2026-10-25 | 155216 | 0.07 | 0.55 | 0.51 | 0.95 | 0.38 | 0.44 |
| 2026-10-26 | 155217 | 0.75 | 0.36 | 0.44 | 0.00 | 0.21 | 0.31 |
| 2026-10-29 | 155596 | 0.12 | 0.38 | 0.53 | 0.13 | 0.25 | 0.30 |
| 2026-11-01 | 155951 | 0.94 | 0.42 | 0.49 | 0.96 | 0.30 | 0.36 |
| 2026-11-02 | 155609 | 0.72 | 0.50 | 0.50 | 0.47 | 0.36 | 0.38 |

**Is the K/DST gap in the sim's winner rates or only in our picking?**
- **Not a standing sim bias.** Pooled, the sim's K FLEX winner rate is 25.2% (0.7) against a real 19.2% (3.3), and DST 17.9% (1.0) against 14.2% (3.6): if anything, slightly high.
- **K/DST is all-or-nothing by game.** Top lineups load up on a kicker in 6 of 25 games (0.7+ per lineup) and on a DST in 4 of 25; otherwise they mostly leave them out.
- **The sim gives every game about the same K/DST rate**, so it can't tell those games apart.
- **The real split runs on the game result.** Real FLEX share on the winner vs the loser: K 28.3% (5.7) vs 8.6% (2.9), DST 25.5% (6.9) vs 2.1% (1.3). The sim has K 25.6% (1.0) vs 24.8% (0.9) and DST 18.4% (1.2) vs 17.3% (1.3).
- **On 11/1 SNF** (draft group 155951) top lineups held 0.94 kickers per lineup vs a field 0.42 and a sim 0.49.
- **Our book's own K/DST rates aren't public,** so "only in our picking" can't be tested here.

## 3. Pre-lock predictors of top-lineup inclusion (any slot); fit weeks 1–6, held out weeks 7–8

| Model (OLS on the player-game's real top-lineup share) | Train R² | Held-out R² | Held-out MAE | Held-out within-game rank corr. |
|---|---:|---:|---:|---:|
| position only | 0.156 | 0.054 | 0.2685 | 0.088 |
| position + proj + salary | 0.366 | 0.330 | 0.2040 | 0.462 |
| position + sim winner exposure | 0.383 | 0.373 | 0.1979 | 0.479 |
| position + proj + salary + sim winner exposure | 0.383 | 0.370 | 0.1988 | 0.471 |
| all pre-lock fields | 0.393 | 0.360 | 0.2028 | 0.465 |

- **Sample:** training 559 player-games (19 games); held out 175 player-games (6 games). Week 9 is unplayed.
- **Winner exposure vs projection:** the sim's winner exposure edges projection + salary on held-out R² and within-game rank correlation. On 6 games that difference is not established.
- **No added value from more fields:** adding ceiling, sd, role winner rates and lines doesn't help.

**What predicts K/DST inclusion?** Rank correlation of each pre-lock field with the real FLEX share, all weeks:
- K (50 player-games): proj -0.19, ceiling -0.17, stdDev -0.04, salary -0.21, exposure -0.15, simFlex -0.13, fieldFlex -0.04, implied -0.13, gameTotalLine -0.06, spread 0.13.
- DST (50 player-games): proj 0.05, ceiling 0.12, stdDev -0.03, salary -0.08, exposure 0.09, simFlex 0.08, fieldFlex 0.05, implied -0.16, gameTotalLine -0.13, spread 0.04.

None is distinguishable from zero at n = 50 (|ρ| < 0.28 is noise at 95%). The real driver is the game result, which isn't known pre-lock; the run's pre-lock spread carries almost none of it.

## What is and isn't proven

**Proven on 25 games:**
- The sim's per-player winner rates are calibrated on average.
- Top lineups don't over-hold K/DST on average.
- K/DST inclusion in top lineups is driven by the game result, and DST travels with the captain's team.
- Top lineups use more salary and captain the winning side.

**Not proven:**
- That our book's picking causes the Showdown losses (book rates aren't public).
- That any pre-lock field identifies K/DST boom games (n = 50 per position).
- That winner exposure beats projection out of sample (6 held-out games).

**Implication to test next on the mini, with book data:** whether books hold enough K/DST in the outcome-paired form top lineups use (K/DST on the captain's team, captain on the expected winner), rather than a flat rate.

**Reproduce:** `node scripts/unit-c.cjs <data dir> unit-c.json`, then `node scripts/render-result-c.cjs unit-c.json RESULT-C.md`.

RESULT: Showdown K/DST is not under-rated by the sim on average. Top-1% lineups hold K/DST at the field rate overall, but load up only in games their team wins, and neither the sim's flat K/DST rate nor any pre-lock field tells which games those are (25 games, small K/DST samples).
