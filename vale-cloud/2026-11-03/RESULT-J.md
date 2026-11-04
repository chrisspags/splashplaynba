TNF 11/5 is JAX at BAL: JAX −9.5, total 41.5 (`nfl-schedule.json` and the published Showdown run `nfl-dk-showdown-156000.json` agree). For favorites of 8–11 points (153 nflverse games, 2021–25), the favorite wins 82.4% (3.1) of the time and wins by 9+ in 52.3% (4.0). Weighted by those odds, history on the 25 public prime-time Showdowns puts JAX DST 0.19 (0.10) per lineup above its field share in top-1% lineups. BAL K is −0.07 (0.02) and BAL DST −0.06 (0.02); JAX RB is 0.11 (0.08). The per-cell shifts rest on 2–6 games each.

# Vale unit 1103CL-J (11/4/2026): TNF 11/5 Showdown, JAX @ BAL

Public data only (`chrisspags/splashplaynba` `origin/main` and nflverse `games.csv`), aggregates only: no lineups, user ids or entry ids. No site code was changed.

## The game and its line

**`nfl-schedule.json`:**
- JAX @ BAL, kickoff 2026-11-06T01:15Z.
- BAL spread +9.5, total 41.5.
- Source: ESPN-pregame, read 2026-11-03T17:34:56.657Z.

**Showdown run:** `nfl-dk-showdown-156000.json` is posted (run 11/3/2026, 12:37:21 PM ET). Its `publicForecast.lines` gives the same numbers: total 41.5, home spread 9.5, implied JAX 25.5 / BAL 16.

**nflverse:** its 2026 row for this game currently shows `spread_line` -10 (home perspective, so JAX −10) and `total_line` 40.5. That is a current line, not a close. Both lines fall in the same cell window used below.

## Outcome cells and their probabilities (nflverse closing lines, 2021–25, regular season + playoffs)

**Cells:**
- Who wins (favorite or underdog), and by 1–3, 4–8 or 9+ points.
- Each split by whether the final total went over (high) or under (low) that game's closing total.
- These are the same margin bands as unit G. If 1103SH-E used other cuts, they are one line to change in `scripts/unit-j.cjs`.

**Windows:**
- **Main:** games with a closing favorite of 8–11 points (153 games).
- **Checks:** 8–11 with a total of 38–45 (87), and 9–10 (70).
- se is binomial.

| Cell | Favorite 8–11 | … high total | … low total | Fav 8–11, total 38–45 | Fav 9–10 |
|---|---:|---:|---:|---:|---:|
| favorite wins by 9+ | 52.3% (4.0) | 28.1% (3.6) | 22.9% (3.4) | 56.3% (5.3) | 50.0% (6.0) |
| favorite wins by 4–8 | 18.3% (3.1) | 10.5% (2.5) | 7.8% (2.2) | 13.8% (3.7) | 17.1% (4.5) |
| favorite wins by 1–3 | 11.8% (2.6) | 7.2% (2.1) | 4.6% (1.7) | 12.6% (3.6) | 17.1% (4.5) |
| underdog wins by 1–3 | 7.8% (2.2) | 3.9% (1.6) | 3.3% (1.4) | 6.9% (2.7) | 11.4% (3.8) |
| underdog wins by 4–8 | 3.9% (1.6) | 2.6% (1.3) | 1.3% (0.9) | 5.7% (2.5) | 0.0% (0.0) |
| underdog wins by 9+ | 5.2% (1.8) | 2.0% (1.1) | 3.3% (1.4) | 3.4% (2.0) | 2.9% (2.0) |
| Favorite wins | 82.4% (3.1) | | | 82.8% (4.0) | 84.3% (4.3) |
| Favorite covers | 48.4% (4.0) | | | 51.7% (5.4) | 44.3% (5.9) |
| Total goes over | 54.2% (4.0) | | | 63.2% (5.2) | 57.1% (5.9) |

Ties: 1. Total pushes: 3 (in the main window; these are counted in the cell rows but not in high/low).

## K/DST and RB shifts within each cell (25 public prime-time Showdowns)

**Measurement:** same games and top-1% rule as units C and G.
- Values are per top-1% lineup, any slot: the share of lineups holding that team's K, or its DST; for RB, the count of that team's RBs.
- "Shift" = top-1% value − field ownership (own / 100).
- "RB as CPT" compares captain share with captain ownership.
- se = game bootstrap.

**Caveat on transfer:** favorite and underdog here are by each game's own line. Those favorites were mostly small: spreads 1.5 to 9.5, with 2 of 25 at 7 or more. The cells condition on the result, not on the size of the spread.

Columns are each past game's favorite (Fav) and underdog (Dog). For TNF, Fav = JAX and Dog = BAL.

| Cell | Games | Fav K | Fav DST | Fav RB | Fav RB as CPT | Dog K | Dog DST | Dog RB | Dog RB as CPT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| favorite wins by 9+ | 6 | 0.09 (0.16) | 0.38 (0.17) | 0.28 (0.11) | 0.18 (0.16) | −0.16 (0.03) | −0.12 (0.03) | −0.32 (0.18) | −0.13 (0.03) |
| favorite wins by 4–8 | 3 | −0.15 (0.05) | −0.02 (0.11) | −0.06 (0.27) | 0.12 (0.14) | −0.11 (0.02) | −0.04 (0.02) | 0.46 (0.02) | 0.05 (0.05) |
| favorite wins by 1–3 | 2 | 0.34 (0.14) | 0.20 (0.35) | 0.01 (0.04) | −0.20 (0.05) | 0.10 (0.10) | 0.03 (0.11) | 0.40 (0.01) | −0.11 (0.03) |
| underdog wins by 1–3 | 3 | −0.24 (0.06) | −0.20 (0.05) | −0.13 (0.16) | −0.17 (0.02) | 0.09 (0.12) | 0.00 (0.09) | 0.63 (0.20) | 0.04 (0.08) |
| underdog wins by 4–8 | 5 | −0.04 (0.11) | −0.14 (0.02) | −0.35 (0.21) | −0.20 (0.06) | 0.04 (0.10) | 0.02 (0.03) | 0.44 (0.19) | 0.15 (0.09) |
| underdog wins by 9+ | 6 | −0.24 (0.02) | −0.18 (0.03) | −0.13 (0.16) | −0.02 (0.07) | 0.18 (0.10) | 0.11 (0.07) | 0.25 (0.19) | 0.15 (0.11) |
| favorite wins (pooled) | 11 | 0.07 (0.10) | 0.24 (0.12) | 0.14 (0.10) | 0.09 (0.10) | −0.10 (0.04) | −0.07 (0.03) | 0.02 (0.15) | −0.07 (0.03) |
| underdog wins (pooled) | 14 | −0.17 (0.05) | −0.17 (0.02) | −0.21 (0.11) | −0.12 (0.04) | 0.11 (0.06) | 0.05 (0.04) | 0.40 (0.12) | 0.13 (0.06) |
| high total (pooled) | 13 | −0.02 (0.08) | −0.17 (0.02) | −0.16 (0.13) | −0.12 (0.05) | 0.08 (0.07) | −0.02 (0.03) | 0.29 (0.14) | 0.11 (0.06) |
| low total (pooled) | 12 | −0.12 (0.08) | 0.20 (0.12) | 0.05 (0.10) | 0.07 (0.10) | −0.04 (0.05) | 0.02 (0.05) | 0.18 (0.15) | −0.04 (0.05) |

**Splitting each cell by high/low total leaves 1–4 games per cell** (fav 9+ / high: 2; fav 9+ / low: 4; fav 4–8 / high: 1; fav 4–8 / low: 2; fav 1–3 / high: 1; fav 1–3 / low: 1; dog 1–3 / high: 2; dog 1–3 / low: 1; dog 4–8 / high: 4; dog 4–8 / low: 1; dog 9+ / high: 3; dog 9+ / low: 3). That is too thin to report shifts, so the total enters only through the pooled high and low rows above.

**Reading:**
- **K/DST shifts follow the winner.** The winning side's DST and K sit above field share; the losing side's sit below.
- **The favorite's DST also does better in low-total games:** 0.20 (0.12) pooled, vs −0.17 (0.02) in high-total games.
- **RB tracks the winner too,** most clearly for the underdog's RBs when the underdog wins (0.40 (0.12)).

## TNF expectation: cell probabilities × measured shifts

**Method:**
- Each outcome cell's measured value is weighted by its probability from the main nflverse window. Every cell has 2 or more games, so no pooling fallback was needed.
- se resamples both the nflverse games and the Showdown games within each cell.
- "Field" is the historical field average in those games. It is not a forecast of TNF ownership.
- "Sim" is the run's pre-lock winner exposure (any slot; RBs summed) from `nfl-dk-showdown-156000.json`.

| Team and position | Sim pre-lock winner share | History-weighted top-1% value | Historical field value | Shift (se) |
|---|---:|---:|---:|---:|
| JAX K | 0.37 | 0.28 | 0.25 | 0.02 (0.09) |
| JAX DST | 0.38 | 0.37 | 0.18 | 0.19 (0.10) |
| JAX RB | 0.90 | 0.80 | 0.69 | 0.11 (0.08) |
| JAX RB as CPT | 0.20 | 0.22 | 0.15 | 0.07 (0.09) |
| BAL K | 0.09 | 0.11 | 0.18 | −0.07 (0.02) |
| BAL DST | 0.13 | 0.05 | 0.11 | −0.06 (0.02) |
| BAL RB | 0.62 | 0.63 | 0.58 | 0.05 (0.10) |
| BAL RB as CPT | 0.12 | 0.06 | 0.12 | −0.05 (0.02) |

**Reading:**
- **The sim mostly matches the outcome-weighted history:** JAX DST 0.38 vs 0.37, and JAX RBs 0.90 vs 0.80.
- **Two places where the sim is higher** (gap in units of the history's se, using the shift se as an approximation):
  - JAX K: 0.37 vs 0.28, about 1.0 se. Not distinguishable.
  - BAL DST: 0.13 vs 0.05, about 3.7 se. History says the sim over-rates a big underdog's DST in winning lineups, if shifts measured on small favorites carry over to a 9.5-point line.

## What is and isn't proven

**Proven:**
- The line: JAX −9.5, total 41.5, posted in both public files.
- The outcome-cell probabilities for an 8–11 point favorite (153 games, se ≤ 4.0 points).
- K/DST shifts follow the winning side.

**Not proven:**
- Any single cell's shift (2–6 games each).
- That shifts measured on small favorites transfer to a 9.5-point favorite.
- Anything about TNF ownership, which is not known before lock.

**Reproduce:** `node scripts/unit-j.cjs <data dir> <nflverse games.csv> unit-j.json`, then `node scripts/render-result-j.cjs unit-j.json RESULT-J.md`. `games.csv` is `https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv`.

RESULT: TNF JAX @ BAL, JAX −9.5, total 41.5: the favorite wins 82.4% (3.1) (9+ in 52.3% (4.0), 153 nflverse games). The history-weighted top-1% shift is JAX DST 0.19 (0.10), BAL K −0.07 (0.02), BAL DST −0.06 (0.02), JAX RB 0.11 (0.08). Cell shifts rest on 2–6 games each.
