RESULT: A=done B=done

# Vale cloud units, Sat 10/17/2026: A (snap ceiling, target step on teams with absences) and B (Classic tail coverage)

Measurements only. No model, data, registration or shipped file was changed. Aggregates only: there are no DraftKings usernames, no lineups and no entry-level rows in this folder.

- **Branch:** `claude/elegant-cerf-xc93y5` (this session's assigned branch).
- **Code repo:** `chrisspags/splashplay-site` `origin/main` @ `cc5d720` (2026-10-17). Engine behavior was read from `nfl/team-opportunity.js` and `nfl/index.html`.
- **Data repo:** `chrisspags/splashplaynba` `main` @ `8c73bcfe` (2026-10-17 10:32 ET).

## Hand-back

**A: no change, on both pieces.** Neither rule beats the plain level-shift bar, and neither beats the engine on held-out weeks.

- **Snap ceiling (share of team snaps vs raw snaps):** no change.
  - The share ceiling chosen on weeks 2–4 was the half-step. Held-out weeks 5–6, all slates (742 rows, 32 team-games, 19 slates):
    - DK-points MAE gain −0.0009 (se 0.0009);
    - snaps MAE gain −0.003 (se 0.005).
  - It also fails the level-shift bar: the shift (−0.96 DK points, fitted on weeks 2–4) beats the chosen rule by 0.066 (se 0.054).
  - The raw ceiling rarely binds in the published runs: 113 of 2,412 training rows and 8 of 742 held-out rows.
  - Where it does bind (training), the engine is already on target (actual minus engine −0.5 snaps, se 1.7). The share ceiling overshoots (actual minus share ceiling −3.6 snaps, se 1.6).
- **Target step on teams with absences:** no change.
  - Measured reality: the direct replacement takes 37% of the absent player's targets (se 7). Proportional-to-prior-share would give him 6%.
  - The engine already gives the replacement at least that much: on held-out rows its targets for him are +0.2 (se 0.6) above actual.
  - Held-out DK-points MAE, 195 rows, 8 team-games, 11 slates:
    - proportional rule +0.24 worse than the engine (se 0.21);
    - measured role split (fitted on weeks 2–4) +0.17 worse (se 0.18).
  - The level-shift bar (−2.0, fitted on training) is itself +0.52 worse than the engine held-out (se 0.21). No rule beats both.
- **n of team-games with an absent regular, weeks 2–6:** 31. That is 0 in week 2 (the definition needs two prior games), 6 in week 3, 11 in week 4, 13 in week 5 and 1 in week 6. 18 of the 31 involve an absent receiver or back with a regular target share.

**B: the Classic sim's tails are too thin at the top, and too thin at the bottom below the stars.** 15 Classic slates (main, early, afternoon, weeks 1–5), 2,922 player-slates, se clustered by slate.

- **Above own p95:** 7.1% (se 0.5) against par 5, z = 3.8. It is the same before 10/1 (7.4%, se 0.8) and since (6.5%, se 0.8). The excess is in players under $4,000 (8.0%), WR (7.6%), TE (9.4%) and both game-total groups.
- **Below own p10, players with p10 > 0 plus DST:** 15.1% (se 2.0) against par 10, z = 2.5.
  - Too many low finishes: under $4,000 (20.0%), $4,000–$6,999 (13.6%), RB (16.8%), WR (16.8%), TE (23.5%).
  - Too few low finishes: $7,000+ (5.0%), QB (4.9%), DST (6.6%). Their floors are set too low.
- **Cells off par by more than 2 se:** listed below, each with its before/since-10/1 split. Every "below p5" cell under par is an artifact of the approximation (see the method notes): the same cells sit *over* par at the measured p10.

## Files used

| What | File(s) |
|---|---|
| Published runs (engine pre-game rows, per-player p10/p95) | every `nfl-dk-main-<dg>.json`, `nfl-dk-early-only-<dg>.json`, `nfl-dk-afternoon-only-<dg>.json`, `nfl-dk-showdown-<dg>.json` with a decodable summary (weeks 1–5, old schema: `summary` → `inputSnapshot.body.forecastContext.rows`, `edits`, `playerResults`); week 6 new schema `nfl-public-pool-summary-v1` (`results.publicForecast.rows/workload`, `results.playerResults`), only `nfl-dk-showdown-154947.json` (SEA @ DEN, 10/15) has been played |
| Snaps, team snaps, targets, box scores | `nfl-recap/2026-w01.json` … `2026-w06.json` (`games[].players[]`: snaps, snapPct, targets, box stats). Team offensive snaps = median of snaps / snapPct over players at 20%+ |
| Real DK points | `nfl-dk-results/<date>.json` → `contests[].playerActuals[].fpts` (Classic contests first, then Showdown, same date). The box-score DK calculation matched DK's official figure exactly for 95.4% of 4,189 player-games (the rest differ by fumbles or 2-point conversions, which the recap lacks) and is used only as a fallback for offense |
| Not used | `nfl-weekly-stats/` (season-to-date SIS sheets, no per-game rows), `dk-standings/` (entry level) |

**Join:**
- 98.0% of engine skill rows with a final team-game matched a box-score player by normalized name and team.
- A player is scored only if he played (snaps > 0 in the box). DST is always scored.
- 126 Classic player-slates were excluded as "did not play" and 116 because the run projected them at 0.

## Unit A, method

- **Rows:**
  - every engine row (RB/WR/TE, expected snaps > 0, not marked OUT) on every published DK run in weeks 2–6, including the single-game Showdown runs;
  - joined to that player's real snaps, targets and DK points;
  - only players with at least one prior game this season;
  - priors (raw snaps, snap shares, target shares) come from recap weeks before the game only.
- **Split:** rules and constants were **chosen on weeks 2–4** (57 runs: 48 Showdown plus main/early/afternoon of weeks 2, 3 and 4) and **held out on weeks 5–6** (19 runs: 15 Showdown plus main/early/afternoon of week 5, plus the week-6 TNF Showdown 154947). The per-run list is in `results/unit-a.json` → `runsUsed`.
- **Clustering:** se clustered by **team-game**. The same team-game appears on the main slate, an early/afternoon slate and its own Showdown, so slate clusters would not be independent.
- **Absent regular:** at least 2 prior games, mean snap share of 50% or more, played in one of the team's last two games, and no snaps this game. The engine "knew" the absence when that player was not an active row of the run. For the target piece, a "regular target share" is at least 12% (WR/TE) or 10% (RB) of team targets.

### A1: snap ceiling

The engine bounds a reallocated role by `upper = min(teamSnaps, max(before, peakSnaps))`, with `peakSnaps` in raw snaps (`nfl/team-opportunity.js` `roles()`). On the live build the v2 vacancy fill is off, so that step only compresses a group and never expands one.

So the ceiling was tested where it can bind:
- **Binding row:** the published expected snaps sit within 1 snap of the player's prior raw-snap high, while his best prior snap share × the engine's expected team plays is at least 1 snap higher.
- **Variants** (non-binding rows unchanged; DK points scaled in proportion to snaps):
  - raise to the best-share ceiling;
  - raise halfway to it;
  - raise to the mean-of-top-two-shares ceiling.

| Split | Rows | Team-games | Slates | Binding rows (team-games) | Engine snaps MAE | Engine DK MAE |
|---|---:|---:|---:|---:|---:|---:|
| Weeks 2–4 (choose), all slates | 2,412 | 96 | 57 | 113 (26) | 8.92 (0.45) | 4.25 (0.14) |
| Weeks 5–6 (hold out), all slates | 742 | 32 | 19 | 8 (4) | 9.22 (0.70) | 4.21 (0.24) |
| Weeks 5–6, Classic slates | 426 | 22 | 3 | 4 (2) | 9.29 (0.78) | 4.27 (0.27) |
| Weeks 5–6, Showdown slates | 316 | 32 | 16 | 4 (4) | 9.12 (0.72) | 4.14 (0.21) |
| Weeks 5–6, rows in a group with a known absence | 137 | 14 | 14 | 3 (1) | 12.33 (1.67) | 4.18 (0.74) |

MAE change against the engine (negative = better; se in brackets):

| Variant | Weeks 2–4 snaps | Weeks 2–4 DK | Weeks 5–6 snaps | Weeks 5–6 DK |
|---|---:|---:|---:|---:|
| Raise to best-share ceiling | +0.009 (0.026) | +0.000 (0.008) | −0.006 (0.010) | −0.002 (0.002) |
| **Raise halfway (chosen on weeks 2–4)** | −0.003 (0.015) | −0.0004 (0.004) | −0.003 (0.005) | **−0.0009 (0.0009)** |
| Raise to mean-of-top-two ceiling | +0.008 (0.011) | +0.008 (0.004) | −0.005 (0.003) | −0.001 (0.0006) |

**Level-shift bar:** −0.96 DK points, the median residual on weeks 2–4.
- Held out, it changes DK MAE by −0.067 (se 0.054) against the engine.
- The chosen ceiling is +0.066 (se 0.054) against the shift. With the shift added to both, the gap is −0.0009 (se 0.0009).
- So the ceiling does not beat the bar.

Binding rows only:

| | Training (n 113, 26 team-games) | Held out (n 8, 4 team-games) |
|---|---:|---:|
| Actual minus engine, snaps | −0.5 (se 1.7) | +8.1 (se 7.5) |
| Actual minus best-share ceiling, snaps | −3.6 (se 1.6) | +6.5 (se 7.1) |

Vale's example (Braelon Allen, NYJ) shows the short-game problem is real but not the raw cap:
- Week 4: 30 of the team's 32 snaps.
- Week 5: the engine expected 31.5 snaps (below his 34-snap raw high, so not at the cap) and he played 60 of 70.
- The week-5 miss came from the role estimate, not the ceiling. In week 4 the engine had expected 45 snaps, above his raw high at the time.

### A2: target step on teams with absences

**Measured redistribution (recap only; 18 team-games with an absent regular receiver or back, weeks 3–6).**
- V = the absent players' prior target share; mean 24.2% of team targets (se 2.1).
- For each remaining player: gain = actual target share minus his own prior share (taken from prior games where the absent player played).
- Direct replacement = the same-position player with the largest snap-share rise.

| Remaining group | Share of V gained, actual | Proportional to prior share would give | Actual minus proportional |
|---|---:|---:|---:|
| Direct replacement | 0.37 (se 0.07) | 0.06 (se 0.01) | **+0.31 (se 0.07)** |
| Other WRs | 0.44 (se 0.13) | 0.51 (se 0.03) | −0.07 (se 0.13) |
| RB and TE | 0.42 (se 0.11) | 0.43 (se 0.03) | −0.01 (se 0.11) |

- The gains sum to about 1.2 × V. Players with no prior share count in full, and part-time absences are not netted out.
- Absent WR: 15 team-games; absent TE: 1; absent RB: 2. Training (weeks 3–4): 10 team-games; held out (weeks 5–6): 8.

**Scoring on DK points:**
- Rows: engine rows of the remaining players on team-games where the run knew the absence. The engine's own team target total is kept.
- Target changes are converted at DK points per target by position, measured on weeks 2–4: RB 1.44, WR 1.79, TE 1.83.
- Measured split fitted on weeks 2–4: replacement 0.34, other WRs 0.45, RB/TE 0.37 of V.

| Split | Rows / team-games / slates | Engine | Proportional | Measured split | Level shift (−2.0) |
|---|---|---:|---:|---:|---:|
| Weeks 2–4 (choose) DK MAE | 191 / 10 / 13 | 5.41 (0.50) | 5.57 (0.55) | 5.36 (0.60) | 5.04 (0.57) |
| Weeks 5–6 (hold out) DK MAE | 195 / 8 / 11 | **3.85 (0.58)** | 4.09 (0.53) | 4.02 (0.53) | 4.37 (0.68) |
| Weeks 5–6, Classic | 118 / 6 / 3 | 3.94 (0.64) | 4.18 (0.59) | 4.09 (0.60) | 4.48 (0.75) |
| Weeks 5–6, Showdown | 77 / 8 / 8 | 3.71 (0.53) | 3.96 (0.47) | 3.91 (0.47) | 4.21 (0.61) |

Held-out paired differences (rule minus engine, DK MAE):
- proportional +0.24 (se 0.21);
- measured split +0.17 (se 0.18);
- level shift +0.52 (se 0.21).

Rule minus level shift:
- proportional −0.28 (se 0.37);
- measured split −0.35 (se 0.33).

With the shift added to both, the rules lose: +0.29 (se 0.18) and +0.17 (se 0.14). Engine targets minus actual on held-out rows: +0.17 (se 0.16) overall, +0.22 (se 0.63) for the replacement.

## Unit B, method

**Rows:**
- Every Classic run (main, early, afternoon) of weeks 1–5. The week-6 Classic slates have not been played.
- Each run's own `playerResults[].floor` and `ceiling`. These are the run's per-player p10 and p95 over its simulated worlds (`nfl/index.html`: `floor:q(0.10)`, `ceiling:q(0.95)`).
- Scored against the player's real DK points. A strict inequality counts as a miss.

**Slates measured:** all 15. No rule is chosen in B, so nothing is held out.

| Slate | Week 1 | Week 2 | Week 3 | Week 4 | Week 5 |
|---|---|---|---|---|---|
| main | 151307 | 153428 | 153769 | 154078 | 154468 |
| early | 153068 | 153429 | 153770 | 154079 | 154469 |
| afternoon | 153070 | 153431 | 153772 | 154081 | 154471 |

- Before 10/1: weeks 1–3, 9 slates. Since 10/1: weeks 4–5, 6 slates.

**Cuts:**
- Salary: under $4,000 / $4,000–$6,999 / $7,000+.
- Position: QB, RB, WR, TE, DST.
- Game total: under 44 vs 44+, from the run's own implied totals.

**p90 and p5 are not published.** I read them off a normal line through the run's own p10 and p95. Because player distributions are right-skewed, that line puts p90 too high and p5 too low, so both understate misses. Treat the two starred columns as approximate.

**Lower-tail tests use only players with p10 > 0, plus DST** (DST can score below zero).
- 35.7% of player-slates (56% of those under $4,000, 19% at $4,000–$6,999, none at $7,000+; DST excluded) have p10 = 0, which is a point mass at zero. For them a strict "below p10" cannot fire.
- 43.2% of that group finished at or below 0 (se 1.7, n 1,043).

**Duplication:** early and afternoon players also appear on the main slate of the same Sunday, and slate clusters do not absorb that duplication. Main slates alone: above p95 7.0% (se 1.0).

### Coverage table (share, se clustered by slate; † = more than 2 se off par; * = approximate quantile)

| Group | Cell | Players | Above p95 (par 5) | Above p90* (par 10) | Below p10 (par 10) | n for p10/p5 | Below p5* (par 5) |
|---|---|---:|---:|---:|---:|---:|---:|
| All | all | 2922 | 7.1% (0.5) **†** | 10.2% (0.8) | 15.1% (2.0) **†** | 1879 | 3.3% (0.7) **†** |
| Salary | under $4,000 | 1378 | 8.0% (0.9) **†** | 10.4% (1.1) | 20.0% (4.1) **†** | 601 | 6.7% (1.7) |
| Salary | $4,000-$6,999 | 1424 | 6.1% (0.6) | 9.7% (1.0) | 13.6% (1.6) **†** | 1158 | 1.8% (0.5) **†** |
| Salary | $7,000+ | 120 | 8.3% (2.8) | 14.2% (2.8) | 5.0% (1.5) **†** | 120 | 0.8% (0.8) **†** |
| Position | QB | 244 | 6.6% (2.0) | 10.7% (2.9) | 4.9% (0.9) **†** | 244 | 3.3% (1.1) |
| Position | RB | 686 | 5.3% (0.6) | 9.6% (1.3) | 16.8% (2.1) **†** | 465 | 1.9% (0.6) **†** |
| Position | WR | 1079 | 7.6% (0.9) **†** | 10.1% (1.1) | 16.8% (3.1) **†** | 607 | 1.5% (0.5) **†** |
| Position | TE | 669 | 9.4% (1.8) **†** | 11.5% (2.0) | 23.5% (2.9) **†** | 319 | 9.4% (3.8) |
| Position | DST | 244 | 4.1% (1.6) | 8.2% (1.8) | 6.6% (1.5) **†** | 244 | 2.5% (1.2) **†** |
| Game total | under 44 | 1487 | 6.0% (0.4) **†** | 8.9% (0.5) **†** | 14.5% (2.0) **†** | 925 | 3.4% (0.9) |
| Game total | 44+ | 1435 | 8.2% (1.1) **†** | 11.5% (1.4) | 15.6% (2.1) **†** | 954 | 3.3% (0.7) **†** |
| Slate type | main | 1460 | 7.0% (1.0) **†** | 10.2% (1.2) | 14.6% (3.4) | 935 | 3.2% (1.2) |
| Slate type | early | 956 | 7.4% (0.9) **†** | 10.5% (1.5) | 14.1% (2.0) **†** | 631 | 3.2% (1.1) |
| Slate type | afternoon | 506 | 6.7% (1.1) | 9.7% (0.8) | 18.2% (6.6) | 313 | 3.8% (1.4) |
| Period | before 10/1 | 1804 | 7.4% (0.8) **†** | 10.1% (1.1) | 15.3% (3.0) | 1272 | 3.7% (0.9) |
| Period | since 10/1 | 1118 | 6.5% (0.8) | 10.4% (0.9) | 14.5% (0.7) **†** | 607 | 2.5% (0.9) **†** |

The "Above p95" and "Above p90" columns use all players in the row. The two lower-tail columns use the "n for p10/p5" players. The mean of actual minus projected points is −0.02 (se 0.13), so the sim mean is unbiased.

### Every cell off par by more than 2 se (salary, position, game total; slate type and period are shown in the table above)

| Cell | Metric | Share (se) | Par | z | n / slates | Before 10/1: share (se), n | Since 10/1: share (se), n |
|---|---|---:|---:|---:|---:|---:|---:|
| All: all | above p95 | 7.1% (0.5) | 5% | 3.78 | 2922 / 15 | 7.4% (0.8), 1804 | 6.5% (0.8), 1118 |
| All: all | below p10 | 15.1% (2.0) | 10% | 2.53 | 1879 / 15 | 15.3% (3.0), 1272 | 14.5% (0.7), 607 |
| All: all | below p5 (approx.) | 3.3% (0.7) | 5% | -2.46 | 1879 / 15 | 3.7% (0.9), 1272 | 2.5% (0.9), 607 |
| Salary: under $4,000 | above p95 | 8.0% (0.9) | 5% | 3.35 | 1378 / 15 | 8.9% (1.1), 843 | 6.5% (1.3), 535 |
| Salary: under $4,000 | below p10 | 20.0% (4.1) | 10% | 2.46 | 601 / 15 | 24.1% (4.3), 448 | 7.8% (2.5), 153 |
| Salary: $4,000-$6,999 | below p10 | 13.6% (1.6) | 10% | 2.27 | 1158 / 15 | 11.0% (1.9), 754 | 18.3% (0.8), 404 |
| Salary: $4,000-$6,999 | below p5 (approx.) | 1.8% (0.5) | 5% | -6.79 | 1158 / 15 | 1.1% (0.1), 754 | 3.2% (1.2), 404 |
| Salary: $7,000+ | below p10 | 5.0% (1.5) | 10% | -3.36 | 120 / 15 | 5.7% (2.1), 70 | 4.0% (2.2), 50 |
| Salary: $7,000+ | below p5 (approx.) | 0.8% (0.8) | 5% | -5.09 | 120 / 15 | 1.4% (1.4), 70 | 0.0% (0.0), 50 |
| Position: QB | below p10 | 4.9% (0.9) | 10% | -5.35 | 244 / 15 | 4.0% (0.8), 152 | 6.5% (2.1), 92 |
| Position: RB | below p10 | 16.8% (2.1) | 10% | 3.27 | 465 / 15 | 13.8% (2.8), 304 | 22.4% (1.4), 161 |
| Position: RB | below p5 (approx.) | 1.9% (0.6) | 5% | -4.86 | 465 / 15 | 1.0% (0.5), 304 | 3.7% (1.4), 161 |
| Position: WR | above p95 | 7.6% (0.9) | 5% | 2.80 | 1079 / 15 | 7.8% (1.5), 678 | 7.2% (0.8), 401 |
| Position: WR | below p10 | 16.8% (3.1) | 10% | 2.21 | 607 / 15 | 17.6% (4.3), 437 | 14.7% (1.4), 170 |
| Position: WR | below p5 (approx.) | 1.5% (0.5) | 5% | -7.04 | 607 / 15 | 0.9% (0.5), 437 | 2.9% (1.1), 170 |
| Position: TE | above p95 | 9.4% (1.8) | 5% | 2.39 | 669 / 15 | 12.1% (2.5), 405 | 5.3% (1.5), 264 |
| Position: TE | below p10 | 23.5% (2.9) | 10% | 4.63 | 319 / 15 | 25.6% (3.6), 227 | 18.5% (1.9), 92 |
| Position: DST | below p10 | 6.6% (1.5) | 10% | -2.34 | 244 / 15 | 7.9% (1.9), 152 | 4.3% (2.2), 92 |
| Position: DST | below p5 (approx.) | 2.5% (1.2) | 5% | -2.17 | 244 / 15 | 4.0% (1.8), 152 | 0.0% (0.0), 92 |
| Game total: under 44 | above p95 | 6.0% (0.4) | 5% | 2.76 | 1487 / 15 | 6.8% (0.3), 871 | 5.0% (0.7), 616 |
| Game total: under 44 | above p90 (approx.) | 8.9% (0.5) | 10% | -2.26 | 1487 / 15 | 8.6% (0.6), 871 | 9.4% (0.7), 616 |
| Game total: under 44 | below p10 | 14.5% (2.0) | 10% | 2.26 | 925 / 15 | 14.3% (3.1), 589 | 14.9% (1.7), 336 |
| Game total: 44+ | above p95 | 8.2% (1.1) | 5% | 3.00 | 1435 / 15 | 8.0% (1.6), 933 | 8.4% (1.0), 502 |
| Game total: 44+ | below p10 | 15.6% (2.1) | 10% | 2.63 | 954 / 15 | 16.3% (3.0), 683 | 14.0% (1.2), 271 |
| Game total: 44+ | below p5 (approx.) | 3.3% (0.7) | 5% | -2.57 | 954 / 15 | 4.1% (0.7), 683 | 1.1% (0.5), 271 |

### Old weeks or recent ones?

- **Top tail (above p95): both periods.** 7.4% before 10/1, 6.5% since. The excess is steady for WR and 44+ games. It has eased for players under $4,000 (8.9% → 6.5%), TE (12.1% → 5.3%) and games under 44 (6.8% → 5.0%).
- **Bottom tail (below p10, p10 > 0 plus DST): both periods overall** (15.3% → 14.5%), with the mix moving:
  - Under $4,000 was the old problem (24.1% before, 7.8% since).
  - $4,000–$6,999 and RB are the recent problem (11.0% → 18.3% and 13.8% → 22.4%).
  - TE is high in both periods (25.6% → 18.5%).
- **Floors set too low, both periods:** stars ($7,000+) 5.7% → 4.0%, QB 4.0% → 6.5%, DST 7.9% → 4.3% below p10.
- **Below-p5 cells under par:** these read as the approximation, not the model. In the same cells the measured p10 is over par.

## Reproduce

```
node --max-old-space-size=8192 scripts/build-dataset.cjs <splashplaynba checkout> dataset.json
node --max-old-space-size=8192 scripts/unit-a.cjs dataset.json results/unit-a.json
node --max-old-space-size=8192 scripts/unit-b.cjs dataset.json results/unit-b.json
node scripts/render-b.cjs results/unit-b.json
```

- `dataset.json` is player-level and is **not** committed.
- `results/*.json` hold aggregates only.
- Runtime is a few seconds per step on this 4-core cloud container.
