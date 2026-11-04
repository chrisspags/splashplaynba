The +2.0 is carried by target share and efficiency for each team's top two WRs, not by team pass volume. Band minus other games, per $4,000+ WR: target share 1.28 (0.41) points, efficiency 0.83 (0.49), TDs 0.42 (0.34), team target volume −0.35 (0.39) (WRs got 0.60 (0.31) more targets than projected, while team pass attempts were −0.39 (1.23) vs their prior average). It is concentrated in WR1/WR2 priced $5,500+ and in a handful of offenses: CIN, GB, BAL, NYG, SF carry 90% of the band's excess. Those same teams' WRs did not beat projection outside the band (−1.01 (1.56)), and team-level residuals in and out of the band correlate −0.36. So it is not a team pass-volume miss, and nothing at the team level from weeks 1–6 predicts weeks 7–8. The likeliest under-set input is top-WR target share in these games (+0.6 targets). The pattern was already there in weeks 1–6 (2.08 (0.88)), but it looks like concentrated variance more than a stable input.

# Vale unit 1103CL-P (11/4/2026): what the projections missed for WRs in 2026 games totalling 44.5–48.5

Public data only, aggregates only: no lineups, user ids or entry ids, and no player-level rows (team codes only). No site code was changed.

## Data and the game list

**Sources:**
- **Projections:** the published pre-lock Classic runs for weeks 1–8, `nfl-dk-main-<dg>.json` and `nfl-dk-afternoon-only-<dg>.json` (16 runs). From each: `results.playerResults` (proj, salary, team) and `results.publicForecast` (rows, lines, and `workload`, the run's projected targets per player).
- **Actuals:** `nfl-dk-results/<date>.json` `playerActuals`.
- **Usage:** nflverse `stats_player_week_2026.csv` (weeks 1, 2, 3, 4, 5, 6, 7, 8; targets, receptions, yards, TDs, team pass attempts) and `stats_player_week_2025.csv` for week-1 priors.
- **Lines:** nflverse `games.csv` closing totals.
- Each WR-week is counted once: the Main run when the WR is in it, otherwise Afternoon (115 duplicates skipped).

**Shep's list:** `sh_o_spec.json` is on the `splashplay-site` repository, which is not attached to this session, so the list was rebuilt.
- Every public line source puts 31 of the 92 Classic games in 44.5–48.5: nflverse closing, ESPN pregame in `nfl-schedule.json`, and the run's own line.
- **Shep's 24-game list could not be reproduced.**
- **His WR set looks like salary ≥ $4,000.** With that filter the band residual is 1.92 (0.70) vs −0.08 (0.46) elsewhere, a difference of 2.00 (0.83). That matches his +2.0, so the main sample uses it.
- **With all WRs** the band residual is 0.65 (0.33) vs −0.15 (0.22), so the effect lives in the priced WRs.
- In this rebuild the band beats the rest in 6 of 8 weeks (Shep had every week), and 22 of 31 band games have a positive WR residual sum. Without the top 3 WR games the band residual is 1.29 (0.65).

**se** = game bootstrap.

## 1. Decomposing the WR residual (actual − proj)

**Identity** (each piece is in DK points; rows need projected targets ≥ 2 and nflverse stats, 353 of 377 WR-games). With ppt = proj ÷ projected targets and team projected targets = Σ projected targets over the team's players in the run:
- **team volume** = ppt × (team targets − team projected targets) × projected share;
- **share** = ppt × team targets × (target share − projected share);
- **efficiency** = targets × (non-TD points per target − the projection's non-TD points per target);
- **TDs** = targets × (TD points per target − the projection's TD points per target);
- plus rushing and a small leftover (DK vs nflverse).

The projection's TD part is set at the 2025 league WR rate (0.294 TD points per target), because the runs don't publish projected TDs.

| Component | Band games | Other games | Band − other |
|---|---:|---:|---:|
| residual (actual − proj) | 1.92 (0.70) | −0.08 (0.46) | 2.00 (0.83) |
| targets: team volume | 0.03 (0.31) | 0.37 (0.25) | −0.35 (0.39) |
| targets: share | 0.92 (0.27) | −0.36 (0.30) | 1.28 (0.41) |
| efficiency (non-TD points per target) | 0.65 (0.39) | −0.18 (0.28) | 0.83 (0.49) |
| TDs | 0.36 (0.28) | −0.06 (0.22) | 0.42 (0.34) |
| rushing | 0.12 (0.04) | 0.10 (0.03) | 0.02 (0.05) |
| leftover (DK vs nflverse rounding, unmatched) | 0.12 (0.11) | 0.00 (0.00) | 0.11 (0.12) |
| residual, decomposed rows only | 2.20 (0.73) | −0.12 (0.48) | 2.32 (0.90) |
| targets − projected targets | 0.47 (0.20) | −0.12 (0.23) | 0.60 (0.31) |
| team targets − projected team targets | 0.13 (0.73) | 0.65 (0.71) | −0.53 (0.95) |
| team pass attempts − prior average | −0.01 (0.95) | 0.38 (0.77) | −0.39 (1.23) |

**Reading:**
- **Target share carries the most:** 1.28 (0.41) points, from 0.60 (0.31) extra targets per WR.
- **Then per-target efficiency (0.83 (0.49)) and TDs (0.42 (0.34)).**
- **Team target volume and team pass attempts go the other way:** these teams did not throw more than projected or than their prior average.

## 2. Which WRs

**Role** = rank by 2026 target share on the team, weeks 1–8. "Share of excess" is each group's part of the band's total excess over the other-games mean.

**Projection tiers** (thirds of proj): cuts at 9.2 and 11.9 DK points.

| Split | Group | Band WR-games | Band residual | Other residual | Share of band excess |
|---|---|---:|---:|---:|---:|
| role | WR1 | 56 | 2.77 (1.19) | 0.42 (0.79) | 64% |
| role | WR2 | 48 | 2.50 (1.30) | −0.55 (0.64) | 49% |
| role | WR3 | 11 | −0.45 (1.25) | −0.47 (1.03) | -2% |
| role | WR4+ | 9 | −3.38 (1.39) | −0.44 (1.93) | -12% |
| salary | $4,000–5,499 | 65 | −0.01 (0.76) | 0.35 (0.52) | 2% |
| salary | $5,500–6,999 | 43 | 3.86 (1.44) | −0.47 (0.94) | 68% |
| salary | $7,000+ | 17 | 4.40 (2.87) | −1.29 (1.80) | 30% |
| projection tier | high | 40 | 4.18 (1.81) | −0.87 (1.01) | 68% |
| projection tier | low | 43 | 1.55 (1.16) | 1.24 (0.75) | 28% |
| projection tier | mid | 42 | 0.15 (1.05) | −0.56 (0.81) | 4% |

**Reading:** it is concentrated in WR1 and WR2, in the $5,500+ salary range, and in the top projection tier. WR3 and deeper WRs did not beat projection in these games.

## 3. Which teams and games

**Spread across teams:**
- 31 band games, 62 team-games, 27 distinct teams.
- Teams by number of band games: 7 teams with 1, 8 teams with 2, 9 teams with 3, 3 teams with 4.

**Concentration:** the top 5 teams carry 90% of the band's summed WR residual, and the top 10 carry 125%. The rest net negative.

| Team | Band games | Band WR-games | Band residual (sum) | Band residual (mean) | Same team outside the band (mean, WR-games) | Team targets − projected, band |
|---|---:|---:|---:|---:|---:|---:|
| CIN | 4 | 8 | 61.4 | 7.67 | −0.68 (6) | 2.3 |
| GB | 3 | 8 | 54.2 | 6.78 | −5.28 (4) | 4.0 |
| BAL | 3 | 3 | 44.6 | 14.85 | −1.68 (6) | 2.9 |
| NYG | 2 | 4 | 34.3 | 8.57 | 2.42 (6) | 5.6 |
| SF | 4 | 8 | 22.3 | 2.79 | −1.53 (3) | −3.8 |
| NYJ | 2 | 5 | 21.4 | 4.29 | 0.81 (13) | 4.9 |
| DAL | 1 | 2 | 21.2 | 10.61 | 1.17 (7) | 9.7 |
| HOU | 3 | 6 | 20.7 | 3.45 | −2.32 (7) | 3.5 |
| BUF | 1 | 3 | 10.7 | 3.55 | 1.71 (11) | −1.1 |
| CAR | 2 | 4 | 10.2 | 2.54 | −1.06 (6) | 0.7 |
| … bottom 5 | | | MIN −8.6, KC −10.8, JAX −13.9, LV −14.8, CHI −17.5 | | | |

**Reading:**
- **It is a few offenses,** each in the band 2–4 times, and the top three teams' WRs average about 7 to 15 points over projection there.
- **Their team targets were only modestly above projection,** so the gain came from the top WRs' share of those targets and what they did with them.

## 4. Did the same teams' WRs beat projection outside the band?

| Group (outside the band) | WR-games | Residual |
|---|---:|---:|
| Teams with 2+ band games | 127 | −0.24 (0.62) |
| Top 5 band teams (CIN, GB, BAL, NYG, SF) | 25 | −1.01 (1.56) |
| Other teams | 125 | 0.08 (0.68) |
| Team targets − projected, teams with 2+ band games | 118 | 0.54 (0.89) |
| Team targets − projected, other teams | 116 | 0.76 (1.07) |

**Team-level check:** across 21 teams with 3+ WR-games in each group, the mean WR residual in band games and in other games correlates −0.36.

**Reading:** no. The same offenses' WRs were at or below projection elsewhere, and their team volume outside the band matches other teams'. The fix is not team pass volume.

## 5. Verdict, and whether weeks 1–6 would have caught it

| Weeks | Band games | Band WR-games | Residual, band − other | Share | Efficiency | TDs | Team volume |
|---|---:|---:|---:|---:|---:|---:|---:|
| weeks 1–6 | 23 | 94 | 2.08 (0.88) | 1.36 (0.47) | 0.70 (0.52) | 0.45 (0.37) | −0.38 (0.37) |
| weeks 7–8 | 8 | 31 | 1.93 (2.25) | 0.83 (0.91) | 1.51 (1.14) | 0.25 (0.85) | −0.11 (1.20) |

| Week | Band games | Band WR-games | Band − other |
|---:|---:|---:|---:|
| 1 | 5 | 23 | 3.15 (1.74) |
| 2 | 6 | 24 | −0.39 (1.66) |
| 3 | 3 | 12 | 2.16 (1.44) |
| 4 | 4 | 17 | 3.45 (2.81) |
| 5 | 4 | 13 | 3.22 (2.75) |
| 6 | 1 | 5 | 0.44 (1.51) |
| 7 | 3 | 11 | −0.72 (4.27) |
| 8 | 5 | 20 | 3.98 (2.81) |

**Team carry-over** (27 teams):
- Weeks 1–6 WR residual vs weeks 7–8 WR residual: corr. −0.04 (0.18).
- Weeks 1–6 team target miss vs weeks 7–8 WR residual: corr. 0.09 (0.20).

**Verdict:** the most likely under-set input is top-WR target share in competitive mid-total games, about +0.6 targets (se 0.3) per WR1/WR2, worth 1.3 points.
- Efficiency adds 0.8 and TDs 0.4.
- Team pass attempts and target share at the team level are not under-set.

**Would weeks 1–6 have caught it?** Weeks 1–6 show the same band pattern, so a band-specific flag would have fired before weeks 7–8. But:
- the effect sits in a few offenses;
- it doesn't persist team by team;
- it has no slope on the total;
- it is absent in 2021–25 (Shep's O).

So a fix fit on weeks 1–6 would likely be fitting variance. Re-test it on weeks 9+ before changing projections.

## What is and isn't proven

**Proven:**
- In this rebuild, $4,000+ WRs beat projection by about 2 points more in 44.5–48.5 games than in other games.
- The gap comes from target share and efficiency, not team pass volume.
- It is concentrated in WR1/WR2 and a few offenses.
- Those offenses' WRs don't beat projection elsewhere.

**Not proven:**
- That this matches Shep's exact 24-game list (his file is on `splashplay-site`, not attached here).
- That any of it persists into weeks 9+.
- The TD split, which assumes a league-average projected TD rate.

**Reproduce:** `node scripts/unit-p.cjs <splashplaynba data dir> <nflverse dir> <games.csv> unit-p.json 4000`, the same with `0` for all WRs, then `node scripts/render-result-p.cjs unit-p.json unit-p-allwr.json RESULT-P.md`.

RESULT: The 44.5–48.5 WR excess (+2.0, se 0.8) is top-WR target share (+1.3) and efficiency (+0.8), not team pass volume (−0.3). 90% of it comes from 5 offenses whose WRs did not beat projection elsewhere. It was present in weeks 1–6 but doesn't carry team by team; re-test before changing an input.
