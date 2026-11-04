Can't tell yet, but it is not a few players or games. The post-10/28 miss is broad: 10 of 247 players carry half of it, close to the 8.7 a uniform shift with noise would give. It is negative on all 5 post nights, including 11/2 (−1.28 (0.56), DK actuals), and it sits in Showdown runs and expensive players (−3.01 (0.89)). The run build did not change, but one input did: from 2026-10-27 the lineup-news step began marking players out and changing projections, where before it had marked none. That lines up with the shift, but the first news night ran 0.94 (0.72), and the players most likely boosted by news (projected 5+ above their own season mean) are not the over-projected ones. For the same players, about three-quarters of the change is production falling against their own season mean (−1.57 (1.09)) and about a quarter is projections rising against it (0.57 (0.36)). At the current size, 3 more post nights put the post level 3 se from zero. The before/after gap can't reach 3 se with only 7 earlier nights.

# Vale unit 1103CL-T (11/4/2026): is the post-10/28 NBA over-projection a model shift?

Public data only (`chrisspags/splashplaynba` `origin/main`), aggregates only: no names; players are counted, not listed. Stokastic fields were not read. No site code was changed.

## Data

- **Games:** the same 44 pre-lock Showdown/Turbo games as unit R (778 player-games, DK points from `nba-box`).
- **Plus 11/2:** `nba-box` still ends with the 11/1 games, but 11/2 has six pre-lock runs, and DK's own `playerActuals` for its Main contests are in `dk-results/2026-11-02.json`. That adds 163 player-games in 4 run units (Showdown games, plus Turbo runs whose games can't be split without a box score). A DK score of 0 is treated as did-not-play.
- **Split** at 10/28, as in R.
- **se** = game-cluster bootstrap, or player bootstrap for the same-player comparison.
- **"Expensive"** = top salary tertile within slate type.

## 1. Who and which games carry the post-10/28 miss (box nights, 373 player-games)

**Benchmark:** "a uniform shift with noise" keeps the same post mean, but randomizes the sign of each player-game's deviation from it (300 draws). It shows how concentrated a broad shift looks at this sample size.

| Measure | Players | Games |
|---|---:|---:|
| Units in the post period | 247 | 22 |
| Units needed to reach 50% of the net miss | 10 | 4 |
| … under a uniform shift with noise | 8.7 | 3.3 |
| Units needed to reach 80% | 17 | 7 |
| … under a uniform shift with noise | 15.5 | 5.7 |
| Share from the single worst unit | 8% | 17% |
| Share from the worst two | 14% | 31% |
| Units with a positive sum | 100 | 7 |

| Post-10/28 bias | Value |
|---|---:|
| All | −1.24 (0.42) |
| Expensive | −3.01 (0.89) |
| Not expensive | −0.36 (0.36) |
| Without the worst 2 games | −0.94 (0.40) |
| Without the worst 10 players | −0.60 (0.43) |
| Share of the net from players with 2+ post games (120 players) | 53% |
| 11/2 (DK actuals) | −1.28 (0.56) |
| 11/2, expensive | −2.07 (1.23) |

**Bias by night** (mean of actual − proj):

| Night | Period | Games | Player-games | Bias |
|---|---|---:|---:|---:|
| 2026-10-21 | before | 3 | 60 | −1.40 (0.44) |
| 2026-10-22 | before | 2 | 38 | −2.90 (1.12) |
| 2026-10-23 | before | 2 | 37 | 0.00 (0.78) |
| 2026-10-24 | before | 4 | 76 | 1.29 (0.76) |
| 2026-10-25 | before | 1 | 17 | 1.59 (0.00) |
| 2026-10-26 | before | 7 | 119 | −0.06 (0.82) |
| 2026-10-27 | before | 3 | 58 | 0.94 (0.71) |
| 2026-10-28 | after | 9 | 152 | −1.13 (0.64) |
| 2026-10-29 | after | 2 | 38 | −1.05 (0.46) |
| 2026-10-30 | after | 8 | 126 | −0.90 (0.84) |
| 2026-10-31 | after | 3 | 57 | −2.41 (1.05) |
| 2026-11-02 | after (DK actuals) | 4 units | 163 | −1.28 (0.56) |

**Slate type:** Showdown 0.10 (0.46) before → −1.62 (0.41) after; Turbo −0.90 (1.24) → −0.66 (0.88). The shift is in Showdown runs. Turbo was already a little high, so the slate mix (more Turbo runs after 10/28) is not the cause.

**Reading:**
- **Not a handful.** The miss is about as spread out as a uniform shift would be.
- **It survives dropping the worst games or players.**
- **It holds on every night after 10/28,** while the nights before were mixed.

## 2. Same players before and after: projection vs their own season-to-date mean

**Sample:** players with a run-game both before and after 10/28, using DK points from all their box-score games before each game (at least 2).

**Identity:** residual = (actual − own mean) − (proj − own mean), so the change in residual = change in production − change in the projection's gap.

| Players | n | proj − own mean, before | after | change | actual − own mean, before | after | change | change in residual | change in proj | change in own mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 174 | 0.99 (0.45) | 1.57 (0.30) | 0.57 (0.36) | 1.66 (0.72) | 0.09 (0.68) | −1.57 (1.09) | −2.15 (0.98) | 1.25 (0.23) | 0.67 (0.33) |
| expensive | 75 | 0.86 (0.62) | 1.49 (0.42) | 0.63 (0.55) | 1.36 (1.27) | −1.18 (0.98) | −2.54 (1.70) | −3.17 (1.48) | 0.87 (0.25) | 0.24 (0.55) |
| not expensive | 99 | 1.10 (0.52) | 1.63 (0.39) | 0.53 (0.40) | 1.88 (0.87) | 1.05 (0.79) | −0.84 (1.29) | −1.37 (1.13) | 1.54 (0.37) | 1.01 (0.32) |

**Reading:**
- **Both moved:** projections rose against recent production (+0.57), and production fell against it (−1.57).
- **The production drop is the larger part,** most clearly for expensive players (−2.54 (1.70) vs 0.63 (0.55)).
- **Neither half is more than about 1.6 se from 0 on its own.**

## 3. Did the published run change?

| Field | Before 10/28 (20 runs) | From 10/28 (21 runs) |
|---|---|---|
| Slate types | Showdown ×18; Turbo ×2 | Showdown ×15; Turbo ×6 |
| `simKernel.build` | nba-sim 2026-10-14a: the real DraftKings contest (A), every lineup in a fixed seat (B, NFL's fixed-seat-replacement-v1), ownership override (C), 100,000 final / 10,000 batch (D), step minutes made full-run (F); minutes-first v1.1 default, inputs v2/v3-step; was 2026-09-29c ×20 | same string ×21 |
| `simKernel.tables` / `created` / `inputs` / `fit` | nba-sim-kernel-v1.1 ×20 · 2026-09-29 ×20 · nba-sim-kernel-inputs-v3-step ×20 · 2025-26+2024-25 ×20 | nba-sim-kernel-v1.1 ×21 · 2026-09-29 ×21 · nba-sim-kernel-inputs-v3-step ×21 · 2025-26+2024-25 ×21 |
| `metricBasis` / `numSim` | fixed-seat-replacement-v1 ×20 / 10000 ×19; 100000 ×1 | fixed-seat-replacement-v1 ×21 / 10000 ×21 |
| Step-minutes table | step-minutes-fullrun-v1 ×20 | step-minutes-fullrun-v1 ×21 |
| Minutes before lock (`simUpdated` vs `slateLabel`) | mean 14.8, range 4–69 | mean 7.3, range 3–10 |
| floor ÷ proj (players proj ≥ 10) | 0.485 | 0.511 |
| ceiling ÷ proj | 1.735 | 1.678 |
| (ceiling − floor) ÷ proj | 1.250 | 1.167 |
| median ÷ proj | 0.940 | 0.941 |
| `lineupNews.changed` per run | 0 ×17; 74 ×2; 99 ×1 | 25 ×1; 50 ×2; 74 ×1; 76 ×1; 120 ×1; 146 ×1; 183 ×3; 214 ×2; 217 ×1; 239 ×1; 242 ×1; 263 ×2; 264 ×1; 288 ×1; 289 ×1; 361 ×1 |

**Lineup-news timeline** (`lineupNews.out` = players the news step marks out; `changed` = projections it changed):

| Night | Runs | Runs marking anyone out | Mean out per run | Mean changed per run |
|---|---:|---:|---:|---:|
| 2026-10-21 | 3 | 0 | 0 | 0 |
| 2026-10-22 | 2 | 0 | 0 | 0 |
| 2026-10-23 | 2 | 0 | 0 | 0 |
| 2026-10-24 | 4 | 0 | 0 | 0 |
| 2026-10-25 | 1 | 0 | 0 | 0 |
| 2026-10-26 | 5 | 0 | 0 | 0 |
| 2026-10-27 | 3 | 3 | 27 | 82 |
| 2026-10-28 | 6 | 6 | 72 | 223 |
| 2026-10-29 | 2 | 2 | 22 | 62 |
| 2026-10-30 | 4 | 4 | 63 | 198 |
| 2026-10-31 | 3 | 3 | 19 | 50 |
| 2026-11-02 | 6 | 6 | 91 | 257 |

**Bias with the news step off vs on:**

| | News step off (no players marked out) | On |
|---|---:|---:|
| All (box nights) | −0.22 (0.49) | −0.95 (0.40) |
| All, including 11/2 | — | −1.04 (0.35) |
| Expensive | −0.56 (0.90) | −2.44 (0.81) |
| proj below own season mean | 1.30 (0.94) | −1.30 (0.61) |
| proj 0–5 above | −2.62 (0.97) | −0.79 (0.59) |
| proj 5+ above | 1.64 (1.17) | −0.71 (0.81) |
| Share of players projected 5+ above their own mean | 30% | 20% |
| 10/27 (news on, before the split) | — | 0.94 (0.72) |

**Reading:**
- **The model version, kernel, inputs and fit are identical throughout.**
- **Two things changed near 10/28.**
  - The lineup-news step went from marking no one out to marking players out in every run, from the evening of 10/27.
  - Runs were saved closer to lock.
  - Ranges also tightened slightly relative to proj.
- **If news-driven minutes redistribution inflated teammates, the miss should sit in players projected well above their own output.** It doesn't. That group's share fell, and its bias is no worse than the rest's.
- **The news step is the leading candidate by timing, not by mechanism.**

## 4. Verdict and nights needed

**Night-level noise:** the night-to-night sd of the mean bias, after removing each period's mean, is 1.24 points.

**The post level vs 0** (the projection target):
- Now: −1.35 on 5 nights, z = 2.45.
- 3 se needs about 8 post nights.

**The before/after gap:**
- Now: −1.28, z = 1.76.
- 2 se needs about 9 post nights.
- 3 se is out of reach with only 7 earlier nights, so the earlier nights can't carry a change-point test.

**Verdict: can't tell yet, with a named lead.**
- **Not noise from a few players or games:** the miss is broad, on every post night, and survives dropping the worst units.
- **Not slate mix.**
- **No model version changed.**
- **The one input that switched on is the lineup-news step (from 2026-10-27).** The per-player signature doesn't point at it, and most of the same-player change is production falling.
- **Next check:** keep the news step's per-player changes (which players it moved, and by how much). With that, its effect can be measured directly instead of by date.

## What is and isn't proven

**Proven:**
- The post-10/28 over-projection is broad, not concentrated.
- It holds on 11/2.
- It is in Showdown runs and expensive players.
- The run build is unchanged.
- The lineup-news step switched on 10/27.

**Not proven:**
- That the news step causes it: it is linked by date only, and the first news night was not over-projected.
- That it persists: z 2.4 on 5 nights.
- Anything about Main-slate runs (none public pre-lock).

**Reproduce:** `node scripts/unit-t.cjs <data dir> unit-t.json`, then `node scripts/render-result-t.cjs unit-t.json RESULT-T.md`.

RESULT: Can't tell yet. The post-10/28 miss is broad (10 of 247 players carry half), negative on all 5 post nights (11/2 −1.28 (0.56)), and in Showdown and expensive players. No build change; the lineup-news step switched on 10/27, linked by date only. About 3 more nights would put it 3 se from zero.
