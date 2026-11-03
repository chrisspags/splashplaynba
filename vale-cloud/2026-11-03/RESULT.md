No position can be called miscalibrated from this unit. The public data repo no longer carries the sim's per-player draw inputs for 2026 weeks 1–8 (week 9 is unplayed), so per the unit's own rule I stopped at step 1 and produced no calibration numbers.

# Vale unit 1103CL (11/3/2026): per-player sim spread calibration by position. Stopped at step 1.

Public data only, aggregates only. No site code was changed and nothing was released. Nothing was pushed except this file and `scripts/inventory.cjs`.

- **Code repo:** `chrisspags/splashplay-site` `origin/main` @ `719ba810` (11/3 00:19 ET), freshly cloned.
- **Data repo:** `chrisspags/splashplaynba` `origin/main` @ `65a59ddc52` (11/3 01:45 ET).
- Main's history was rewritten since 10/17: it now has 59 commits, and the earlier main `8c73bcfe` is not its ancestor.

## Step 1: what the sim draws from

`nfl/index.html`, inline worker `_nflSimWorker` (lines 393–1377). Its message reads:

`np, proj, tier, teamOf, gameOf, q21, knobs, reserveKernels, kickerKernels, qbOf, dstOf, kOf, oppOf, recvIdx, recvW, recvOffs, wNorm, kIdx, kOffs, qbSec, aMarg, rbIdx, rbOffs, isPC, isRB, seed`

**Marginal per player:**
- a mean (`proj`, or a reserve kernel's `productionMu`) times a tier quantile curve;
- `q21[tier]`: a 21-point table per position tier, from the run's empirical priors `runPriors.marginals[tier].q21` (`nfl/index.html` ~11032);
- with a generalized-Pareto upper tail per tier (`runPriors.marginals[tier].tail`) and `knobs.tailScale`;
- overridden for:
  - backups by **reserve kernels** (`nfl/reserve-marginals.js`, prepared from the pool);
  - kickers by **kicker kernels**;
  - DK WR rows by the **continuous position law** (`aMarg`, `runPriors.posMarginalA`).

**Dependence:** a Gaussian copula with game, pass-catcher, QB composite, DST and kicker loads (`knobs.gLoad`, `pLoad`, `rbGameLoad`, `qbLambda`, `qbGameLoad`, `dstRho`, `kDstLoad`, `kGameLoad`, `corrScale`). Dependence doesn't affect a single player's percentile; the marginal inputs above do.

So reproducing a player's predictive distribution needs, per run:
- his `proj` and `tier`;
- whether a reserve or kicker kernel applied, and its parameters;
- the priors artifact version in force that day (`assets/nfl-empirical-priors.json` in the **private** site repo, which changes over time);
- the run's knobs.

## Files I looked at

| Where | File(s) | What it holds | Draw inputs? |
|---|---|---|---|
| data repo, main | all 145 `nfl-dk-main-<dg>.json`, `nfl-dk-early-only-<dg>.json`, `nfl-dk-afternoon-only-<dg>.json`, `nfl-dk-showdown-<dg>.json` (weeks 1–8: 8 main, 8 early, 8 afternoon, 121 Showdown) | every one is schema `nfl-public-pool-summary-v1`, weeks 1–5 included (republished). `results.playerResults`: avgPts, stdDev, floor (p10), ceiling (p95), boomPct, proj, salary, exposures, win counts. `results.publicForecast.rows`: projDK/FD/CPT, expSnaps, expRoutes, expRushes, plays, implied, opp implied, total, spread, status, value. `private`: `pointer`, `scope` | **No.** 0 of 145 carry `q21`, reserve/kicker kernels, `aMarg`, knobs, `inputSnapshot`, `outcomeReplay` or `workerSource` (`scripts/inventory.cjs`). The inputs sit behind `private.pointer` (`nfl-portfolio-snapshots/_pool/...`), which is not in the repo |
| data repo, main | `nfl-recap/2026-w01..w08.json`, `nfl-dk-results/<date>.json` (9/9–11/2), `nfl-schedule.json` | box scores, snaps; DK's own per-player points (`playerActuals[].fpts`, Showdown includes K and DST) | outcomes only; enough for step 2 |
| site repo (private) | `nfl/index.html`, `nfl/reserve-marginals.js` (by reference), `assets/nfl-empirical-priors.json` (current version only) | the kernel code and today's tier curves | code yes; per-run inputs no |
| site repo (private) | `tools/release-gate/tests/fixtures/nfl-kernel-20260921-slice.json` | a slice of one 9/21 kernel capture | `q21` present; one slate, partial |
| site repo (private) | `tools/release-gate/tests/fixtures/nfl-run-record-sd154947-w06.json.gz` | one week-6 Showdown run record (SEA @ DEN), fields blanked | `q21` present; one slate |
| site repo (private) | `nfl/contest-outcomes/*.json` (2 files + catalog) | contest payout tensors, not player draws | no |

That is two private test-fixture slates, against the 145 runs the unit asks about. They can't support a by-position, by-salary table or a weeks 1–6 versus 7–9 split. They aren't public either.

Pre-rewrite copies of older run summaries (weeks 1–5, which then carried `inputSnapshot` and `outcomeReplay`) still exist as git objects in an old clone. **I did not use them:** main's rewrite took them out of the public repo, and this unit is public data only.

## What the public files do allow (not run; needs a go-ahead)

Each public run publishes its own p10, p95, mean and sd per player, K and DST included on Showdowns. A reduced version of steps 3–5 is possible without draws:
- share of real DK points above the run's p95 and below its p10, by position (QB/RB/WR/TE/K/DST), salary band and prime-time Showdown, for weeks 1–8 with bootstrap se;
- a per-position spread multiplier `m` on `mean + m·(q − mean)`, fitted on weeks 1–6 and checked on weeks 7–8.

It **cannot** give a PIT decile histogram or exact p90/p99 shares, because those need the full marginal.

The 10/17 cloud unit B (same branch, `vale-cloud/2026-10-17/`) already measured the Classic part for weeks 1–5: 7.1% of players beat their own p95 (par 5, se 0.5, 15 slates).

RESULT: blocked at step 1. Per-player sim draw inputs (tier q21 curves and tails, reserve and kicker kernels, WR position law, knobs) are not in the public repo for 2026 weeks 1–9, so no calibration was computed; a p10/p95-only version is possible from the public summaries if wanted.
