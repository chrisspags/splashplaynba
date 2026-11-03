No game-level shared scale is visible: σ_g = 0.00 (0.02) (night-bootstrap 95th percentile 0.06). The additive game term is 0.5% (1.4) of the variance of z, and the night term is 0.3% (0.5). Neither the pre-lock total nor the spread predicts a game's scale held out (leave-one-night-out R² −0.12 (0.06) and −0.04 (0.03), 32 games). The sample is small: 44 games and 778 player-games over 11 nights, from Showdown and Turbo runs only. A true σ_g of 0.24 would be missed in 30% of injected runs.

# Vale unit 1103CL-F (11/3/2026): NBA game-level shared volatility

Public data only (`chrisspags/splashplaynba` `origin/main`), aggregates only: no lineups, user ids or entry ids. Stokastic fields in the run files were not read. No site code was changed.

## Data and coverage

**Actuals:** DK points computed from `nba-box/2026-27/*.json.gz`.
- Check against DraftKings' own `playerActuals` in `dk-results/<night>.json`: 1021 player-games matched, 98.5% exact, mean absolute difference 0.020 points.

**Pre-lock means and spreads:** each run's `playerResults` (`proj`, `floor`, `median`, `ceiling`, `hit30`–`hit60`).
- A run counts as pre-lock only if its `simUpdated` is before the lock time in its `slateLabel`.
- 53 runs fall in the window. 41 are pre-lock; 12 were re-run after lock and are excluded.
- Each game uses its latest pre-lock run that carries both teams: 30 Showdown, 14 Turbo games.
- Dropped: 15 projected players who did not play, and 2 with no name match in the box score.

**Not public, which limits coverage to 44 games on 11 nights (2026-10-21 to 2026-10-31):**
- **No pre-lock run for Main or Night slates on these nights.** On `main`, `nba-dk-main.json` and `nba-dk-night.json` are the 11/4 slates, and their history starts 11/2.
- **`nba-proj/` holds only `2026-10-20.json`.** It is a 10/16 read, with no sd and only 3 games.
- **10/20 and 11/1 have no pre-lock run.**
- **The box file ends with the 11/1 games,** so 11/2 has no actuals yet.

**Pre-lock lines:** `nba-tonight/<night>.json` (DraftKings total and spread).
- On some nights that file was re-read after the first tip.
- Those games' lines are excluded from the main fit, leaving 32 of 44 games.
- A second fit includes them.

## 1. z per player-game

**The runs publish quantiles, not an sd.** Checked against the runs' own hit rates:
- `floor` is the 10th percentile (players with floor ≈ 30 have hit30 = 0.899 (n=10)).
- `median` is the 50th (median ≈ 30: hit30 = 0.504 (n=60); ≈ 40: hit40 = 0.501 (n=33)).
- `ceiling` is the 95th (players with ceiling ≈ 30/40/50/60 have hit rates 0.050 (n=49), 0.050 (n=54), 0.050 (n=48), 0.051 (n=22)).

**The sim's player distributions are right-skewed.** On average (ceiling − median) is 1.81 × (median − floor), against 1.28 for a normal, so three versions of z are used:
- **zq (main):** (actual − proj) / sd, with sd = (p95 − p10) / 2.927, the normal-equivalent spread.
- **zpit:** the sim's own CDF at the actual score, mapped to a normal score. It uses floor, median, ceiling and the hit30–60 points, interpolated, and needs no normal assumption.
- **zband:** (actual − proj) / residual sd per $1,000 price band. This is Vale's fallback; it is in-sample.

| z | Mean | Variance | Share with abs(z) > 2 (normal: 4.6%) |
|---|---:|---:|---:|
| zq | −0.054 (0.031) | 0.891 (0.036) | 3.1% (0.5) |
| zpit | −0.004 (0.029) | 0.944 (0.032) | 2.8% (0.5) |
| zband | −0.059 (0.032) | 0.996 (0.035) | 3.7% (0.4) |

The sim's spread is slightly wide: var(zq) is below 1, and fewer than 4.6% of results land beyond 2 sd.

## 2. Variance of z: player, team-in-game, game and night

Method of moments on pair products of centered z, one weight per cluster:
- **night** = mean product for pairs in different games on the same night;
- **game** = (opposite-team pairs in the same game) − night;
- **team-in-game** = (same-team pairs) − (opposite-team pairs);
- **player** = total − same-team pairs.

se = night-cluster bootstrap (600 replicates, 11 nights).

| z | Total variance | Player share | Team-in-game share | Game share | Night share |
|---|---:|---:|---:|---:|---:|
| zq | 0.892 (0.036) | 102.2% (1.2) | −3.0% (2.3) | 0.5% (1.4) | 0.3% (0.5) |
| zpit | 0.942 (0.029) | 102.7% (1.3) | −2.9% (2.8) | 0.2% (1.6) | 0.1% (0.4) |
| zband | 1.003 (0.035) | 102.1% (1.3) | −3.1% (2.5) | 0.8% (1.5) | 0.2% (0.4) |

**Reading:**
- **Almost all the variance is the player's own.**
- **Teammates offset each other slightly** (the team-in-game share is −3.0% (2.3)). This is the zero-sum share of minutes and usage.
- **No shared game or night shift is detectable.** Both shares are within 1 se of 0.

## 3. Lognormal shared scale (z = s · ε, log s ~ N(μ, σ²))

**Estimator:**
- z is demeaned within each team-game, which removes any location shift.
- E[s⁴]/E[s²]² = exp(4σ²), read from products of the two teams' mean squared z in the same game (game + night scale), and from different games on the same night (night scale).
- σ_g² = (game + night) − night.
- se = night bootstrap.

| z | σ_g (game) | σ_night | σ game + night | σ_g² | Bootstrap 95th pct of σ_g | Bootstrap share at 0 | Games |
|---|---:|---:|---:|---:|---:|---:|---:|
| zq | 0.000 (0.021) | 0.135 (0.069) | 0.000 (0.036) | −0.031 (0.019) | 0.062 | 93% | 44 |
| zpit | 0.000 (0.022) | 0.111 (0.062) | 0.000 (0.025) | −0.028 (0.018) | 0.053 | 92% | 44 |

**Leave-one-night-out:**
- zq: σ_g ranges 0.000–0.000 across 11 folds, and 11 of 11 folds put it at 0. Jackknife se of σ_g² is 0.021.
- zpit: σ_g ranges 0.000–0.000 across 11 folds, and 11 of 11 folds put it at 0. Jackknife se of σ_g² is 0.020.
- The zero is stable: no single night carries it.

**Night scale:** Muse's nightly scale was σ ≈ 0.24. Here σ_night = 0.14 (0.07) (zq) on 10 nights with 2 or more games: lower than 0.24, but within about 1.5 se of it.

**Power check:** a known lognormal game scale was injected on these same rows (200 runs each).

| Injected σ_g | Mean estimate, zq (sd) | Runs estimating above 0.10, zq | Mean estimate, zpit (sd) |
|---:|---:|---:|---:|
| 0.12 | 0.011 (0.031) | 6% | 0.017 (0.039) |
| 0.24 | 0.147 (0.100) | 71% | 0.145 (0.104) |
| 0.36 | 0.296 (0.133) | 90% | 0.292 (0.116) |

**What the power check shows:**
- The estimator is biased low at this sample size.
- A game scale of 0.36 would show (estimate above 0.10) in 90% of runs.
- 0.24 would show in 71%.
- 0.12 can't be told from 0 (6%).

**Pre-lock predictors of a game's realized scale.** The target is ½·log of the product of the two teams' mean squared z. Each fit is leave-one-night-out OLS. se = night bootstrap of the held-out predictions.

| Target | Predictor | Lines | Games | Held-out R² | Held-out corr. | Full-sample slope | Folds with the full-sample sign |
|---|---|---|---:|---:|---:|---:|---:|
| game scale | total | read before tip | 32 | −0.12 (0.06) | −0.54 | 0.0010 | 70% |
| game scale | abs. spread | read before tip | 32 | −0.04 (0.03) | −0.27 | 0.0082 | 90% |
| game scale | total + abs. spread | read before tip | 32 | −0.17 (0.08) | −0.45 | 0.0008, 0.0082 | 70%, 90% |
| game mean z | total | read before tip | 32 | 0.01 (0.06) | 0.02 | −0.0062 | 100% |
| game mean z | abs. spread | read before tip | 32 | −0.07 (0.07) | −0.24 | −0.0091 | 90% |
| game scale | total | incl. read after tip | 44 | −0.03 (0.05) | −0.28 | −0.0070 | 100% |
| game scale | abs. spread | incl. read after tip | 44 | −0.05 (0.02) | −0.48 | −0.0054 | 91% |
| game scale | total + abs. spread | incl. read after tip | 44 | −0.08 (0.06) | −0.37 | −0.0069, −0.0052 | 100%, 91% |
| game mean z | total | incl. read after tip | 44 | −0.02 (0.03) | −0.16 | −0.0035 | 100% |
| game mean z | abs. spread | incl. read after tip | 44 | −0.04 (0.03) | −0.23 | −0.0066 | 91% |

No held-out R² is above 0.01. The total and the spread don't predict which games run hot or cold, in scale or in mean.

## 4. Field side: ownership concentration vs (top-1% line − the sim's line). Stopped: input not public

**The sim's projected top-1% line is not in any public file.** `nba-dk-*.json` carry `playerResults`, `contestScenario` (field size and payout table), `settlementCheck` and `simKernel`, but no projected top-1% score.

**What is public:** the real top-1% line and field ownership, in `dk-results/<night>.json` (`summary.topThresholds.top1pct`, `summary.fieldOwnership`).

**What would unblock it:** the projected top-1% score per run, published next to `contestScenario`.

## What is and isn't proven

**Proven on 44 games, 11 nights:**
- A game-level scale of 0.36 or more is unlikely: it would show in 90% of injected runs, and the estimate here is 0.
- No measurable additive game or night shift.
- The pre-lock total and spread don't predict game volatility.
- Teammates' z are slightly negatively related.

**Not proven:**
- That σ_g is 0. A true scale of 0.24 would be missed in 30% of injected runs, and 0.12 can't be told from 0.
- Anything about Main slates. Their pre-lock runs for these nights aren't public, and the Showdown and Turbo runs used here were built close to lock.
- The field-side question (step 4).

**Reproduce:** `node scripts/unit-f.cjs <data dir> unit-f.json`, then `node scripts/render-result-f.cjs unit-f.json RESULT-F.md`. The data dir holds `nba-box`, `nba-tonight`, `dk-results` and `nba-dk-*.json` from `origin/main`.

RESULT: σ_g = 0.00 (0.02) (95th pct 0.06); the game share of z variance is 0.5% (1.4) and the night share 0.3% (0.5). The pre-lock total and spread don't predict it held out (best held-out R² 0.01). 44 Showdown and Turbo games only. Step 4 stopped: the sim's top-1% line is not public.
