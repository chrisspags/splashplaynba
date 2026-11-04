An RB1's bad games are a separate bust mode, not a wider lognormal. Against a lognormal fitted on the middle 80%, 12.0% (0.7) of games land under its p10 and 8.4% (0.7) under its p05, while only 3.1% (0.4) land above its p95. A sim-ready mixture with the mean held at 1 fixes the floor: with probability 21% (top tier) or 32% (mid) the RB draws from a bust component at 0.44 × expectation (sd 0.25), and otherwise from a lognormal whose mean rises to 1.15 (top) or 1.27 (mid) × expectation. On 2025 that gives a floor rate of 11.6% (2.2) and a ceiling rate of 4.2% (1.9) for the top tier (target 10% / 5%). The cost of holding the mean: expensive RB1s average 0.93 (0.02) × expectation, so on 2021–24 only 2.4% (0.6) of top-tier games clear the spec's p95. Spread, total and underdog status do not move the bust rate held out.

# Vale unit 1103CL-N (11/4/2026): the real shape of an RB1's bad games

Public data only (nflverse), aggregates only: no player-game rows are published. No site code was changed.

## Data and definitions

**Sources:** nflverse release files `stats_player_week_YYYY.csv` and `snap_counts_YYYY.csv` (2020–25), `play_by_play_YYYY.csv.gz` (2021–25), and `nfldata/games.csv` (closing `spread_line` and `total_line`). Regular season only.
- **Fit:** 2021–24.
- **Check:** 2025.

**DK scoring:**
- 0.1 per rushing or receiving yard, 6 per TD, 1 per reception.
- +3 at 100 rushing yards and +3 at 100 receiving yards.
- −1 per lost fumble, +2 per two-point conversion, 6 per return TD (plus passing on trick plays).
- Check: DK − nflverse PPR averages 0.24 points per RB game (the bonuses, less the PPR fumble difference).

**Expectation** (the H-style pre-game mean, built here because the H inputs are not public):
- The player's DK mean this season before the game, shrunk toward his last-season mean with a weight of 4 games.
- Players with no last season need 2 or more prior games.
- Games with an expectation under 5 points are dropped.

**Sample:**
- **RB1** = the team RB in that game's snap counts with the highest expectation.
- 2644 RB1-games: 2130 in 2021–24 and 514 in 2025.
- **Tiers** are thirds of expectation on 2021–24: low (expectation < 12.3), mid (12.3–15.9), top (≥ 15.9; the expensive RBs) DK points.

**se** = bootstrap over season-weeks.

**Caveat on the low tier:** the low tier's mean ratio is 1.27 (0.04). These are mostly new starters whose history includes backup games, so this expectation sits under them. A projection that knows the role would not, so read the low-tier rows as an artifact of the expectation, and use the mid and top tiers for the spec.

## 1. DK actual ÷ expectation, real quantiles vs shapes fitted on the middle 80%

**Fitting:**
- **Lognormal:** least squares on log quantiles p10–p90.
- **Gamma:** least squares on quantiles p10–p90.
- se on the real quantiles = week bootstrap (300 replicates).

| Tier | Games | Mean ratio | Real sd | Quantile | Real | Lognormal | Gamma |
|---|---:|---:|---:|---|---:|---:|---:|
| all RB1s | 2130 | 1.067 (0.016) | 0.728 (0.019) | p05 | 0.200 (0.012) | 0.284 | 0.239 |
|  |  |  |  | p10 | 0.325 (0.017) | 0.365 | 0.337 |
|  |  |  |  | p25 | 0.560 (0.015) | 0.554 | 0.562 |
|  |  |  |  | p50 | 0.917 (0.021) | 0.883 | 0.917 |
|  |  |  |  | p75 | 1.400 (0.029) | 1.406 | 1.401 |
|  |  |  |  | p90 | 1.939 (0.040) | 2.138 | 1.956 |
|  |  |  |  | p95 | 2.485 (0.058) | 2.748 | 2.347 |
| low (expectation < 12.3) | 710 | 1.267 (0.040) | 0.963 (0.036) | p05 | 0.140 (0.026) | 0.245 | 0.180 |
|  |  |  |  | p10 | 0.269 (0.037) | 0.333 | 0.286 |
|  |  |  |  | p25 | 0.570 (0.034) | 0.557 | 0.559 |
|  |  |  |  | p50 | 1.045 (0.040) | 0.987 | 1.041 |
|  |  |  |  | p75 | 1.735 (0.061) | 1.749 | 1.749 |
|  |  |  |  | p90 | 2.620 (0.057) | 2.926 | 2.605 |
|  |  |  |  | p95 | 3.074 (0.103) | 3.981 | 3.224 |
| mid (12.3–15.9) | 710 | 1.006 (0.021) | 0.601 (0.019) | p05 | 0.194 (0.020) | 0.293 | 0.260 |
|  |  |  |  | p10 | 0.306 (0.030) | 0.372 | 0.356 |
|  |  |  |  | p25 | 0.580 (0.024) | 0.556 | 0.572 |
|  |  |  |  | p50 | 0.893 (0.028) | 0.869 | 0.903 |
|  |  |  |  | p75 | 1.354 (0.031) | 1.358 | 1.345 |
|  |  |  |  | p90 | 1.792 (0.061) | 2.030 | 1.846 |
|  |  |  |  | p95 | 2.071 (0.065) | 2.581 | 2.196 |
| top (≥ 15.9; the expensive RBs) | 710 | 0.927 (0.021) | 0.490 (0.015) | p05 | 0.259 (0.019) | 0.307 | 0.279 |
|  |  |  |  | p10 | 0.359 (0.019) | 0.382 | 0.368 |
|  |  |  |  | p25 | 0.544 (0.019) | 0.549 | 0.561 |
|  |  |  |  | p50 | 0.868 (0.024) | 0.821 | 0.847 |
|  |  |  |  | p75 | 1.211 (0.034) | 1.229 | 1.218 |
|  |  |  |  | p90 | 1.596 (0.053) | 1.768 | 1.632 |
|  |  |  |  | p95 | 1.826 (0.039) | 2.197 | 1.917 |

| Tier | Below the fitted lognormal p10 (par 10%) | Below its p05 (par 5%) | Above its p95 (par 5%) | Below the fitted gamma p10 | DK points ≤ 0 | Mean actual − expectation (points) |
|---|---:|---:|---:|---:|---:|---:|
| all | 12.0% (0.7) | 8.4% (0.7) | 3.1% (0.4) | 10.5% (0.7) | 0.8% (0.2) | 0.21 (0.20) |
| low | 11.5% (1.3) | 9.0% (1.1) | 1.7% (0.4) | 10.6% (1.2) | 2.1% (0.5) | 2.06 (0.34) |
| mid | 12.4% (1.3) | 9.6% (1.2) | 1.5% (0.4) | 11.5% (1.2) | 0.4% (0.2) | 0.07 (0.33) |
| top | 11.8% (1.3) | 7.3% (1.0) | 1.5% (0.5) | 10.6% (1.3) | 0.0% (0.0) | −1.49 (0.38) |

**Reading:**
- **The real low tail is heavier than the lognormal at p05 and p10.** Overall p05 is 0.20 vs 0.28 fitted, and the gap is biggest at p05.
- **The real upper tail is lighter than the lognormal.** p95 is 2.48 vs 2.75.
- **A gamma fitted the same way is closer at the bottom** (10.5% (0.7) under its p10), but still slightly short.
- **For expensive RBs (top tier):**
  - p05 is 0.26 vs 0.31.
  - 11.8% (1.3) of games fall under the lognormal p10.
  - The mean is 0.93 (0.02) × expectation (−1.49 (0.38) points), the same shortfall Shep's M found against the published projections.

## 2. What causes the low tail (games below the tier's fitted lognormal p10, 2021–24)

Causes are assigned in order, so each game counts once:
1. **Early exit:** snap share under 40% when the prior average was 60%+. The prior is this season's earlier games, else last season; it is missing for 42 games.
2. **Game script:** the team trailed by 14+ at half (from pbp).
3. **Committee:** another RB had 40%+ of the team's RB carries.
4. **Residual.**

All three causes are in-game facts, so they describe busts rather than predict them. The committee flag is partly a result of the bust itself: when RB1 fails or gets hurt, the other back takes the carries.

| Tier | Tail games | Tail rate | Early exit | Game script | Committee | Residual |
|---|---:|---:|---:|---:|---:|---:|
| all | 254 | 11.9% (0.7) | 14.2% (2.2) | 16.9% (2.3) | 44.5% (3.2) | 24.4% (2.6) |
| low | 82 | 11.5% (1.2) | 7.3% (3.1) | 14.6% (3.7) | 65.9% (5.0) | 12.2% (3.6) |
| mid | 88 | 12.4% (1.3) | 18.2% (4.1) | 18.2% (4.6) | 44.3% (5.0) | 19.3% (4.1) |
| top | 84 | 11.8% (1.4) | 16.7% (4.1) | 17.9% (4.0) | 23.8% (4.6) | 41.7% (6.0) |

**How often each flag appears, in the tail vs outside it** (not exclusive):

| Tier | Early exit: tail / outside | Game script: tail / outside | Committee: tail / outside |
|---|---:|---:|---:|
| all | 14.2% (1.9) / 1.6% (0.3) | 17.3% (2.4) / 9.7% (0.7) | 65.4% (2.7) / 25.7% (1.0) |
| low | 7.3% (2.8) / 1.1% (0.4) | 14.6% (3.2) / 10.5% (1.1) | 82.9% (4.3) / 35.8% (1.8) |
| mid | 18.2% (4.0) / 1.6% (0.5) | 18.2% (4.3) / 9.5% (1.2) | 69.3% (4.8) / 24.6% (2.0) |
| top | 16.7% (3.8) / 2.1% (0.5) | 19.0% (4.1) / 9.1% (1.2) | 44.0% (5.1) / 16.6% (1.3) |

**Reading:**
- **For expensive RBs the largest share is residual** (41.7% (6.0)): ordinary bad days with a normal workload.
- **Early exits are rare but almost always end in a bust,** about 9× as common in the tail as outside it.
- **Trailing big at half doubles in the tail but explains under a fifth of it.**

## 3. Do spread, total or underdog status shift the bust probability? Logistic fit on 2021–24, scored on 2025

Coefficients are per 7 points of team spread (positive = underdog) and per 7 points of total above 45. The held-out gain is the 2025 log-loss gain over the tier-only model; positive means better.

| Model | Coefficients (se), after intercept and tier | 2025 log-loss | 2025 gain vs tier only |
|---|---|---:|---:|
| tier only | — | 0.3562 | 0.0000 (0.0000) |
| tier + team spread | 0.170 (0.087) | 0.3582 | −0.0019 (0.0020) |
| tier + total | −0.171 (0.123) | 0.3539 | 0.0023 (0.0018) |
| tier + underdog | 0.265 (0.164) | 0.3576 | −0.0014 (0.0013) |
| tier + spread + total | 0.173 (0.080); −0.177 (0.118) | 0.3558 | 0.0004 (0.0019) |

| Group | 2021–24 games | 2021–24 tail rate | 2025 games | 2025 tail rate |
|---|---:|---:|---:|---:|
| favorite by 7+ | 294 | 10.9% (2.1) | 81 | 19.8% (3.9) |
| favorite by <7 | 776 | 10.4% (1.2) | 180 | 8.3% (1.7) |
| underdog by <7 | 780 | 13.1% (1.4) | 178 | 9.0% (1.9) |
| underdog by 7+ | 280 | 13.9% (2.0) | 75 | 16.0% (4.7) |
| total under 42 | 569 | 13.2% (1.7) | 128 | 17.2% (3.8) |
| total 42–47 | 989 | 12.1% (1.1) | 228 | 10.1% (1.8) |
| total over 47 | 572 | 10.3% (1.2) | 158 | 8.9% (2.1) |

**Reading:**
- **In 2021–24, underdogs bust slightly more** (0.17 per 7 points, se 0.09) and high totals slightly less.
- **Neither holds in 2025.** No model beats tier-only by more than about 1 se. 2025 big favorites busted more, not less.
- **The spec therefore keeps only a small spread term.** Its tilt is the fitted β below, and it is optional.

## 4. Sim-ready spec (also in `n_rb_spec.json`)

**Model:** ratio = actual ÷ expectation ~ (1 − p) · LogNormal(μ_t, σ_t) + p · Normal(m, s).
- p = logistic(a_t + β · teamSpread / 7).
- μ_t = ln((M_t − p·m) / (1 − p)) − σ_t² / 2, so the mixture mean is exactly M_t.
- The spec holds M_t = 1, because projections can't move.

**Fit:** quantile matching on 2021–24 (p02–p98, extra weight on p05, p10 and p95), by tier × favorite/underdog.

| Parameter | Mean held at 1 (the spec) | Mean free (for reference) |
|---|---:|---:|
| a_low (bust logit) → p at spread 0 | -0.304 → 42.5% | -0.674 → 33.8% |
| a_mid (bust logit) → p at spread 0 | -0.735 → 32.4% | -0.840 → 30.2% |
| a_top (bust logit) → p at spread 0 | -1.297 → 21.5% | -1.120 → 24.6% |
| β (per 7 points of team spread) | 0.104 | 0.126 |
| m (bust mean, × expectation) | 0.443 | 0.424 |
| s (bust sd, × expectation) | 0.253 | 0.244 |
| σ_low (main lognormal) | 0.374 | 0.507 |
| σ_mid (main lognormal) | 0.410 | 0.418 |
| σ_top (main lognormal) | 0.455 | 0.407 |
| M_low (mixture mean) | 1 | 1.265 |
| M_mid (mixture mean) | 1 | 1.001 |
| M_top (mixture mean) | 1 | 0.924 |
| main-part mean, low tier, spread 0 | 1.411 | 1.694 |
| main-part mean, mid tier, spread 0 | 1.267 | 1.251 |
| main-part mean, top tier, spread 0 | 1.152 | 1.087 |

**Calibration: the share of games below the spec's p10 (target 10%) and above its p95 (target 5%):**

| Tier | Spec (mean 1), 2025 floor | Spec, 2025 ceiling | Spec, 2021–24 floor | Spec, 2021–24 ceiling | Mean free, 2025 floor | Mean free, 2025 ceiling | Single lognormal (mean 1), 2025 floor | Single lognormal, 2025 ceiling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 10.1% (1.5) | 7.4% (1.0) | 10.7% (0.7) | 7.7% (0.5) | 10.1% (1.4) | 5.3% (1.1) | 9.7% (1.6) | 3.9% (0.5) |
| low | 7.4% (1.6) | 14.3% (1.8) | 9.7% (1.2) | 15.9% (1.4) | 9.1% (1.6) | 5.7% (1.5) | 5.7% (1.8) | 8.6% (1.3) |
| mid | 11.3% (2.7) | 3.3% (1.6) | 10.1% (1.2) | 4.6% (0.8) | 11.3% (2.8) | 3.3% (1.5) | 11.3% (2.7) | 1.3% (0.9) |
| top | 11.6% (2.2) | 4.2% (1.9) | 12.1% (1.4) | 2.4% (0.6) | 10.1% (2.1) | 6.3% (2.2) | 12.2% (2.2) | 1.6% (0.9) |

**Shapes for a typical RB1** (ratio quantiles, spec vs the single lognormal with mean 1):

| Tier / team spread | Bust p | Main mean | Spec p05 / p10 / p50 / p90 / p95 | Single lognormal p05 / p10 / p50 / p90 / p95 |
|---|---:|---:|---|---|
| low / spread -3 | 41.4% | 1.393 | 0.15 / 0.26 / 0.91 / 1.85 / 2.17 | 0.17 / 0.24 / 0.70 / 2.07 / 2.82 |
| low / spread +3 | 43.6% | 1.430 | 0.14 / 0.26 / 0.89 / 1.89 / 2.21 | 0.17 / 0.24 / 0.70 / 2.07 / 2.82 |
| mid / spread -3 | 31.4% | 1.256 | 0.19 / 0.32 / 0.91 / 1.78 / 2.09 | 0.27 / 0.34 / 0.80 / 1.88 / 2.39 |
| mid / spread +3 | 33.4% | 1.280 | 0.18 / 0.31 / 0.91 / 1.80 / 2.12 | 0.27 / 0.34 / 0.80 / 1.88 / 2.39 |
| top / spread -3 | 20.7% | 1.146 | 0.26 / 0.39 / 0.90 / 1.74 / 2.07 | 0.31 / 0.39 / 0.84 / 1.80 / 2.24 |
| top / spread +3 | 22.2% | 1.159 | 0.25 / 0.38 / 0.90 / 1.75 / 2.09 | 0.31 / 0.39 / 0.84 / 1.80 / 2.24 |

**Does the mean have to stay fixed, and what does it cost?** Holding M = 1 is possible, and the spec does it. To keep the mean while moving mass into the bust component, the main part has to shift up: its mean rises to 1.15 × expectation for top-tier RBs and 1.27 for mid. Against a single lognormal with the same mean, the spec raises the median (top tier, spread −3: 0.84 → 0.90) and lowers p95 (2.24 → 2.07), since the real upper tail is thinner too.

- **The floor is right:** the spec's p05/p10 come down to about the real ones.
- **The all-tier 2025 ceiling of 7.4% (1.0) comes from the low tier** (14.3% (1.8)), where this expectation sits under new starters (see the data notes).
- **For the mid and top tiers the 2025 ceiling rates are within about 1 se of 5%.**
- **On 2021–24, expensive RBs clear the spec's p95 only 2.4% (0.6) of the time.** Their real mean is 0.93 (0.02) × expectation, and a mean-1 spec overstates their upside.
- **The free-mean fit is what the data want:** M_top = 0.924. It hits both rates (10.1% (2.1) / 6.3% (2.2) on 2025).
- **So a low-side widening alone fixes the floor but not the expensive RBs' mean.** If the projections stay fixed, expect their simulated ceilings to run a little hot.

## What is and isn't proven

**Proven on 2130 RB1-games (2021–24) and checked on 514 (2025):**
- The low tail is heavier and the high tail lighter than a lognormal fitted on the middle 80%.
- A bust-mixture with the mean held at 1 reproduces the floor rate held out.
- Spread, total and underdog status don't shift busts held out.

**Not proven:**
- That this expectation behaves like the published projection. It is a stand-in for H, and the low tier shows where it differs.
- Causal readings of committee and game script (both are in-game facts).
- Anything about WR/TE.

**Reproduce:** `node scripts/unit-n.cjs <nflverse dir> <games.csv> unit-n.json n_rb_spec.json`, then `node scripts/render-result-n.cjs unit-n.json RESULT-N.md`. The nflverse files are the public release assets named above.

RESULT: RB1 busts are a separate mode (12.0% (0.7) under a middle-80% lognormal p10). The spec is a bust component at 0.44 × expectation (sd 0.25) with p = 21% (top) / 32% (mid), the mean held at 1, and the upper part lifted to 1.15 / 1.27. 2025 top-tier floor 11.6% (2.2), ceiling 4.2% (1.9). Holding the mean leaves expensive-RB ceilings slightly hot; spread and total don't predict busts.
