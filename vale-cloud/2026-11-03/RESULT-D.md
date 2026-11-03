On Classic, the sim's player winner shares are calibrated (slope 1.04, se 0.03). They rank players about as well as real field ownership does (within-slate rank correlation 0.86 vs 0.89 on Afternoon) and far better than projection (0.80). The sim misjudges QB stacks: its view of which stacks over-perform has no relation to the real result (within-slate rank correlation -0.07), and full game stacks (QB + 2 or more WR/TE + bring-back) are 36.6% (5.5) of top-1% lineups vs 23.3% (2.9) of the sim's field (Main 29.9% (7.1) vs 12.4% (1.2)). Afternoon winners are more concentrated than Main winners and use more salary than the field, so "near-max loses" does not hold across the 8 afternoon slates. These are 8 Afternoon and 8 Main slates, with only 2 held-out slates each.

# Vale unit 1103CL-D (11/3/2026): Afternoon Only and Main, what wins vs what the sim calls winners

Public data only, aggregates only: no lineups, user ids, anonIds or entry ids. No site code was changed and the pre-rewrite git objects were not used.

**Data:** `chrisspags/splashplaynba` `origin/main`.
- **Results:** `nfl-dk-results/<date>.json`, Classic contests (topFieldLineups, topThresholds, fieldOwnership).
- **Pre-lock runs:** `nfl-dk-main-<dg>.json` and `nfl-dk-afternoon-only-<dg>.json` (`results.playerResults`, `publicForecast`, `topStacks`).
- nflverse lines were not fetched; pre-lock lines come from each run's own `publicForecast`.

## Sample

- 8 Afternoon Only and 8 Main draft groups with results, weeks 1–8 (week 9 unplayed).
- Main week 8 is draft group 153728: DraftKings opened it early, and the run's own pool pointer is dated 11/1.
- 95 contests and 21991 top lineups (6 dropped for an unmatched name).
- **Top lineups:** per contest, `topFieldLineups` at or above the top-1% line, or all 250 published when the line sits deeper. In Classic contests of 25,000+ entries, "top" is therefore the top 250 (about the top 0.1–1%).
- Each slate's value is the mean over its contests.
- **se = slate bootstrap** (1,000 replicates; 8 slates per type, so read se as rough).
- **Field-implied rates** come from each contest's `fieldOwnership`. Stack shape, bring-back, games used and RB+DST pairing need the field's joint lineups, which aren't published, so they have no field column.

## 1. Anatomy of top lineups: Afternoon vs Main

| Feature | Afternoon top | Afternoon field | Afternoon top − field | Main top | Main field | Main top − field | Afternoon − Main (top) |
|---|---:|---:|---:|---:|---:|---:|---:|
| QB naked (0 same-team WR/TE) | 6.3% (1.7) | — | — | 9.2% (2.2) | — | — | −2.8% (2.7) |
| QB + 1 WR/TE | 35.2% (4.6) | — | — | 41.1% (6.4) | — | — | −5.9% (8.0) |
| QB + 2 WR/TE | 51.1% (4.6) | — | — | 45.2% (6.7) | — | — | 5.9% (8.2) |
| QB + 3+ WR/TE | 7.4% (2.2) | — | — | 4.6% (1.4) | — | — | 2.8% (2.6) |
| bring-back (opposing RB/WR/TE) | 72.8% (6.3) | — | — | 59.6% (4.9) | — | — | 13.2% (8.0) |
| games used | 3.41 (0.19) | — | — | 5.53 (0.13) | — | — | −2.12 (0.22) |
| max players from one game | 4.32 (0.20) | — | — | 3.41 (0.15) | — | — | 0.92 (0.25) |
| RB + DST same team | 46.6% (8.8) | — | — | 20.7% (3.2) | — | — | 25.9% (9.5) |
| salary left ($) | $190 (23) | $425 (33) | −$235 (18) | $114 (3) | $215 (9) | −$101 (9) | $76 (24) |
| salary left <= $200 (near-max) | 75.5% (2.9) | — | — | 85.6% (0.6) | — | — | −10.1% (2.9) |
| salary left >= $1,000 | 2.3% (0.7) | — | — | 0.7% (0.1) | — | — | 1.7% (0.7) |
| sum of field own (%) | 270 (16) | 270 (14) | 0 (9) | 125 (5) | 125 (3) | 0 (6) | 145 (17) |
| players over 30% own (count) | 3.95 (0.39) | 3.81 (0.25) | 0.14 (0.30) | 0.68 (0.10) | 0.88 (0.12) | −0.20 (0.11) | 3.27 (0.41) |
| players under 5% own (count) | 0.32 (0.11) | 0.46 (0.06) | −0.14 (0.06) | 1.65 (0.19) | 1.95 (0.07) | −0.30 (0.19) | −1.33 (0.22) |

**Definitions:**
- "QB + n WR/TE" counts same-team WR/TE with the lineup's QB.
- Bring-back = an opposing RB/WR/TE.
- Near-max = $200 or less left.

### By slate (key features, top lineups)

| Slate | Week | Date | Top lineups | Salary left, top | Salary left, field-implied | Near-max (≤$200) | Bring-back | RB + DST same team | Max players from one game |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Afternoon 153070 | 1 | 2026-09-13 | 999 | $251 | $590 | 69% | 58% | 67% | 3.72 |
| Afternoon 153431 | 2 | 2026-09-20 | 999 | $132 | $348 | 83% | 47% | 32% | 3.81 |
| Afternoon 153772 | 3 | 2026-09-27 | 500 | $138 | $347 | 84% | 57% | 20% | 4.19 |
| Afternoon 154081 | 4 | 2026-10-04 | 1131 | $173 | $384 | 76% | 63% | 40% | 4.01 |
| Afternoon 154471 | 5 | 2026-10-11 | 1130 | $102 | $382 | 87% | 98% | 13% | 5.62 |
| Afternoon 154944 | 6 | 2026-10-18 | 1123 | $211 | $417 | 67% | 87% | 91% | 4.65 |
| Afternoon 155201 | 7 | 2026-10-25 | 1107 | $191 | $355 | 74% | 80% | 40% | 4.36 |
| Afternoon 155593 | 8 | 2026-11-01 | 1107 | $321 | $580 | 64% | 92% | 70% | 4.23 |
| Main 151307 | 1 | 2026-09-13 | 1250 | $112 | $268 | 86% | 68% | 12% | 3.61 |
| Main 153428 | 2 | 2026-09-20 | 1736 | $120 | $221 | 85% | 31% | 12% | 2.84 |
| Main 153769 | 3 | 2026-09-27 | 1975 | $95 | $194 | 89% | 68% | 7% | 3.72 |
| Main 154078 | 4 | 2026-10-04 | 2076 | $120 | $195 | 84% | 66% | 24% | 3.11 |
| Main 154468 | 5 | 2026-10-11 | 1738 | $125 | $218 | 83% | 52% | 21% | 3.33 |
| Main 154941 | 6 | 2026-10-18 | 1738 | $109 | $231 | 87% | 53% | 26% | 3.11 |
| Main 155198 | 7 | 2026-10-25 | 1691 | $115 | $199 | 86% | 58% | 33% | 3.20 |
| Main 153728 | 8 | 2026-11-01 | 1691 | $116 | $192 | 86% | 82% | 30% | 4.32 |

**Reading:**
- **Afternoon top lineups are more concentrated than Main:** fewer games, more players from one game, and bring-back and RB+DST pairing more often.
- **On both slates, top lineups leave less salary than the field's average.** The 10/25 afternoon slate is no exception.
- **Top lineups match the field on total ownership.** The difference is in counts: they hold fewer sub-5% players than the field (and, on Main, fewer 30%+ players).

## 2. Sim vs real

### Players: the run's `exposure` (share of simulated winning lineups) vs the real share of top lineups

| Slates | Player-slates | Real on sim: intercept | slope | Real on field own: slope |
|---|---:|---:|---:|---:|
| all | 3249 | -0.002 (0.001) | 1.037 (0.030) | 1.004 (0.036) |
| Afternoon | 767 | -0.002 (0.004) | 1.018 (0.043) | 1.002 (0.050) |
| Main | 2482 | -0.002 (0.002) | 1.086 (0.053) | 1.009 (0.060) |

| Slates | Group | n | Real | Sim | Field own | Real − sim | Real − field |
|---|---|---:|---:|---:|---:|---:|---:|
| Afternoon | pos RB | 188 | 9.5% (1.0) | 9.7% (0.9) | 9.9% (0.9) | −0.2% (0.2) | −0.5% (0.2) |
| Afternoon | pos WR | 287 | 9.0% (0.7) | 9.1% (0.7) | 9.0% (0.6) | −0.2% (0.1) | −0.1% (0.1) |
| Afternoon | pos TE | 176 | 7.1% (0.4) | 6.6% (0.5) | 6.4% (0.5) | 0.5% (0.3) | 0.7% (0.3) |
| Afternoon | salary under $4,000 | 378 | 4.0% (0.3) | 4.5% (0.4) | 4.1% (0.3) | −0.5% (0.3) | −0.1% (0.3) |
| Afternoon | salary $4,000-$6,999 | 353 | 11.6% (1.4) | 11.3% (0.8) | 11.7% (1.0) | 0.3% (0.7) | −0.1% (0.6) |
| Afternoon | salary $7,000+ | 36 | 43.6% (4.5) | 41.6% (3.9) | 41.2% (3.4) | 2.0% (4.0) | 2.4% (3.7) |
| Main | pos RB | 606 | 3.1% (0.2) | 2.9% (0.1) | 3.1% (0.1) | 0.2% (0.1) | 0.0% (0.1) |
| Main | pos WR | 922 | 2.8% (0.1) | 2.9% (0.1) | 2.9% (0.1) | 0.0% (0.0) | 0.0% (0.0) |
| Main | pos TE | 560 | 2.0% (0.1) | 2.1% (0.1) | 1.9% (0.1) | −0.1% (0.1) | 0.1% (0.1) |
| Main | salary under $4,000 | 1198 | 1.3% (0.1) | 1.3% (0.1) | 1.2% (0.1) | 0.0% (0.1) | 0.1% (0.1) |
| Main | salary $4,000-$6,999 | 1173 | 3.5% (0.3) | 3.4% (0.1) | 3.6% (0.1) | 0.0% (0.2) | −0.1% (0.2) |
| Main | salary $7,000+ | 111 | 14.2% (1.4) | 14.1% (0.7) | 13.7% (0.8) | 0.1% (1.2) | 0.5% (1.0) |

QB and DST are left out of the bias table: every lineup holds exactly one of each, so the real, sim and field means are equal by construction.

**Within-slate rank agreement with the real top-lineup share** (mean over slates):

| Slates | Sim exposure | Real field own (not pre-lock) | Projection | Ceiling |
|---|---:|---:|---:|---:|
| Afternoon | 0.863 (0.009) | 0.887 (0.009) | 0.796 (0.010) | 0.796 (0.014) |
| Main | 0.856 (0.009) | 0.874 (0.009) | 0.763 (0.020) | 0.768 (0.019) |

### QB stacks: the run's `topStacks` team groups vs real top lineups

**What the sim publishes:** for each team, four exclusive stack shapes, with:
- the sim's field share of each shape;
- each shape's per-lineup top-10 rate. Its lift = top-10 rate ÷ (10 / contest size).

**What it's compared with:**
- the real share of top lineups using that team-shape;
- real lift = real top share ÷ sim field share.

| Slates | Groups | Rank corr., sim field share vs real top share | Rank corr., sim lift vs real lift | Log-lift slope (real on sim) |
|---|---:|---:|---:|---:|
| all | 701 | 0.414 | -0.071 | -1.31 (0.56) |
| Afternoon | 232 | 0.452 | -0.074 | -0.12 (0.56) |
| Main | 469 | 0.377 | -0.068 | -1.48 (0.73) |

| Slates | Shape | Real share of top lineups (summed over teams) | Sim field share (summed) |
|---|---|---:|---:|
| all | QB+1, no bring-back | 11.1% (1.8) | 13.8% (1.2) |
| all | QB+1 + bring-back | 22.1% (2.8) | 23.5% (1.9) |
| all | QB+2+, no bring-back | 12.8% (2.3) | 14.0% (0.7) |
| all | QB+2+ + bring-back | 36.6% (5.5) | 23.3% (2.9) |
| Afternoon | QB+1, no bring-back | 10.1% (2.7) | 10.6% (1.2) |
| Afternoon | QB+1 + bring-back | 25.0% (3.1) | 30.7% (0.5) |
| Afternoon | QB+2+, no bring-back | 15.1% (4.4) | 15.0% (0.7) |
| Afternoon | QB+2+ + bring-back | 43.3% (6.9) | 34.2% (2.2) |
| Main | QB+1, no bring-back | 12.0% (2.3) | 17.0% (1.0) |
| Main | QB+1 + bring-back | 19.2% (4.4) | 16.4% (1.2) |
| Main | QB+2+, no bring-back | 10.5% (1.6) | 13.1% (1.1) |
| Main | QB+2+ + bring-back | 29.9% (7.1) | 12.4% (1.2) |

**Reading:**
- **The sim knows which stacks will be popular** (positive rank correlation with the real top-lineup share).
- **It has no read on which stacks over-perform** (lift correlation about 0; the log-lift slope is not positive).
- **Full game stacks (QB+2+ with a bring-back) are a much larger share of top lineups than of the sim's field,** most clearly on Main.
- **That gap is measured against the sim's field, not the real field.** It means either full game stacks win more than their ownership, or the sim's field under-builds them (or both). Real field stack shares aren't public, so this can't be split.

## 3. Pre-lock predictors of top-1% inclusion; fit weeks 1–6, held out weeks 7–8

| Slates | Model (OLS on the real top-lineup share) | Train R² | Held-out R² | Held-out within-slate rank corr. | Rows (train / held out) | Held-out slates |
|---|---|---:|---:|---:|---:|---:|
| Afternoon | position only | 0.009 | -0.004 | 0.253 | 613 / 154 | 2 |
| Afternoon | position + proj + salary | 0.337 | 0.364 | 0.852 | 613 / 154 | 2 |
| Afternoon | position + sim exposure | 0.444 | 0.415 | 0.837 | 613 / 154 | 2 |
| Afternoon | position + proj + salary + sim exposure | 0.448 | 0.404 | 0.796 | 613 / 154 | 2 |
| Afternoon | all pre-lock fields | 0.457 | 0.405 | 0.782 | 613 / 154 | 2 |
| Main | position only | 0.004 | 0.005 | 0.112 | 1904 / 578 | 2 |
| Main | position + proj + salary | 0.190 | 0.173 | 0.820 | 1904 / 578 | 2 |
| Main | position + sim exposure | 0.377 | 0.254 | 0.837 | 1904 / 578 | 2 |
| Main | position + proj + salary + sim exposure | 0.380 | 0.251 | 0.715 | 1904 / 578 | 2 |
| Main | all pre-lock fields | 0.380 | 0.251 | 0.710 | 1904 / 578 | 2 |
| all | position only | 0.005 | 0.005 | 0.185 | 2517 / 732 | 4 |
| all | position + proj + salary | 0.211 | 0.202 | 0.832 | 2517 / 732 | 4 |
| all | position + sim exposure | 0.438 | 0.385 | 0.835 | 2517 / 732 | 4 |
| all | position + proj + salary + sim exposure | 0.439 | 0.382 | 0.785 | 2517 / 732 | 4 |
| all | all pre-lock fields | 0.441 | 0.382 | 0.771 | 2517 / 732 | 4 |

**Standardized slopes (per 1 sd) in the all-fields model, weeks 1–6:**
- Afternoon: proj 0.034, salary 0.012, ceiling -0.157, stdDev 0.088, exposure 0.153, implied 0.004, total 0.003, spread -0.008.
- Main: proj 0.001, salary -0.005, ceiling -0.002, stdDev -0.003, exposure 0.063, implied 0.000, total 0.001, spread 0.000.

**Reading:**
- **On both slates the sim's exposure is the strongest pre-lock field.** It beats projection + salary on held-out R².
- **On Afternoon the other fields carry more weight than on Main,** with a negative ceiling slope next to exposure. Ceiling, sd and exposure are strongly collinear, so this is not an established difference.
- **The held-out check is 2 slates per type,** which is too few to call Afternoon different from Main.

## What is and isn't proven

**Proven on 16 slates:**
- Player winner shares are calibrated.
- The sim ranks players about as well as real ownership and better than projection.
- Top lineups use more salary than the field on both slate types.
- Full game stacks are a larger share of top lineups than of the sim's field (36.6% (5.5) vs 23.3% (2.9) overall; Main 29.9% (7.1) vs 12.4% (1.2)).

**Not proven:**
- That the sim's stack lift is backwards. The log-lift slope is negative (Main -1.48, se 0.73), but the rank correlation is about −0.07, so the safe reading is "no signal", not "inverse signal".
- Whether full game stacks beat their real ownership, or the sim's field simply under-builds them.
- That Afternoon needs a different model from Main (2 held-out slates each).
- Any field-level comparison of stack shape, bring-back or RB+DST pairing (the field's joint lineups aren't public).

**Reproduce:** `node scripts/unit-d.cjs <data dir> unit-d.json`, then `node scripts/render-result-d.cjs unit-d.json RESULT-D.md`.

RESULT: Classic player winner shares are calibrated and rank about as well as real ownership, but the sim can't tell which QB stacks over-perform. Full game stacks are 36.6% (5.5) of top-1% lineups vs 23.3% (2.9) in the sim field. Afternoon winners are more concentrated and use more salary than the field (8+8 slates, 2 held-out slates each).
