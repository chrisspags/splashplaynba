No pre-lock signal reliably moves K/DST picks held out. Fit on weeks 1–6, every signal does worse than the training mean on weeks 7–8 (held-out R² −0.56 to −0.95, 6 games). Pooled over 25 games, the side a pre-lock signal favors gets at most 57.5% (8.2) of top-1% K/DST picks (sim captain team share); 50% means no information. The spread favorite gets 42.8% (8.1). The side that actually won gets 78.6% (5.8). Favorites won only 44.0% (10.0) of these games, so the sample cannot separate "no signal" from "unlucky favorites".

# Vale unit 1103CL-G (11/3/2026): prime-time NFL Showdown, the winning-team K/DST effect

Public data only (`chrisspags/splashplaynba` `origin/main`), aggregates only: no lineups, user ids or entry ids. No site code was changed.

**Data:** the same 25 prime-time Showdowns as unit C (weeks 1–8; 5 / 8 / 12 games in the 1–3 / 4–8 / 9+ margin bands).
- **Results:** `nfl-dk-results/<date>.json`: top-1% `topFieldLineups`, `fieldOwnership`, `captainOwnership`.
- **Pre-lock:** `nfl-dk-showdown-<dg>.json`: `results.playerResults` (`cptWinnerExposure`, `flexWinnerExposure`, `exposure`) and `results.publicForecast.lines` (home spread).
- **Finals:** `nfl-schedule.json`, re-read 11/3 after main picked up the 11/2 MNF final.
- **"One score"** is read as a 1–3 point margin, since Vale lists 4–8 separately.
- **se = game bootstrap.** Each game's value is the mean over its contests.

## 1. Top-1% K/DST: how much comes from the winning team, by final margin

Share of K/DST picks that are the winning team's, top-1% lineups vs the field (ownership). "Games with any" counts games where top lineups used that slot at all; when it is 1, the se is 0 and means nothing.

| Margin | Games | Slot | Games with any | Top: on winner | Field: on winner | Top per lineup, winner | Top per lineup, loser |
|---|---:|---|---:|---:|---:|---:|---:|
| all | 25 | CPT K/DST | 10 | 91.5% (6.2) | 45.5% (4.6) | 0.034 (0.017) | 0.003 (0.002) |
| all | 25 | FLEX K/DST | 24 | 84.6% (4.5) | 50.4% (2.1) | 0.564 (0.096) | 0.103 (0.032) |
| all | 25 | FLEX K | 24 | 78.4% (6.3) | 50.3% (1.9) | 0.301 (0.057) | 0.083 (0.028) |
| all | 25 | FLEX DST | 20 | 93.0% (4.0) | 50.6% (3.1) | 0.264 (0.067) | 0.020 (0.013) |
| 1–3 | 5 | CPT K/DST | 1 | 89.4% (0.0) | 53.6% (10.5) | 0.056 (0.052) | 0.007 (0.006) |
| 1–3 | 5 | FLEX K/DST | 5 | 77.5% (6.7) | 51.3% (4.9) | 0.635 (0.162) | 0.184 (0.084) |
| 1–3 | 5 | FLEX K | 5 | 75.4% (8.4) | 50.5% (4.6) | 0.370 (0.079) | 0.121 (0.066) |
| 1–3 | 5 | FLEX DST | 4 | 80.7% (9.6) | 52.4% (6.1) | 0.265 (0.162) | 0.063 (0.057) |
| 4–8 | 8 | CPT K/DST | 5 | 89.8% (21.7) | 41.8% (9.9) | 0.051 (0.040) | 0.006 (0.003) |
| 4–8 | 8 | FLEX K/DST | 7 | 58.7% (8.2) | 48.8% (4.2) | 0.249 (0.087) | 0.175 (0.065) |
| 4–8 | 8 | FLEX K | 7 | 52.5% (12.2) | 48.5% (3.6) | 0.170 (0.075) | 0.153 (0.063) |
| 4–8 | 8 | FLEX DST | 6 | 78.3% (10.8) | 49.2% (6.8) | 0.079 (0.025) | 0.022 (0.014) |
| 9+ | 12 | CPT K/DST | 4 | 100.0% (0.0) | 43.4% (5.2) | 0.014 (0.007) | 0.000 (0.000) |
| 9+ | 12 | FLEX K/DST | 12 | 97.3% (1.7) | 51.0% (2.7) | 0.745 (0.150) | 0.021 (0.010) |
| 9+ | 12 | FLEX K | 12 | 94.7% (4.0) | 51.4% (2.2) | 0.359 (0.096) | 0.020 (0.010) |
| 9+ | 12 | FLEX DST | 10 | 99.9% (0.1) | 50.4% (4.2) | 0.386 (0.105) | 0.001 (0.000) |

**Reading:**
- **The field splits K/DST about 50/50 between the teams in every band.**
- **Top-1% lineups take them almost only from the winner in blowouts** (9+): FLEX K/DST 97.3% (1.7) on the winner.
- **In closer games the tilt is weaker:** 77.5% (6.7) in 1–3 and 58.7% (8.2) in 4–8.
- **4–8 is weakest, mostly through K:** FLEX K on the winner is 52.5% (12.2) there.
- **Captain K/DST is rare** (0.034 (0.017) per top lineup on the winner), so the captain rows rest on a few games.
- **Bands are 5, 8 and 12 games,** so differences between bands are not established.

## 2. Does a pre-lock signal pick the side whose K/DST lands in the top 1%?

**Target:** the home team's share of top-lineup K/DST picks (CPT + FLEX), per game. 24 games; in 1 game the top lineups held no K/DST.

**Signals (pre-lock, home side):**
- spread → win probability, Φ(−spread / 13.5);
- the sim's captain winner share by team (Σ `cptWinnerExposure`);
- the sim's K/DST winner share by team (CPT + FLEX);
- the sim's all-slot winner share by team (Σ `exposure`).

**References (not pre-lock):** the field's K/DST ownership split, and the actual winner.

**Held-out test:** fit target = a + b·signal on weeks 1–6 (19 games), score weeks 7–8 (6 games). R² is against the training mean; its se is a bootstrap over the 6 held-out games.

**Favored-side share:** the share of top-lineup K/DST picks on the side the signal favors (signal above 0.5). 50% = no information.

| Signal | Games | Corr. with target, all games | Held-out R² | Favored-side share, held out | Favored-side share, all games | Games where the favored side got most K/DST |
|---|---:|---:|---:|---:|---:|---:|
| spread (home win prob.) | 24 | −0.240 (0.205) | −0.65 (0.33) | 78.4% (14.5) | 42.8% (8.1) | 41.7% (10.0) |
| sim captain team share | 24 | −0.134 (0.210) | −0.56 (0.18) | 95.1% (3.1) | 57.5% (8.2) | 58.3% (10.0) |
| sim K/DST team share | 24 | −0.092 (0.201) | −0.60 (0.20) | 78.4% (14.3) | 47.0% (7.9) | 45.8% (10.2) |
| sim all-slot team share | 24 | −0.278 (0.198) | −0.95 (0.32) | 95.1% (3.1) | 47.3% (8.2) | 50.0% (10.1) |
| field K/DST ownership split (post-lock) | 24 | −0.188 (0.210) | −0.87 (0.41) | 78.4% (14.5) | 39.2% (7.7) | 41.7% (10.1) |
| actual winner (not pre-lock; ceiling) | 24 | 0.722 (0.117) | 0.74 (0.06) | 95.1% (3.2) | 78.6% (5.8) | 87.5% (6.8) |

**Context:**
- The sim's captain share and the spread agree closely (corr. 0.848 (0.068)), as do the sim's K/DST share and the spread (0.900 (0.033)).
- Spread favorites won 44.0% (10.0) of games; the sim's captain-favored side won 52.0% (10.1).

**Reading:**
- **The sim adds nothing over the spread.** Its per-team shares mostly restate the spread.
- **Neither one picked winners in this sample,** so neither picks the K/DST side.
- **The weeks 1–6 fits learned a negative slope** (favorites lost), so held out they score below the training mean.
- **The raw favored-side shares look high held out** (6 games where favorites mostly won), but pooled over all games they sit within about 1 se of 50%.
- **What would move K/DST is knowing the winner,** and nothing pre-lock here does that better than a coin flip.

## 3. Winning team's K or DST plus that team's QB or RB

**Rates:** share of lineups holding the team's K or DST and its QB or an RB (any slots).

**Field-implied rate:** from ownership, treating players as independent. The field's joint lineups aren't public, so this is a reference, not a measured field rate.

| Team | Feature | Top 1% | Field-implied | Top − field |
|---|---|---:|---:|---:|
| winner | K/DST | 49.5% (6.4) | 33.8% (1.4) | 15.7% (5.9) |
| winner | QB/RB | 98.2% (1.7) | 84.3% (1.4) | 13.9% (2.0) |
| winner | K/DST + QB/RB | 47.6% (6.2) | 28.6% (1.4) | 19.0% (5.7) |
| loser | K/DST | 10.4% (3.3) | 33.5% (2.1) | −23.1% (4.1) |
| loser | QB/RB | 78.2% (6.5) | 83.5% (1.6) | −5.2% (6.3) |
| loser | K/DST + QB/RB | 7.6% (2.3) | 28.2% (2.0) | −20.6% (3.5) |

**Reading:**
- **The pairing adds nothing beyond its two parts.** Inside top lineups the joint rate equals the product of the two marginals (winner 0.997 (0.002), loser 0.975 (0.024)), because the winner's QB or RB is in 98.2% (1.7) of top lineups anyway. The excess over the field comes from two separate winner-side tilts, larger on K/DST (15.7% (5.9)) than on QB/RB (13.9% (2.0)).
- **No extra K/DST + QB/RB pairing effect is visible** beyond picking the winning side.

## What is and isn't proven

**Proven on 25 games:**
- Top-1% K/DST comes from the winning team, overwhelmingly in 9+ point games.
- The field splits them evenly.
- Winner K/DST + winner QB/RB is 19.0% (5.7) above its field-implied rate. Within top lineups the pairing itself adds nothing; the excess is the separate K/DST and QB/RB tilts toward the winner.

**Not proven:**
- That spread or sim signals carry no K/DST information. Favorites lost more than half these games, and the held-out set is 6 games.
- Any band difference (5, 8 and 12 games).
- Anything about captain K/DST (it is in about 3% of top lineups).

**Reproduce:** `node scripts/unit-g.cjs <data dir> unit-g.json`, then `node scripts/render-result-g.cjs unit-g.json RESULT-G.md`.

RESULT: No pre-lock signal moves K/DST held out (R² below 0 on 6 games). The best pooled favored-side K/DST share is 57.5% (8.2) (sim captain team share) vs 78.6% (5.8) for the real winner. The top-1% K/DST tilt follows who wins, and neither the spread nor the sim saw it in advance in this sample (25 games).
