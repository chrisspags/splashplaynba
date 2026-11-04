Not answerable yet: no NBA night has both a pre-lock published run and `dk-results` standings for the same slate. The public results hold only the Main Classic contests (one draft group per night, 2026-10-20 to 2026-11-02). The only committed run for one of those draft groups is the 2026-11-02 Main run (`nba-dk-main.json`, slate 156054). It was saved at 20:25 ET, after its 7:00 PM lock, and carries in-game fields (`liveProj`, `liveFinal`), so it fails unit F's pre-lock rule. The first qualifying night will be 11/4 (Main slate 156078, pre-lock versions already on main) once `dk-results/2026-11-04.json` lands. The leave-one-night-out step needs 3 such nights. The script is written and checked.

# Vale unit 1103CL-L (11/4/2026): does the published NBA sim aim at the ownership band that wins?

Public data only (`chrisspags/splashplaynba` `origin/main`, read with `git show`). No numbers are reported, because no night qualifies. No site code was changed, and the branch was left as it is.

## What was checked

**Results nights:** `dk-results/<night>.json` for 2026-10-20, 2026-10-21, 2026-10-22, 2026-10-23, 2026-10-24, 2026-10-25, 2026-10-26, 2026-10-27, 2026-10-28, 2026-10-29, 2026-10-30, 2026-10-31, 2026-11-02. Each holds Classic contests on a single Main draft group; none holds Showdown or Turbo contests.

**Runs:** every committed version on `origin/main` of `nba-dk-main.json`, `nba-dk-night.json`, `nba-dk-early.json`, `nba-dk-late-night.json` and `nba-dk-turbo.json`.
- 225 file-versions were read.
- 6 have a `slateId` equal to a results draft group. All are the 2026-11-02 Main run.
- 6 of those 6 fail the pre-lock rule (`simUpdated` after the lock in `slateLabel`), and 6 carry `liveProj` / `liveFinal`.
- Main's history starts at the 11/2 rewrite, so no earlier Main versions are public.

**Why the Showdown and Turbo runs can't stand in:**
- Their slates have no public results.
- The ownership bands must come from the same contest's `fieldOwnership` as the top-1% lineups.
- A Turbo or Showdown winner share set against Main-contest bands would compare two different contests.

## When it can run

- **11/4 is the first night that can qualify.** `nba-dk-main.json` already has pre-lock versions for Main slate 156078 ("Main: 11/4 7:00PM Eastern"). The script takes the latest version saved before lock with no in-game fields.
- **Each later Main night adds one night,** as long as a pre-lock Main run is committed to `main` that day and results land the next morning.
- **Steps 1, 2 and 4 can run from one night.** Step 3 (leave-one-night-out) needs at least 3 qualifying nights, so the earliest full answer is after the 11/6 results, if 11/5 and 11/6 also get pre-lock Main runs.

## The script, so the design can be checked now (`scripts/unit-l.cjs`)

**Bands:** under 5%, 5–15%, 15–30% and 30%+, from each contest's `fieldOwnership`.

**Step 1, slot share by band:**
- **Sim:** Σ `simExp` (the NBA runs' winner share; equal to `winPct`) over the band's players, divided by Σ `simExp` (= 8).
- **Real:** the share of top-1% lineup slots in the band (same top-1% rule as unit I).
- **Output:** sim − real by band, pooled, with a night-bootstrap se.

**Step 2, excess points by band:**
- **Sim:** Σ over the band's players of (8 · winner share − field share) × pre-lock `proj`.
- **Real:** top-1% points from the band − the ownership-weighted field's points from the band (unit I §4).
- **Caveat:** the sim side uses mean projections, not points conditional on winning, because the runs don't publish those. So the sim's excess will read low in every band; the comparison is between bands.

**Step 3:** per night and band, real excess points regressed on the sim's tilt (sim slot share − field slot share), leave-one-night-out, held-out R² with a night bootstrap.

**Step 4:** the band with the largest |sim − real|, in slot share and in points.

**Mechanics check:** the whole pipeline was run once with `--dry-run-postlock`, which admits the post-lock 2026-11-02 run. It covered 4 contests; band shares sum to 1 on the sim, real and field sides, and step 3 correctly reports too few nights. Those numbers are not reported, because that run fails the pre-lock rule and carries in-game fields.

**Reproduce:** `node scripts/unit-l.cjs <repo dir> unit-l.json`. It reads `origin/main` with `git` (read-only), so a fresh fetch picks up new nights.

RESULT: blocked. No NBA night yet has a pre-lock published run and results for the same slate (the only match, 2026-11-02 Main, was saved after lock with in-game fields). The first qualifying night is 11/4 once its results land; the script is ready.
