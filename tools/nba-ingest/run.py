#!/usr/bin/env python3
# The only script the data repo's nba-ingest workflow runs; each step declares exactly the paths it may write.
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def steps(a):
    return [
        {"script": "nba_ingest.py", "args": ["--season", a.season, "--cache", a.cache, "--out", a.out],
         "outputs": ["nba-tables", "nba-stints", "nba-events", "nba-ingest-status.json"]},
        # (Wes's profiles step, nba_profiles.py, no longer runs here. Vale 10/13 un-held player-profiles.json this way: it is
        # written by the minutes step below, from the same rows as minutes-data.json. The file stays as that step's library.)
        # Wes: the Game Recaps tab (last 60 days). Leaves recaps.json alone when no game is in the window.
        {"script": "nba_recaps.py", "args": ["--season", a.season, "--cache", a.cache, "--repo", a.out],
         "outputs": ["recaps.json"]},
        # Vale 10/12 (opening-night plan piece 2): games, lines and rosters from ESPN, and the DraftKings slate joined to ESPN
        # athlete ids by name: one file for today (US Eastern) and one for each date an NBA slate in dk-slates.json starts on.
        # --soft: a date whose requests fail is reported under "tonight" in the status file and does not lose the run's history.
        {"script": "nba_tonight.py", "args": ["--repo", a.out, "--out", a.out, "--soft"],
         "outputs": ["nba-tonight", "nba-ingest-status.json"]},
        # Vale's projection step (Monty CLEAR 10/12), unpacked from nba_projection_bundle.py and run for each date the tonight step
        # wrote that has joined slate rows. Writes nba-proj/<date>.json and its entry in the status file; never data.json (the page
        # builds that on an admin's press). Soft: a date that cannot be projected is reported under "projection", not fatal.
        # The news it was handed (the league's report, the lineups file, DraftKings' prop prices) is kept in nba-public-record/<date>/:
        # nba-record/ is the PRIVATE repo's (Stokastic's numbers live there, 10/14), and this job never writes it.
        # Orders 080/084: it also writes nba-model-inputs.json, the saved stats the page's projection step (assets/nba-model.js) reads,
        # from its own reads of the public history only, and runs the model check on every date before anything is published: on a
        # disagreement, or when it cannot say, it publishes no projection and no saved stats. Order 085: only the projections stop. It
        # writes a stop entry for the date in the status file and exits 0, so the run goes on (stats, minutes) and commits the history,
        # tonight's slate and the status as on any night; the last good nba-proj/<date>.json stays as it was.
        {"script": "nba_project_step.py", "args": ["--repo", a.out, "--cache", a.cache],
         "outputs": ["nba-proj", "nba-model-inputs.json", "nba-public-record", "nba-ingest-status.json"]},
        # Vale's stats step (10/13; Monty CLEAR at nba_stats.py 26355373c1f1 on nbalib.py 9863a35a5768), unpacked from the same
        # bundle: every number the Stats tab and the Matchups screens used to get from the Excel, one row a player by ESPN id, and
        # the substitution chain. Writes nba-stats.json and "stats" in the status file. Soft; never a thinner file.
        {"script": "nba_stats_step.py", "args": ["--season", a.season, "--repo", a.out, "--cache", a.cache],
         "outputs": ["nba-stats.json", "nba-ingest-status.json"]},
        # Vale 10/12: minutes-data.json for the slate just projected, so the page's sim has every player's game log (by ESPN id,
        # across seasons), the step's minutes and its Out rows; 10/13: player-profiles.json from the same rows, with each player's
        # substitution chain from the stats file (the chain is NEVER on the minutes file: the lineup builder reads it there).
        # Never a thinner file; both or neither; soft; reported under "minutes".
        {"script": "nba_minutes_file.py", "args": ["--season", a.season, "--repo", a.out],
         "outputs": ["minutes-data.json", "player-profiles.json", "nba-ingest-status.json"]},
        # (Vale's morning card no longer runs here: Monty's HOLD of 8caf95e, 10/14. It is graded from the night's record, which carries
        # Stokastic's numbers, so it runs in the PRIVATE repo's own workflow beside the props ledger: nba_private_record.py.)
    ]


def main(argv=None, run=subprocess.run):
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--outputs-file", required=True)
    a = ap.parse_args(argv)
    outputs = []
    for step in steps(a):
        print(f"== {step['script']}", flush=True)
        r = run([sys.executable, os.path.join(HERE, step["script"]), *step["args"]])
        if r.returncode:
            print(f"NBA ingest FAILED at {step['script']} (exit {r.returncode})", flush=True)
            return r.returncode
        outputs += [p for p in step["outputs"] if p not in outputs]
    with open(a.outputs_file, "w") as f:
        f.write("".join(p + "\n" for p in outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
