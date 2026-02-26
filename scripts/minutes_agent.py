#!/usr/bin/env python3
"""
NBA Minutes Agent — GitHub Actions Pipeline
Fetches NBA game logs, box scores, and substitution data, sends to Claude Haiku
for contextual minutes projections, and writes minutes-data.json to the repo.

Converted from nba-minutes-agent.ipynb (Cells 1-4) for automated daily runs.
"""

import json
import os
import re
import base64
import time
import unicodedata
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

from nba_api.stats.static import players as nba_players_static
from nba_api.stats.static import teams as nba_teams_static
from nba_api.stats.endpoints import (
    playergamelog,
    boxscoretraditionalv3,
    leaguegamefinder,
    commonallplayers,
    playbyplayv3,
)

import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────
SEASON = "2025-26"
DELAY = 0.7
MAX_RETRIES = 3
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
REPO_OWNER = "chrisspags"
REPO_NAME = "splashplaynba"
CACHE_DIR = ".nba-cache"
CACHE_FILE = os.path.join(CACHE_DIR, f"nba_minutes_cache_{SEASON}.json")
OUTPUT_FILE = "minutes-data.json"

# ── DraftKings team abbreviation mapping ──────────────────────────────────────
DK_TO_NBA = {"GS": "GSW", "SA": "SAS", "NY": "NYK", "NO": "NOP", "CHA": "CHO"}
NBA_TO_DK = {v: k for k, v in DK_TO_NBA.items()}


def dk_to_nba_team(dk_abbr):
    upper = dk_abbr.upper()
    return DK_TO_NBA.get(upper, upper)


def nba_to_dk_team(nba_abbr):
    return NBA_TO_DK.get(nba_abbr, nba_abbr)


# ── Known DraftKings name → NBA API player ID overrides ───────────────────────
DK_ID_OVERRIDES = {
    "gregory jackson": 1641713,
    "gregory jackson ii": 1641713,
    "gg jackson": 1641713,
    "gg jackson ii": 1641713,
    "jimmy butler": 202710,
    "jimmy butler iii": 202710,
    "nicolas claxton": 203998,
    "nic claxton": 203998,
    "alexandre sarr": 1642259,
    "alex sarr": 1642259,
    "kenyon martin": 1630231,
    "kenyon martin jr": 1630231,
    "cameron thomas": 1630560,
    "cam thomas": 1630560,
    "ej harkless": 1641989,
    "elijah harkless": 1641989,
    "hansen yang": 1642905,
    "yang hansen": 1642905,
    "ronald holland": 1642267,
    "ronald holland ii": 1642267,
    "ron holland": 1642267,
    "daron holmes": 1642270,
    "daron holmes ii": 1642270,
    "lj cryer": 1642903,
    "l.j. cryer": 1642903,
    "carlton carrington": 1642257,
    "bub carrington": 1642257,
    "trentyn flowers": 1642928,
    "jazian gortman": 1642938,
    "kj martin": 1630231,
    "kj martin jr": 1630231,
}


# ══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_name(name):
    """Strip diacritics, remove periods/apostrophes, normalize whitespace."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(ascii_name.split()).strip()


def parse_v3_minutes(mins_raw):
    """Parse V3 minutes: 'PT30M45.00S' (ISO 8601), 'MM:SS', or number."""
    mins_str = str(mins_raw or "0")
    if mins_str.startswith("PT"):
        m_match = re.search(r"(\d+)M", mins_str)
        s_match = re.search(r"([\d.]+)S", mins_str)
        return (int(m_match.group(1)) if m_match else 0) + (
            float(s_match.group(1)) / 60 if s_match else 0
        )
    elif ":" in mins_str:
        parts = mins_str.split(":")
        return int(parts[0]) + int(parts[1]) / 60
    else:
        try:
            return float(mins_str)
        except ValueError:
            return 0


def parse_minutes(min_val):
    """Parse NBA API minutes format (could be 'MM:SS' or float)."""
    if pd.isna(min_val) or min_val is None:
        return 0.0
    s = str(min_val).strip()
    if ":" in s:
        parts = s.split(":")
        return int(parts[0]) + int(parts[1]) / 60
    try:
        return float(s)
    except ValueError:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD SLATE FROM GITHUB
# ══════════════════════════════════════════════════════════════════════════════

def load_slate():
    """Read data.json from GitHub, extract slate players and games."""
    print("Loading slate from GitHub data.json...")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/data.json"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    gh_data = resp.json()
    data_json = json.loads(base64.b64decode(gh_data["content"]))

    slate_players = data_json.get("myProj", [])
    games = data_json.get("games", [])

    teams_playing = set()
    for g in games:
        if isinstance(g, dict):
            if "away" in g and isinstance(g["away"], dict):
                teams_playing.add(g["away"].get("team", ""))
            if "home" in g and isinstance(g["home"], dict):
                teams_playing.add(g["home"].get("team", ""))
            if "awayTeam" in g:
                teams_playing.add(g["awayTeam"])
            if "homeTeam" in g:
                teams_playing.add(g["homeTeam"])
    teams_playing.discard("")

    # Build matchup map: team -> opponent
    matchups = {}
    for g in games:
        if isinstance(g, dict):
            away = ""
            home = ""
            if "away" in g and isinstance(g["away"], dict):
                away = g["away"].get("team", "")
            elif "awayTeam" in g:
                away = g["awayTeam"]
            if "home" in g and isinstance(g["home"], dict):
                home = g["home"].get("team", "")
            elif "homeTeam" in g:
                home = g["homeTeam"]
            if away and home:
                matchups[away] = home
                matchups[home] = away

    print(f"  Loaded {len(slate_players)} players, {len(teams_playing)} teams")
    return slate_players, games, teams_playing, matchups


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: FETCH NBA DATA (game logs, box scores, PBP)
# ══════════════════════════════════════════════════════════════════════════════

def build_player_name_index():
    """Build player name -> NBA API ID index."""
    all_nba_players = nba_players_static.get_players()
    player_name_to_id = {}
    for p in all_nba_players:
        full_norm = normalize_name(p["full_name"]).lower()
        last_norm = normalize_name(p["last_name"]).lower()
        player_name_to_id[full_norm] = p["id"]
        player_name_to_id[last_norm] = p["id"]

    # Supplement with live current-season roster
    try:
        cap = commonallplayers.CommonAllPlayers(
            is_only_current_season=1, league_id="00", season=SEASON
        )
        time.sleep(DELAY)
        cap_df = cap.get_data_frames()[0]
        added = 0
        for _, row in cap_df.iterrows():
            full = str(row.get("DISPLAY_FIRST_LAST", "")).strip()
            pid = row.get("PERSON_ID", 0)
            if full and pid:
                full_norm = normalize_name(full).lower()
                if full_norm not in player_name_to_id:
                    player_name_to_id[full_norm] = pid
                    added += 1
        if added:
            print(f"  Added {added} players from live roster")
    except Exception as e:
        print(f"  Live roster lookup skipped ({e})")

    return player_name_to_id, all_nba_players


def find_player_id(name, player_name_to_id, all_nba_players):
    """Find NBA API player ID from a DraftKings name string."""
    name_norm = normalize_name(name).lower()
    if name_norm in DK_ID_OVERRIDES:
        return DK_ID_OVERRIDES[name_norm]
    if name_norm in player_name_to_id:
        return player_name_to_id[name_norm]
    for key, pid in player_name_to_id.items():
        if name_norm in key or key in name_norm:
            return pid
    parts = name_norm.split()
    if len(parts) >= 2:
        for p in all_nba_players:
            last_norm = normalize_name(p["last_name"]).lower()
            first_norm = normalize_name(p["first_name"]).lower()
            if last_norm == parts[-1] and first_norm and first_norm.startswith(parts[0][0]):
                return p["id"]
    return None


def fetch_box_score_v3(game_id):
    """Fetch a single box score from V3 API, return normalized DataFrame."""
    bs = boxscoretraditionalv3.BoxScoreTraditionalV3(
        game_id=game_id,
        start_period=0, end_period=14,
        start_range=0, end_range=0, range_type=0,
    )
    df = bs.get_data_frames()[0]
    normalized = []
    for _, row in df.iterrows():
        first = str(row.get("firstName", "")).strip()
        last = str(row.get("familyName", "")).strip()
        pname = f"{first} {last}".strip()
        tid = row.get("teamId", 0)
        mins = parse_v3_minutes(row.get("minutes", "0"))
        started = bool(str(row.get("position", "")).strip())
        normalized.append({
            "PLAYER_NAME": pname,
            "TEAM_ID": tid,
            "MIN_FLOAT": mins,
            "STARTED": started,
        })
    return pd.DataFrame(normalized)


def fetch_box_score_with_retry(game_id):
    """Fetch a box score, retrying with exponential backoff on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            result = fetch_box_score_v3(game_id)
            return result, None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                backoff = DELAY * (2 ** (attempt + 1))
                time.sleep(backoff)
            else:
                return None, str(e)[:80]
    return None, "max retries exceeded"


def fetch_sub_pairs(game_id):
    """Fetch PBP V3 for a game, extract substitution events."""
    try:
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)

        df = None
        try:
            df = pbp.play_by_play.get_data_frame()
        except (AttributeError, Exception):
            try:
                frames = pbp.get_data_frames()
                if frames and len(frames) > 0:
                    df = frames[0]
            except Exception:
                pass

        if df is None or len(df) == 0:
            return []

        col_map = {c.lower(): c for c in df.columns}
        action_col = col_map.get("actiontype")
        sub_type_col = col_map.get("subtype")
        desc_col = col_map.get("description")
        player_col = col_map.get("playername") or col_map.get("player_name")
        person_col = col_map.get("personid") or col_map.get("player_id")
        team_col = col_map.get("teamid") or col_map.get("team_id")
        period_col = col_map.get("period")
        clock_col = col_map.get("clock") or col_map.get("pctimestring")

        if not action_col:
            return []

        subs = df[df[action_col].astype(str).str.lower() == "substitution"].copy()
        if len(subs) == 0:
            return []

        # Method 1: subType column
        subs_out = pd.DataFrame()
        subs_in = pd.DataFrame()
        if sub_type_col:
            subs_out = subs[subs[sub_type_col].astype(str).str.lower() == "out"].copy()
            subs_in = subs[subs[sub_type_col].astype(str).str.lower() == "in"].copy()

        # Method 2: Parse "SUB: InPlayer FOR OutPlayer" from description
        if (len(subs_out) == 0 or len(subs_in) == 0) and desc_col:
            direct_pairs = []
            for _, row in subs.iterrows():
                desc = str(row[desc_col]) if desc_col else ""
                match = re.match(r"SUB:\s*(.+?)\s+FOR\s+(.+)", desc, re.IGNORECASE)
                if match:
                    in_name = match.group(1).strip()
                    out_name = match.group(2).strip()
                    period = int(row[period_col]) if period_col else 0
                    team_id = int(row[team_col]) if team_col else 0
                    direct_pairs.append({
                        "out_name": out_name,
                        "in_name": in_name,
                        "out_id": 0,
                        "in_id": 0,
                        "period": period,
                        "team_id": team_id,
                    })
            if direct_pairs:
                return direct_pairs

        if len(subs_out) == 0 or len(subs_in) == 0:
            return []

        # Pair OUT/IN by (period, clock, teamId)
        pairs = []
        out_groups = defaultdict(list)
        for _, row in subs_out.iterrows():
            p = row[period_col] if period_col else 0
            c = str(row[clock_col] if clock_col else "")
            t = int(row[team_col]) if team_col else 0
            out_groups[(p, c, t)].append(row)

        in_groups = defaultdict(list)
        for _, row in subs_in.iterrows():
            p = row[period_col] if period_col else 0
            c = str(row[clock_col] if clock_col else "")
            t = int(row[team_col]) if team_col else 0
            in_groups[(p, c, t)].append(row)

        for key in out_groups:
            outs = out_groups[key]
            ins = in_groups.get(key, [])
            for i in range(min(len(outs), len(ins))):
                out_row = outs[i]
                in_row = ins[i]
                out_name = str(out_row[player_col] if player_col else "").strip()
                in_name = str(in_row[player_col] if player_col else "").strip()
                out_id = out_row[person_col] if person_col else 0
                in_id = in_row[person_col] if person_col else 0
                period = key[0]
                team_id = key[2]
                if out_name and in_name:
                    pairs.append({
                        "out_name": out_name,
                        "in_name": in_name,
                        "out_id": int(out_id) if out_id else 0,
                        "in_id": int(in_id) if in_id else 0,
                        "period": int(period) if period else 0,
                        "team_id": team_id,
                    })
        return pairs
    except Exception:
        return []


def compute_rotation_players(box_scores_dict):
    """Identify rotation players (avg >= 15 min) per team."""
    team_rotation = defaultdict(lambda: defaultdict(list))
    for gid, bs_df in box_scores_dict.items():
        for _, row in bs_df.iterrows():
            pname = row["PLAYER_NAME"]
            team_id = row["TEAM_ID"]
            mins = row["MIN_FLOAT"]
            team_abbr = team_id_to_abbr.get(team_id, "")
            if pname and team_abbr:
                team_rotation[team_abbr][pname].append(mins)
    result = {}
    for team, players_dict in team_rotation.items():
        result[team] = set()
        for pname, mins_list in players_dict.items():
            avg = sum(mins_list) / len(mins_list) if mins_list else 0
            if avg >= 15:
                result[team].add(pname)
    return result


def load_cache():
    """Load cache from JSON file."""
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Cache load failed: {e}")
        return None


def save_cache(player_game_logs_ser, player_id_map, box_scores_ser,
               sub_pairs_ser, started_data):
    """Save cache as JSON (serialized DataFrames as list-of-dicts)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    data = {
        "player_game_logs": player_game_logs_ser,
        "player_id_map": player_id_map,
        "box_scores": box_scores_ser,
        "sub_pairs": sub_pairs_ser,
        "started_data": started_data,
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
    }
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f)
    print(f"  Cache saved ({len(box_scores_ser)} box scores)")


def serialize_game_logs(player_game_logs):
    """Convert DataFrames to serializable dicts."""
    out = {}
    for name, df in player_game_logs.items():
        out[name] = df.to_dict(orient="records")
    return out


def deserialize_game_logs(ser):
    """Convert serialized dicts back to DataFrames."""
    out = {}
    for name, records in ser.items():
        out[name] = pd.DataFrame(records)
    return out


def serialize_box_scores(box_scores):
    """Convert box score DataFrames to serializable dicts."""
    out = {}
    for gid, df in box_scores.items():
        out[gid] = df.to_dict(orient="records")
    return out


def deserialize_box_scores(ser):
    """Convert serialized box scores back to DataFrames."""
    out = {}
    for gid, records in ser.items():
        out[gid] = pd.DataFrame(records)
    return out


def fetch_nba_data(slate_players, teams_playing):
    """Fetch game logs, box scores, and PBP substitution data."""
    global team_abbr_to_id, team_id_to_abbr

    nba_teams = nba_teams_static.get_teams()
    team_abbr_to_id = {t["abbreviation"]: t["id"] for t in nba_teams}
    team_id_to_abbr = {t["id"]: t["abbreviation"] for t in nba_teams}

    player_name_to_id, all_nba_players = build_player_name_index()

    # Initialize data structures
    player_game_logs = {}
    player_id_map = {}
    box_scores = {}
    sub_pairs = {}
    started_data = {}  # {game_id: {player_name: bool}}

    today_str = datetime.now().strftime("%Y-%m-%d")

    # Try loading cache
    cache = load_cache()
    cache_mode = "full"
    if cache:
        last_updated = cache.get("last_updated", "")
        if last_updated == today_str:
            cache_mode = "cached"
        else:
            cache_mode = "incremental"
        print(f"  Cache mode: {cache_mode} (last updated: {last_updated})")

    if cache_mode == "cached":
        player_game_logs = deserialize_game_logs(cache.get("player_game_logs", {}))
        player_id_map = cache.get("player_id_map", {})
        box_scores = deserialize_box_scores(cache.get("box_scores", {}))
        sub_pairs = cache.get("sub_pairs", {})
        started_data = cache.get("started_data", {})
        print(f"  Loaded from cache: {len(player_game_logs)} logs, {len(box_scores)} box scores")

        # Check for new slate players not in cache
        new_slate = [sp for sp in slate_players if sp.get("name", "") and sp["name"] not in player_game_logs]
        if new_slate:
            print(f"  {len(new_slate)} new slate players — fetching...")
            for sp in new_slate:
                nm = sp.get("name", "")
                pid = find_player_id(nm, player_name_to_id, all_nba_players)
                if not pid:
                    continue
                player_id_map[nm] = pid
                for attempt in range(3):
                    try:
                        log = playergamelog.PlayerGameLog(
                            player_id=pid, season=SEASON, season_type_all_star="Regular Season"
                        )
                        df = log.get_data_frames()[0]
                        if len(df) > 0:
                            player_game_logs[nm] = df
                            break
                        elif attempt < 2:
                            time.sleep(DELAY * (attempt + 2))
                    except Exception:
                        if attempt < 2:
                            time.sleep(DELAY * (attempt + 2))
                time.sleep(DELAY)

    elif cache_mode == "incremental":
        player_game_logs = deserialize_game_logs(cache.get("player_game_logs", {}))
        player_id_map = cache.get("player_id_map", {})
        box_scores = deserialize_box_scores(cache.get("box_scores", {}))
        sub_pairs = cache.get("sub_pairs", {})
        started_data = cache.get("started_data", {})

        last_updated = cache["last_updated"]
        last_dt = datetime.strptime(last_updated, "%Y-%m-%d")
        day_after = last_dt + timedelta(days=1)
        date_from = day_after.strftime("%m/%d/%Y")
        date_to = datetime.now().strftime("%m/%d/%Y")

        teams_that_played = set()
        try:
            print(f"  Checking schedule {day_after.strftime('%m/%d')} -> {datetime.now().strftime('%m/%d')}...")
            finder = leaguegamefinder.LeagueGameFinder(
                league_id_nullable="00",
                season_nullable=SEASON,
                season_type_nullable="Regular Season",
                date_from_nullable=date_from,
                date_to_nullable=date_to,
            )
            time.sleep(DELAY)
            games_df = finder.get_data_frames()[0]
            if len(games_df) > 0:
                teams_that_played = set(games_df["TEAM_ABBREVIATION"].unique())
            print(f"  Teams that played: {', '.join(sorted(teams_that_played)) or 'none'}")
        except Exception as e:
            print(f"  Schedule lookup failed ({e}) — refreshing all")
            teams_that_played = None

        players_to_fetch = []
        for sp in slate_players:
            name = sp.get("name", "")
            team = sp.get("team", "")
            if not name:
                continue
            if teams_that_played is None or name not in player_game_logs:
                players_to_fetch.append(sp)
            elif team.upper() in {t.upper() for t in teams_that_played}:
                players_to_fetch.append(sp)

        print(f"  Fetching {len(players_to_fetch)} player game logs...")
        for i, sp in enumerate(players_to_fetch):
            name = sp.get("name", "")
            pid = find_player_id(name, player_name_to_id, all_nba_players)
            if not pid:
                continue
            player_id_map[name] = pid
            for attempt in range(3):
                try:
                    log = playergamelog.PlayerGameLog(
                        player_id=pid, season=SEASON, season_type_all_star="Regular Season"
                    )
                    df = log.get_data_frames()[0]
                    if len(df) > 0:
                        player_game_logs[name] = df
                        break
                    elif attempt < 2:
                        time.sleep(DELAY * (attempt + 2))
                except Exception:
                    if attempt < 2:
                        time.sleep(DELAY * (attempt + 2))
            if (i + 1) % 10 == 0:
                print(f"    Fetched {i + 1}/{len(players_to_fetch)}...")
            time.sleep(DELAY)

    else:  # full fetch
        print("  Full fetch from NBA API...")
        for i, sp in enumerate(slate_players):
            name = sp.get("name", "")
            if not name:
                continue
            pid = find_player_id(name, player_name_to_id, all_nba_players)
            if not pid:
                continue
            player_id_map[name] = pid
            for attempt in range(3):
                try:
                    log = playergamelog.PlayerGameLog(
                        player_id=pid, season=SEASON, season_type_all_star="Regular Season"
                    )
                    df = log.get_data_frames()[0]
                    if len(df) > 0:
                        player_game_logs[name] = df
                        break
                    elif attempt < 2:
                        time.sleep(DELAY * (attempt + 2))
                except Exception:
                    if attempt < 2:
                        time.sleep(DELAY * (attempt + 2))
            if (i + 1) % 10 == 0:
                print(f"    Fetched {i + 1}/{len(slate_players)}...")
            time.sleep(DELAY)
        print(f"  Fetched game logs for {len(player_game_logs)} players")

    # Fetch box scores for all games referenced in game logs
    all_game_ids = set()
    for name, df in player_game_logs.items():
        all_game_ids.update(df["Game_ID"].astype(str).tolist())

    missing_box_ids = all_game_ids - set(box_scores.keys())
    if missing_box_ids:
        print(f"  Fetching {len(missing_box_ids)} box scores...")
        consecutive_failures = 0
        for i, gid in enumerate(sorted(missing_box_ids)):
            result, err = fetch_box_score_with_retry(gid)
            if result is not None:
                box_scores[gid] = result
                consecutive_failures = 0

                # Extract started data
                for _, row in result.iterrows():
                    pname = row["PLAYER_NAME"]
                    if gid not in started_data:
                        started_data[gid] = {}
                    started_data[gid][pname] = row["STARTED"]

                # Co-fetch PBP
                if gid not in sub_pairs:
                    time.sleep(DELAY)
                    pairs = fetch_sub_pairs(gid)
                    sub_pairs[gid] = pairs
            else:
                consecutive_failures += 1

            if (i + 1) % 20 == 0:
                print(f"    Fetched {i + 1}/{len(missing_box_ids)} box scores...")
            if (i + 1) % 50 == 0:
                save_cache(
                    serialize_game_logs(player_game_logs),
                    player_id_map,
                    serialize_box_scores(box_scores),
                    sub_pairs,
                    started_data,
                )

            if consecutive_failures >= 15:
                time.sleep(6.0)
            elif consecutive_failures >= 5:
                time.sleep(3.0)
            else:
                time.sleep(DELAY)
    else:
        # Extract started data from existing box scores
        for gid, bs_df in box_scores.items():
            if gid not in started_data:
                started_data[gid] = {}
                for _, row in bs_df.iterrows():
                    started_data[gid][row["PLAYER_NAME"]] = row["STARTED"]

    print(f"  Total: {len(box_scores)} box scores, {len(sub_pairs)} PBP games")

    # Save cache
    save_cache(
        serialize_game_logs(player_game_logs),
        player_id_map,
        serialize_box_scores(box_scores),
        sub_pairs,
        started_data,
    )

    return player_game_logs, player_id_map, box_scores, sub_pairs, started_data


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: BUILD PLAYER PROFILES (L5/L10, last 10 games, sub matrix)
# ══════════════════════════════════════════════════════════════════════════════

def build_replacement_matrix(sub_pairs_dict):
    """Build substitution replacement matrix from PBP data."""
    team_pair_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    team_out_totals = defaultdict(lambda: defaultdict(int))
    for gid, pairs in sub_pairs_dict.items():
        for pair in pairs:
            team_id = pair.get("team_id", 0)
            team_abbr = team_id_to_abbr.get(team_id, "")
            out_name = pair.get("out_name", "")
            in_name = pair.get("in_name", "")
            if team_abbr and out_name and in_name:
                team_pair_counts[team_abbr][out_name][in_name] += 1
                team_out_totals[team_abbr][out_name] += 1
    matrix = {}
    for team in team_pair_counts:
        matrix[team] = {}
        for out_name in team_pair_counts[team]:
            total_out = team_out_totals[team][out_name]
            if total_out == 0:
                continue
            matrix[team][out_name] = {}
            for in_name, count in team_pair_counts[team][out_name].items():
                matrix[team][out_name][in_name] = {
                    "count": count,
                    "pct": round(count / total_out, 3),
                }
    return matrix


def build_player_profiles(slate_players, player_game_logs, box_scores,
                          sub_pairs, started_data, matchups):
    """Build structured player profiles with L5/L10 and last 10 game data."""
    # Build team game timeline
    gid_dates = {}
    for _name, _gl in player_game_logs.items():
        for _, _row in _gl.iterrows():
            _gid = str(_row["Game_ID"])
            if _gid not in gid_dates:
                try:
                    gid_dates[_gid] = pd.to_datetime(_row["GAME_DATE"])
                except Exception:
                    pass

    team_games = defaultdict(list)
    for _gid, _bs in box_scores.items():
        if _gid not in gid_dates:
            _gid_key = str(_gid) if str(_gid) in gid_dates else _gid
            if _gid_key not in gid_dates:
                continue
            _date = gid_dates[_gid_key]
        else:
            _date = gid_dates[_gid]
        _teams_seen = set()
        for _, _row in _bs.iterrows():
            _ta = team_id_to_abbr.get(_row["TEAM_ID"], "")
            if _ta and _ta not in _teams_seen:
                _teams_seen.add(_ta)
                team_games[_ta].append((_date, _gid))

    for _ta in team_games:
        team_games[_ta].sort(key=lambda x: x[0], reverse=True)

    # Build box score name indexes
    bs_full_index = defaultdict(lambda: defaultdict(float))
    bs_index = defaultdict(lambda: defaultdict(float))
    bs_started_index = defaultdict(lambda: defaultdict(bool))

    for _gid, _bs in box_scores.items():
        for _, _row in _bs.iterrows():
            _ta = team_id_to_abbr.get(_row["TEAM_ID"], "")
            _pname = _row["PLAYER_NAME"]
            _mins = _row["MIN_FLOAT"]
            _started = _row["STARTED"]
            if _ta and _pname:
                _pname_norm = normalize_name(_pname).lower()
                _last = _pname_norm.split()[-1] if _pname_norm.split() else ""
                bs_full_index[(_ta, _pname_norm)][_gid] = _mins
                bs_index[(_ta, _last)][_gid] = _mins
                bs_started_index[(_ta, _pname_norm)][_gid] = _started

    # Build replacement matrix
    replacement_matrix = build_replacement_matrix(sub_pairs) if sub_pairs else {}

    # Build per-player profiles
    player_profiles = []
    for sp in slate_players:
        name = sp.get("name", "")
        team = sp.get("team", "")
        stok_min = sp.get("min", 0) or 0
        pos = sp.get("pos", "")
        price = sp.get("price", 0) or 0
        if not name:
            continue

        nba_tm = dk_to_nba_team(team)
        recent_10 = team_games.get(nba_tm, [])[:10]

        # Build last-10 game data
        last_10_games = []
        name_norm = normalize_name(name).lower()
        last_name = name_norm.split()[-1] if name_norm.split() else ""

        lk = bs_full_index.get((nba_tm, name_norm), None)
        if not lk:
            lk = bs_index.get((nba_tm, last_name), {})
        started_lk = bs_started_index.get((nba_tm, name_norm), {})

        if recent_10:
            for date_obj, gid in recent_10:
                mins = round(lk.get(gid, 0.0), 1)
                started = started_lk.get(gid, False)
                # Also check started_data dict
                if not started and gid in started_data:
                    for pn, s in started_data[gid].items():
                        if normalize_name(pn).lower() == name_norm:
                            started = s
                            break
                date_str = date_obj.strftime("%Y-%m-%d") if hasattr(date_obj, "strftime") else str(date_obj)[:10]
                last_10_games.append({
                    "date": date_str,
                    "minutes": mins,
                    "started": bool(started),
                })
        else:
            # Fallback: use player game log
            if name in player_game_logs:
                gl = player_game_logs[name].copy()
                gl["MIN_FLOAT"] = gl["MIN"].apply(parse_minutes)
                for _, row in gl.head(10).iterrows():
                    try:
                        date_str = pd.to_datetime(row["GAME_DATE"]).strftime("%Y-%m-%d")
                    except Exception:
                        date_str = str(row["GAME_DATE"])[:10]
                    last_10_games.append({
                        "date": date_str,
                        "minutes": round(row["MIN_FLOAT"], 1),
                        "started": False,  # game log doesn't have started info
                    })

        # Pad to 10 games
        while len(last_10_games) < 10:
            last_10_games.append({"date": "", "minutes": 0, "started": False})

        # Compute L5/L10
        last_10_mins = [g["minutes"] for g in last_10_games]
        l5_games = [m for m in last_10_mins[:5] if m > 0]
        l10_games = [m for m in last_10_mins if m > 0]
        l5 = round(sum(l5_games) / len(l5_games), 1) if l5_games else 0.0
        l10 = round(sum(l10_games) / len(l10_games), 1) if l10_games else 0.0

        # Get substitution chain for this player
        sub_chain = []
        nba_tm_matrix = replacement_matrix.get(nba_tm, {})
        if nba_tm_matrix:
            for mk in nba_tm_matrix:
                if mk.lower() == name_norm or (last_name and mk.split()[-1].lower() == last_name):
                    subs_sorted = sorted(
                        nba_tm_matrix[mk].items(),
                        key=lambda x: x[1]["pct"],
                        reverse=True,
                    )
                    for in_name, info in subs_sorted[:4]:
                        if info["pct"] >= 0.05:
                            sub_chain.append({
                                "replacement": in_name,
                                "pct": round(info["pct"], 3),
                            })
                    break

        opponent = matchups.get(team, "")

        player_profiles.append({
            "name": name,
            "team": team,
            "nba_team": nba_tm,
            "pos": pos,
            "price": price,
            "currentProjMin": stok_min,
            "l5Avg": l5,
            "l10Avg": l10,
            "last10Games": last_10_games[:10],
            "substitutionChain": sub_chain,
            "opponent": opponent,
        })

    return player_profiles, replacement_matrix


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: QUERY CLAUDE FOR MINUTES PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════

CLAUDE_SYSTEM = """You are an NBA minutes projection analyst for DFS (daily fantasy sports).
Given a team's recent game logs, substitution patterns, and injury context,
project realistic minutes for each active player in tonight's game.

Rules:
- Team minutes must total approximately 240 (5 players x 48 min, accounting for overtime probability)
- Consider recent trends heavily (L3 > L5 > L10 weighting)
- Account for minutes redistribution when players are OUT
- Players who consistently start get priority for 30+ min projections
- Consider blowout risk: high spreads may reduce starter minutes in Q4
- Players with 0 projected minutes from the data source are OUT — do not project minutes for them

Respond ONLY with valid JSON. No markdown fences, no commentary outside the JSON."""


def build_team_prompt(team, opponent, team_players, replacement_matrix):
    """Build a Claude prompt for one team's minutes projections."""
    active = [p for p in team_players if p["currentProjMin"] > 0]
    out = [p for p in team_players if p["currentProjMin"] == 0 and p["l5Avg"] > 0]

    if not active:
        return None

    lines = [f"Project minutes for {team} vs {opponent} tonight.\n"]
    lines.append("ACTIVE PLAYERS (sorted by L5 avg):")

    active_sorted = sorted(active, key=lambda x: x["l5Avg"], reverse=True)
    for p in active_sorted:
        games_str = "  ".join(
            f"{g['minutes']:.0f}{'*' if g['started'] else ''}"
            for g in p["last10Games"]
            if g["date"]
        )
        lines.append(
            f"- {p['name']} ({p['pos']}, ${p['price']}): L5={p['l5Avg']}, L10={p['l10Avg']}"
        )
        lines.append(f"  Last 10: {games_str}")
        lines.append(f"  (* = started that game)")

    if out:
        lines.append("\nOUT PLAYERS:")
        for p in out:
            lines.append(f"- {p['name']}: L5={p['l5Avg']} when active")

        # Substitution chains for out players
        nba_team = dk_to_nba_team(team)
        tm_matrix = replacement_matrix.get(nba_team, {})
        if tm_matrix:
            lines.append("\nSUBSTITUTION PATTERNS (when player goes OUT):")
            for p in out:
                name_norm = normalize_name(p["name"]).lower()
                last_name = name_norm.split()[-1] if name_norm.split() else ""
                for mk in tm_matrix:
                    if mk.lower() == name_norm or mk.split()[-1].lower() == last_name:
                        subs = sorted(tm_matrix[mk].items(), key=lambda x: x[1]["pct"], reverse=True)
                        chain = ", ".join(
                            f"{in_name} ({info['pct']*100:.0f}%)"
                            for in_name, info in subs[:4]
                            if info["pct"] >= 0.05
                        )
                        if chain:
                            lines.append(f"- When {p['name']} sits: {chain}")
                        break

    proj_str = ", ".join(f"{p['name']}={p['currentProjMin']}" for p in active_sorted)
    lines.append(f"\nCurrent projected minutes from data source: {proj_str}")

    lines.append("""
Respond with this exact JSON format:
{"players":[{"name":"Player Name","projectedMinutes":35.5,"confidence":"high","reasoning":"Brief 1-sentence explanation"}]}""")

    return "\n".join(lines)


def query_claude(player_profiles, replacement_matrix):
    """Send per-team data to Claude Haiku, get projected minutes."""
    if not ANTHROPIC_API_KEY:
        print("  No ANTHROPIC_API_KEY — skipping Claude projections")
        return {}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Group players by team
    team_players = defaultdict(list)
    for p in player_profiles:
        team_players[p["team"]].append(p)

    projections = {}  # name -> {projectedMinutes, confidence, reasoning}

    for team, players in team_players.items():
        opponent = players[0].get("opponent", "?") if players else "?"
        prompt = build_team_prompt(team, opponent, players, replacement_matrix)
        if not prompt:
            continue

        print(f"  Querying Claude for {team} vs {opponent}...")

        for attempt in range(2):
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=1024,
                    temperature=0.3,
                    system=CLAUDE_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )

                text = response.content[0].text.strip()
                # Strip markdown fences if present
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\n?", "", text)
                    text = re.sub(r"\n?```$", "", text)

                result = json.loads(text)
                for p in result.get("players", []):
                    name = p.get("name", "")
                    if name:
                        projections[name] = {
                            "projectedMinutes": round(p.get("projectedMinutes", 0), 1),
                            "confidence": p.get("confidence", "medium"),
                            "reasoning": p.get("reasoning", ""),
                        }
                break  # success
            except json.JSONDecodeError as e:
                print(f"    JSON parse error for {team}: {e}")
                if attempt == 0:
                    time.sleep(2)
            except Exception as e:
                print(f"    Claude API error for {team}: {e}")
                if attempt == 0:
                    time.sleep(5)

        time.sleep(1)  # rate limit between teams

    print(f"  Got projections for {len(projections)} players")
    return projections


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: BUILD OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def build_output(player_profiles, claude_projections):
    """Assemble the final minutes-data.json structure."""
    players_out = []
    for p in player_profiles:
        proj = claude_projections.get(p["name"], {})

        players_out.append({
            "name": p["name"],
            "team": p["team"],
            "pos": p["pos"],
            "price": p["price"],
            "projectedMinutes": proj.get("projectedMinutes", None),
            "confidence": proj.get("confidence", None),
            "reasoning": proj.get("reasoning", None),
            "l5Avg": p["l5Avg"],
            "l10Avg": p["l10Avg"],
            "currentProjMin": p["currentProjMin"],
            "last10Games": p["last10Games"],
        })

    # Sort by team, then projected minutes descending
    players_out.sort(key=lambda x: (
        x["team"],
        -(x["projectedMinutes"] or x["l5Avg"] or 0),
    ))

    output = {
        "lastUpdated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": SEASON,
        "players": players_out,
    }

    return output


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  NBA Minutes Agent")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    # Step 1: Load slate
    try:
        slate_players, games, teams_playing, matchups = load_slate()
    except Exception as e:
        print(f"ERROR loading slate: {e}")
        output = {
            "lastUpdated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "season": SEASON,
            "error": f"No slate data: {str(e)[:100]}",
            "players": [],
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote empty {OUTPUT_FILE}")
        return

    if not slate_players:
        print("No slate players found in data.json")
        output = {
            "lastUpdated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "season": SEASON,
            "error": "No slate data available",
            "players": [],
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)
        return

    # Step 2: Fetch NBA data
    print("\nFetching NBA data...")
    player_game_logs, player_id_map, box_scores, sub_pairs, started_data = fetch_nba_data(
        slate_players, teams_playing
    )

    # Step 3: Build player profiles
    print("\nBuilding player profiles...")
    player_profiles, replacement_matrix = build_player_profiles(
        slate_players, player_game_logs, box_scores, sub_pairs, started_data, matchups
    )
    print(f"  {len(player_profiles)} player profiles built")

    # Step 4: Query Claude
    print("\nQuerying Claude for projections...")
    claude_projections = query_claude(player_profiles, replacement_matrix)

    # Step 5: Build output
    print("\nBuilding output...")
    output = build_output(player_profiles, claude_projections)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    proj_count = sum(1 for p in output["players"] if p.get("projectedMinutes"))
    print(f"\nWrote {OUTPUT_FILE}: {len(output['players'])} players, {proj_count} with Claude projections")
    print("Done!")


if __name__ == "__main__":
    main()
