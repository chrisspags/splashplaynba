#!/usr/bin/env python3
"""
NBA Minutes Agent — Claude-Only Pipeline
Reads player-profiles.json (game history) and dk-slates.json/fd-slates.json (fresh slate data)
from GitHub, fetches the latest NBA injury report PDF, sends per-team data to Claude Haiku
for minutes projections, and writes minutes-data.json.

No NBA API calls — game data is from Colab, slate data is from DK/FD API pulls on the site.
"""

import csv
import io
import json
import os
import re
import base64
import time
import unicodedata
import requests
from datetime import datetime, timedelta

import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
REPO_OWNER = "chrisspags"
REPO_NAME = "splashplaynba"
PROFILES_FILE = "player-profiles.json"
OUTPUT_FILE = "minutes-data.json"
CSV_FILE = "minutes-export.csv"


# ── Utility ───────────────────────────────────────────────────────────────────

def normalize_name(name):
    """Strip diacritics, remove periods/apostrophes, normalize whitespace."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(ascii_name.split()).strip().lower()


def gh_fetch_json(filename):
    """Fetch a JSON file from GitHub repo, handling large files."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{filename}"
    resp = requests.get(url, headers=headers)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    gh_data = resp.json()

    if gh_data.get("content"):
        return json.loads(base64.b64decode(gh_data["content"]))
    elif gh_data.get("download_url"):
        raw = requests.get(gh_data["download_url"], headers=headers)
        raw.raise_for_status()
        return raw.json()
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD PROFILES FROM GITHUB
# ══════════════════════════════════════════════════════════════════════════════

def load_profiles():
    """Read player-profiles.json from GitHub repo."""
    print("Loading player-profiles.json from GitHub...")
    data = gh_fetch_json(PROFILES_FILE)
    if not data:
        raise Exception("player-profiles.json not found on GitHub")

    profiles = data.get("profiles", [])
    generated_at = data.get("generatedAt", "unknown")
    season = data.get("season", "")

    print(f"  Loaded {len(profiles)} profiles (generated: {generated_at})")
    return profiles, season, generated_at


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1b: LOAD SLATE DATA (dk-slates.json → fd-slates.json → data.json)
# ══════════════════════════════════════════════════════════════════════════════

def load_slate_data():
    """Read slate data from DK slates, FD slates, or data.json as fallback."""
    slate = {}

    # Try dk-slates.json first
    print("Loading dk-slates.json from GitHub...")
    dk_data = gh_fetch_json("dk-slates.json")
    if dk_data and dk_data.get("slates"):
        # Find main slate (most games, SalaryCap, has players)
        best_key = None
        best_gc = 0
        for key, sl in dk_data["slates"].items():
            if sl.get("players") and sl.get("gameCount", 0) > best_gc:
                gt = sl.get("gameType", "")
                if gt == "SalaryCap" or not gt:
                    best_gc = sl["gameCount"]
                    best_key = key

        if best_key:
            dk_slate = dk_data["slates"][best_key]
            label = dk_slate.get("label", f"DK Slate {best_key}")
            print(f"  Using DK slate: {label} ({len(dk_slate['players'])} players)")

            for p in dk_slate["players"]:
                name = normalize_name(p.get("name", ""))
                if name:
                    slate[name] = {
                        "min": p.get("fppg", 0),  # FPPG as projection proxy
                        "price": p.get("salary", 0),
                        "pos": p.get("pos", ""),
                        "team": p.get("team", ""),
                    }

            if slate:
                print(f"  Loaded {len(slate)} players from DK slate")
                return slate

    # Try fd-slates.json
    print("  No DK slates, trying fd-slates.json...")
    fd_data = gh_fetch_json("fd-slates.json")
    if fd_data:
        best_key = None
        best_gc = 0
        for key, sl in fd_data.items():
            if isinstance(sl, dict) and sl.get("players") and sl.get("gameCount", 0) > best_gc:
                best_gc = sl["gameCount"]
                best_key = key

        if best_key:
            fd_slate = fd_data[best_key]
            label = fd_slate.get("label", f"FD Slate {best_key}")
            print(f"  Using FD slate: {label} ({len(fd_slate['players'])} players)")

            for p in fd_slate["players"]:
                name = normalize_name(p.get("name", ""))
                if name:
                    slate[name] = {
                        "min": p.get("fppg", 0),
                        "price": p.get("salary", 0),
                        "pos": p.get("pos", ""),
                        "team": p.get("team", ""),
                    }

            if slate:
                print(f"  Loaded {len(slate)} players from FD slate")
                return slate

    # Fallback to data.json
    print("  No DK/FD slates, falling back to data.json...")
    data_json = gh_fetch_json("data.json")
    if data_json:
        my_proj = data_json.get("myProj", [])
        for p in my_proj:
            name = normalize_name(p.get("name", ""))
            if name:
                slate[name] = {
                    "min": p.get("min", 0),
                    "price": p.get("price", 0),
                    "pos": p.get("pos", ""),
                    "team": p.get("team", ""),
                }
        print(f"  Loaded {len(slate)} players from data.json")

    return slate


def merge_slate_into_profiles(profiles, slate):
    """Update price/pos/team in profiles from fresh slate data. No external minutes projection."""
    updated = 0
    for p in profiles:
        key = normalize_name(p["name"])
        if key in slate:
            s = slate[key]
            p["price"] = s["price"]
            if s["pos"]:
                p["pos"] = s["pos"]
            if s["team"]:
                p["team"] = s["team"]
            updated += 1
    print(f"  Merged slate data (price/pos/team) for {updated}/{len(profiles)} players")
    return profiles


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1c: FETCH NBA INJURY REPORT
# ══════════════════════════════════════════════════════════════════════════════

def fetch_injury_report(game_date_str=None):
    """Fetch and parse the latest NBA injury report PDF."""
    try:
        import pdfplumber
    except ImportError:
        print("  pdfplumber not installed — skipping injury report")
        print("  Add 'pdfplumber' to scripts/requirements.txt")
        return {}

    if not game_date_str:
        game_date_str = datetime.utcnow().strftime("%Y-%m-%d")

    print(f"Fetching NBA injury report for {game_date_str}...")

    # Try descending timestamps — every 15 min from 11:45PM to 10:00AM
    times_to_try = [
        "11_45PM", "11_30PM", "11_15PM", "11_00PM", "10_45PM", "10_30PM", "10_15PM", "10_00PM",
        "09_45PM", "09_30PM", "09_15PM", "09_00PM", "08_45PM", "08_30PM", "08_15PM", "08_00PM",
        "07_45PM", "07_30PM", "07_15PM", "07_00PM", "06_45PM", "06_30PM", "06_15PM", "06_00PM",
        "05_45PM", "05_30PM", "05_15PM", "05_00PM", "04_45PM", "04_30PM", "04_15PM", "04_00PM",
        "03_45PM", "03_30PM", "03_15PM", "03_00PM", "02_45PM", "02_30PM", "02_15PM", "02_00PM",
        "01_45PM", "01_30PM", "01_15PM", "01_00PM", "12_45PM", "12_30PM", "12_15PM", "12_00PM",
        "11_45AM", "11_30AM", "11_15AM", "11_00AM", "10_45AM", "10_30AM", "10_15AM", "10_00AM",
    ]

    injury_pdf_url = None

    # Try today
    for t in times_to_try:
        url = f"https://ak-static.cms.nba.com/referee/injury/Injury-Report_{game_date_str}_{t}.pdf"
        try:
            r = requests.head(url, timeout=5)
            if r.status_code == 200:
                injury_pdf_url = url
                print(f"  Found: {url}")
                break
        except Exception:
            continue

    # Try yesterday (day-before reports)
    if not injury_pdf_url:
        yesterday = (datetime.strptime(game_date_str, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        for t in times_to_try[:10]:
            url = f"https://ak-static.cms.nba.com/referee/injury/Injury-Report_{yesterday}_{t}.pdf"
            try:
                r = requests.head(url, timeout=5)
                if r.status_code == 200:
                    injury_pdf_url = url
                    print(f"  Found day-before report: {url}")
                    break
            except Exception:
                continue

    if not injury_pdf_url:
        print("  No injury report PDF found")
        return {}

    # Parse PDF
    try:
        pdf_resp = requests.get(injury_pdf_url)
        pdf_resp.raise_for_status()

        injuries = {}  # normalized_name -> {status, reason, team}

        with pdfplumber.open(io.BytesIO(pdf_resp.content)) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if not table:
                    continue
                for row in table:
                    if not row or len(row) < 7 or row[0] == "Game Date":
                        continue
                    player_name = (row[4] or "").strip()
                    status = (row[5] or "").strip()
                    reason = (row[6] or "").strip()
                    team = (row[3] or "").strip()

                    if player_name and status:
                        key = normalize_name(player_name)
                        injuries[key] = {
                            "status": status,
                            "reason": reason,
                            "team": team,
                        }

        out_count = sum(1 for v in injuries.values() if v["status"].lower() in ("out", "doubtful"))
        q_count = sum(1 for v in injuries.values() if v["status"].lower() == "questionable")
        print(f"  Parsed {len(injuries)} injury entries: {out_count} Out/Doubtful, {q_count} Questionable")
        return injuries

    except Exception as e:
        print(f"  Error parsing injury report: {e}")
        return {}


def apply_injuries_to_profiles(profiles, injuries):
    """Flag injured players in profiles."""
    flagged = 0
    for p in profiles:
        key = normalize_name(p["name"])
        if key in injuries:
            inj = injuries[key]
            p["injuryStatus"] = inj["status"]
            p["injuryReason"] = inj["reason"]
            flagged += 1
        else:
            p["injuryStatus"] = "Available"
            p["injuryReason"] = ""

    print(f"  Flagged {flagged} players with injury status")
    return profiles


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: QUERY CLAUDE FOR MINUTES PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════

CLAUDE_SYSTEM = """You are the world's most accurate NBA minutes allocation analyst for DFS (daily fantasy sports).
You must distribute exactly 240 team minutes across active players for tonight's game (5 players × 48 min = 240). You may add up to 2 minutes for overtime probability (target: 240-242 total). Do not go beneath 240.

You are building minutes projections FROM SCRATCH using only historical game data, substitution patterns, and injury reports. There is no external data source to anchor to — you are the projection engine.

ALLOCATION PROCESS — follow these steps in order:

1. ESTABLISH THE ROTATION from game logs:
   - Players who started 80%+ of recent games are STARTERS. They get priority minutes.
   - Look at L5 average as the primary baseline for each player's expected minutes.
   - If L3 trend diverges from L5 by 3+ minutes, weight L3 more heavily (coach is adjusting rotation).
   - Players with L5 < 10 min are deep bench — they get leftover minutes.

2. ACCOUNT FOR INJURIES using substitution chain data:
   - When a player is OUT, their minutes get redistributed. The SUBSTITUTION CHAIN shows exactly who historically replaces them based on play-by-play data.
   - Example: If Player A (30 min avg) is OUT and their chain shows "Player B (60%), Player C (40%)", give ~18 extra min to B and ~12 to C.
   - The sub chain is the BEST signal for how a coach actually redistributes minutes — use it.
   - If no sub chain data exists for an OUT player, distribute their minutes to same-position players proportionally by current minutes.

3. ADJUST FOR CONTEXT:
   - Questionable players: reduce by ~20-30%, redistribute the difference via sub chains.
   - Back-to-back games: reduce starters by 1-3 min unless their game log shows they maintain minutes on B2Bs.
   - Blowout risk (large spread): starters may lose 2-4 min to bench in 4th quarter.
   - Players returning from injury (recent 0s in game log but now active): start conservative, use their most recent active game as baseline rather than L5/L10.

4. BALANCE TO 240-242:
   - Sum all projections. Adjust bench players (lowest-minutes) up or down by 1-2 min each.
   - Do NOT reduce starters to fix the total — only adjust bench/role players.

INJURY STATUS KEY:
- "Out" or "Doubtful" = will NOT play. Project 0 minutes. Redistribute using sub chains.
- "Questionable" = uncertain. Assume they will play normally and we'll update if they're out.
- "Probable" or "Available" = will play normally.

CRITICAL RULES:
- USE THE SUBSTITUTION CHAIN DATA. This is real play-by-play data showing coaching patterns. It's your best tool.
- L5 average is your primary baseline for each player. L3 trend is your adjustment signal.
- Starters (started 80%+ of games): typically 24-38 min. Never below 20 unless injury-limited.
- Rotation players (started <50%, L5 15-28 min): 12-28 min range.
- Deep bench (L5 < 12 min): 0-15 min range. These are your flex for hitting 240.
- No player should exceed 40 minutes unless their L5 is 38+.
- Never project an active player below 5 minutes if they've played 10+ min in L5.

MANDATORY: Your projected minutes MUST sum to between 240 and 242. Add them up before responding. Include the sum as "totalMinutes" in your JSON output.

Respond ONLY with valid JSON. No markdown fences, no commentary."""


def build_team_prompt(team, opponent, team_players):
    """Build a Claude prompt for one team's minutes projections with full rotation data."""
    # Active = has recent minutes (L5 or L10 > 0) and not OUT/Doubtful
    active = [p for p in team_players if
              (p.get("l5Avg", 0) > 0 or p.get("l10Avg", 0) > 0)
              and p.get("injuryStatus", "Available").lower() not in ("out", "doubtful")]
    # Out = explicitly OUT/Doubtful, or no recent minutes at all
    out = [p for p in team_players if
           p.get("injuryStatus", "").lower() in ("out", "doubtful")
           or (p.get("l5Avg", 0) == 0 and p.get("l10Avg", 0) == 0)]
    questionable = [p for p in active if p.get("injuryStatus", "").lower() == "questionable"]

    if not active:
        return None

    lines = [f"Project minutes for {team} vs {opponent} tonight.\n"]

    # ── Active players with full game-by-game data ──
    lines.append("ACTIVE PLAYERS (sorted by L5 avg):")

    active_sorted = sorted(active, key=lambda x: x.get("l5Avg", 0), reverse=True)
    for p in active_sorted:
        injury_tag = ""
        if p.get("injuryStatus", "").lower() == "questionable":
            injury_tag = f" [QUESTIONABLE: {p.get('injuryReason', '')}]"
        elif p.get("injuryStatus", "").lower() == "probable":
            injury_tag = f" [PROBABLE: {p.get('injuryReason', '')}]"

        lines.append(
            f"\n- {p['name']} ({p.get('pos', '?')}, ${p.get('price', 0)}){injury_tag}: "
            f"L5={p.get('l5Avg', 0)}, L10={p.get('l10Avg', 0)}"
        )

        # Game-by-game minutes with stats and starter flag
        games = p.get("last10Games", [])
        if games:
            game_details = []
            for g in games:
                if not g.get("date"):
                    continue
                mins = g.get("minutes", 0)
                started = "S" if g.get("started") else "B"  # Starter vs Bench
                dkfp = g.get("dkfp", 0)
                pts = g.get("pts", 0)
                reb = g.get("reb", 0)
                ast = g.get("ast", 0)
                game_details.append(
                    f"{g['date'][-5:]}: {mins:.0f}min({started}) {pts}p/{reb}r/{ast}a {dkfp:.0f}fp"
                )
            if game_details:
                lines.append(f"  Game log (recent→old): {' | '.join(game_details)}")

            # Starter consistency
            started_count = sum(1 for g in games if g.get("started") and g.get("date"))
            total_games = sum(1 for g in games if g.get("date") and g.get("minutes", 0) > 0)
            if total_games > 0:
                lines.append(f"  Started {started_count}/{total_games} recent games")

        # Substitution chain for this player (who replaces them)
        chain = p.get("substitutionChain", [])
        if chain:
            chain_str = ", ".join(
                f"{s['replacement']} ({s['pct']*100:.0f}%)"
                for s in chain[:3]
            )
            lines.append(f"  Sub chain (when sits): {chain_str}")

    # ── Out/Doubtful players + their rotation impact ──
    if out:
        lines.append("\n\nOUT/DOUBTFUL PLAYERS & ROTATION IMPACT:")
        out_with_minutes = [p for p in out if p.get("l5Avg", 0) > 0]
        for p in out_with_minutes:
            reason = ""
            if p.get("injuryReason"):
                reason = f" — {p['injuryReason']}"
            elif p.get("injuryStatus"):
                reason = f" — {p['injuryStatus']}"

            lines.append(
                f"- {p['name']} ({p.get('pos', '?')}): L5={p.get('l5Avg', 0)} min/game when active{reason}"
            )

            # Who absorbs their minutes
            chain = p.get("substitutionChain", [])
            if chain:
                chain_str = ", ".join(
                    f"{s['replacement']} ({s['pct']*100:.0f}%)"
                    for s in chain
                )
                lines.append(f"  → Their {p.get('l5Avg', 0):.0f} minutes typically go to: {chain_str}")
            else:
                lines.append(f"  → No historical sub data — distribute to same-position players")

            # Show their recent game log so Claude can see the role they occupied
            games = p.get("last10Games", [])
            if games:
                recent = [g for g in games[:5] if g.get("date") and g.get("minutes", 0) > 0]
                if recent:
                    game_str = ", ".join(f"{g['minutes']:.0f}min" for g in recent)
                    lines.append(f"  Recent when active: {game_str}")

    if questionable:
        lines.append(f"\n⚠️ {len(questionable)} QUESTIONABLE players — reduce their minutes ~20-30% and redistribute to teammates")

    # ── Summary ──
    l5_str = ", ".join(
        f"{p['name']}={p.get('l5Avg', 0):.0f}"
        for p in active_sorted
    )
    lines.append(f"\nL5 averages (your baseline): {l5_str}")

    total_l5 = sum(p.get("l5Avg", 0) for p in active_sorted)
    out_minutes = sum(p.get("l5Avg", 0) for p in out if p.get("l5Avg", 0) > 0)
    lines.append(f"Active players L5 total: {total_l5:.0f} min ({len(active_sorted)} players)")

    if out_minutes > 0:
        lines.append(f"\n🔴 {len([p for p in out if p.get('l5Avg', 0) > 0])} rotation players are OUT, freeing ~{out_minutes:.0f} minutes.")
        lines.append("Use the substitution chains above to redistribute these minutes to the correct teammates.")
        lines.append(f"Target total after redistribution: 240-242 min across {len(active_sorted)} active players.")
    else:
        lines.append(f"\nNo significant OUT players — distribute 240-242 total minutes, using L5 as baseline and adjusting for L3 trends.")

    lines.append("\nAdd up your projections before responding to verify the sum equals 240-242.")

    lines.append("""
Respond with this exact JSON format:
{"totalMinutes":240.5,"players":[{"name":"Player Name","projectedMinutes":35.5,"confidence":"high","reasoning":"Brief 1-sentence explanation"}]}""")

    return "\n".join(lines)


def query_claude(profiles):
    """Send per-team data to Claude Haiku, get projected minutes."""
    if not ANTHROPIC_API_KEY:
        print("  No ANTHROPIC_API_KEY — skipping Claude projections")
        return {}

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Group players by team
    from collections import defaultdict
    team_players = defaultdict(list)
    for p in profiles:
        team_players[p["team"]].append(p)

    projections = {}  # name -> {projectedMinutes, confidence, reasoning}

    for team, players in team_players.items():
        opponent = players[0].get("opponent", "?") if players else "?"
        prompt = build_team_prompt(team, opponent, players)
        if not prompt:
            continue

        print(f"  Querying Claude for {team} vs {opponent}...")

        messages = [{"role": "user", "content": prompt}]
        players_list = None

        for attempt in range(3):
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=2048,
                    temperature=0.3,
                    system=CLAUDE_SYSTEM,
                    messages=messages,
                )

                text = response.content[0].text.strip()
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\n?", "", text)
                    text = re.sub(r"\n?```$", "", text)

                result = json.loads(text)
                players_list = result.get("players", [])
                team_total = sum(p.get("projectedMinutes", 0) for p in players_list)

                if team_total > 0 and not (239 <= team_total <= 242):
                    diff = 241.0 - team_total
                    if abs(diff) <= 5:
                        bench = [p for p in players_list if p.get("projectedMinutes", 0) < 25]
                        if bench:
                            adj = diff / len(bench)
                            for p in bench:
                                p["projectedMinutes"] = round(p["projectedMinutes"] + adj, 1)
                            new_total = sum(p.get("projectedMinutes", 0) for p in players_list)
                            print(f"    Adjusted {team} bench: {team_total:.0f} → {new_total:.1f} min")
                        break
                    elif attempt < 2:
                        proj_summary = ", ".join(
                            f"{p['name']}={p.get('projectedMinutes', 0)}"
                            for p in players_list
                        )
                        correction = (
                            f"Your total was {team_total:.1f} minutes, which is off by {abs(diff):.0f}. "
                            f"An NBA team plays exactly 240 minutes per game (5 x 48). "
                            f"Here were your projections: {proj_summary}. "
                            f"Redistribute these to sum to exactly 240-241. "
                            f"Keep starters close to their current values and adjust bench players. "
                            f"Respond with the corrected JSON in the same format."
                        )
                        messages.append({"role": "assistant", "content": text})
                        messages.append({"role": "user", "content": correction})
                        print(f"    Total was {team_total:.0f} — sending correction (attempt {attempt + 2})...")
                        time.sleep(1)
                        continue
                    else:
                        print(f"    WARNING: {team} still at {team_total:.0f} after correction — accepting")
                        break
                else:
                    break

            except json.JSONDecodeError as e:
                print(f"    JSON parse error for {team}: {e}")
                if attempt < 2:
                    time.sleep(2)
            except Exception as e:
                print(f"    Claude API error for {team}: {e}")
                if attempt < 2:
                    time.sleep(5)

        if players_list:
            for p in players_list:
                name = p.get("name", "")
                if name:
                    projections[name] = {
                        "projectedMinutes": round(p.get("projectedMinutes", 0), 1),
                        "confidence": p.get("confidence", "medium"),
                        "reasoning": p.get("reasoning", ""),
                    }
            final_total = sum(p.get("projectedMinutes", 0) for p in players_list)
            print(f"    ✓ Got {len(players_list)} projections ({final_total:.1f} min)")

        time.sleep(1)

    print(f"  Got projections for {len(projections)} players total")
    return projections


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: BUILD OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def build_output(profiles, claude_projections, season):
    """Assemble the final minutes-data.json structure."""
    players_out = []
    for p in profiles:
        proj = claude_projections.get(p["name"], {})

        players_out.append({
            "name": p["name"],
            "team": p["team"],
            "pos": p.get("pos", ""),
            "price": p.get("price", 0),
            "projectedMinutes": proj.get("projectedMinutes", None),
            "confidence": proj.get("confidence", None),
            "reasoning": proj.get("reasoning", None),
            "injuryStatus": p.get("injuryStatus", "Available"),
            "injuryReason": p.get("injuryReason", ""),
            "l5Avg": p.get("l5Avg", 0),
            "l10Avg": p.get("l10Avg", 0),
            "last10Games": p.get("last10Games", []),
        })

    players_out.sort(key=lambda x: (
        x["team"],
        -(x["projectedMinutes"] or x["l5Avg"] or 0),
    ))

    output = {
        "lastUpdated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": season,
        "players": players_out,
    }

    return output


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: WRITE CSV EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def write_csv(output):
    """Write a flat CSV of all minutes data for the Minutes tab."""
    players = output.get("players", [])
    if not players:
        print("  No players — skipping CSV")
        return

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Player", "Team", "Proj Min", "Confidence", "Injury",
            "L5", "L10",
            "G1", "G2", "G3", "G4", "G5",
            "G6", "G7", "G8", "G9", "G10",
        ])
        for p in players:
            games = p.get("last10Games", [])
            while len(games) < 10:
                games.append({})
            row = [
                p.get("name", ""),
                p.get("team", ""),
                p.get("projectedMinutes", ""),
                p.get("confidence", ""),
                p.get("injuryStatus", ""),
                p.get("l5Avg", 0),
                p.get("l10Avg", 0),
            ]
            for g in games[:10]:
                row.append(g.get("minutes", "") if g else "")
            writer.writerow(row)

    print(f"  Wrote {CSV_FILE}: {len(players)} rows")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  NBA Minutes Agent (Claude-only)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 60)

    # Step 1: Load profiles
    try:
        profiles, season, generated_at = load_profiles()
    except Exception as e:
        print(f"ERROR loading profiles: {e}")
        output = {
            "lastUpdated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "season": "2025-26",
            "error": f"No profile data: {str(e)[:100]}",
            "players": [],
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote empty {OUTPUT_FILE}")
        return

    if not profiles:
        print("No profiles found in player-profiles.json")
        output = {
            "lastUpdated": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "season": "2025-26",
            "error": "No profile data available",
            "players": [],
        }
        with open(OUTPUT_FILE, "w") as f:
            json.dump(output, f, indent=2)
        return

    # Step 1b: Merge fresh slate data (DK/FD/data.json)
    try:
        slate = load_slate_data()
        if slate:
            profiles = merge_slate_into_profiles(profiles, slate)
    except Exception as e:
        print(f"WARNING: Could not load slate data: {e}")
        print("  Falling back to currentProjMin from player-profiles.json")

    # Step 1c: Fetch injury report
    injuries = fetch_injury_report()
    if injuries:
        profiles = apply_injuries_to_profiles(profiles, injuries)
    else:
        for p in profiles:
            p["injuryStatus"] = "Available"
            p["injuryReason"] = ""

    # Step 2: Query Claude
    print("\nQuerying Claude for projections...")
    claude_projections = query_claude(profiles)

    # Step 3: Build output
    print("\nBuilding output...")
    output = build_output(profiles, claude_projections, season)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    # Step 4: Write CSV export
    write_csv(output)

    proj_count = sum(1 for p in output["players"] if p.get("projectedMinutes"))
    out_count = sum(1 for p in output["players"] if p.get("injuryStatus", "").lower() in ("out", "doubtful"))
    print(f"\nWrote {OUTPUT_FILE}: {len(output['players'])} players, {proj_count} with projections, {out_count} OUT")
    print("Done!")


if __name__ == "__main__":
    main()
