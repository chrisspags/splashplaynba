#!/usr/bin/env python3
"""
NBA Minutes Agent — Claude-Only Pipeline
Reads player-profiles.json (game history) and data.json (fresh Stokastic minutes)
from GitHub, sends per-team data to Claude Haiku for minutes projections,
and writes minutes-data.json.

No NBA API calls — game data is from Colab, Stokastic data is from site upload.
"""

import csv
import json
import os
import re
import base64
import time
import unicodedata
import requests
from datetime import datetime

import anthropic

# ── Configuration ─────────────────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
REPO_OWNER = "chrisspags"
REPO_NAME = "splashplaynba"
PROFILES_FILE = "player-profiles.json"
OUTPUT_FILE = "minutes-data.json"


# ── Utility ───────────────────────────────────────────────────────────────────

def normalize_name(name):
    """Strip diacritics, remove periods/apostrophes, normalize whitespace."""
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    ascii_name = ascii_name.replace(".", "").replace("'", "").replace("-", " ")
    return " ".join(ascii_name.split()).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: LOAD PROFILES FROM GITHUB
# ══════════════════════════════════════════════════════════════════════════════

def load_profiles():
    """Read player-profiles.json from GitHub repo."""
    print("Loading player-profiles.json from GitHub...")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/{PROFILES_FILE}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    gh_data = resp.json()
    data = json.loads(base64.b64decode(gh_data["content"]))

    profiles = data.get("profiles", [])
    generated_at = data.get("generatedAt", "unknown")
    season = data.get("season", "")

    print(f"  Loaded {len(profiles)} profiles (generated: {generated_at})")
    return profiles, season, generated_at


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1b: LOAD FRESH SLATE DATA FROM data.json
# ══════════════════════════════════════════════════════════════════════════════

def load_slate_data():
    """Read data.json from GitHub to get fresh Stokastic minutes + prices."""
    print("Loading data.json from GitHub...")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/contents/data.json"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    gh_data = resp.json()
    data = json.loads(base64.b64decode(gh_data["content"]))

    my_proj = data.get("myProj", [])
    # Build lookup: normalized name -> {min, price, pos, team}
    slate = {}
    for p in my_proj:
        name = normalize_name(p.get("name", ""))
        if name:
            slate[name] = {
                "min": p.get("min", 0),
                "price": p.get("price", 0),
                "pos": p.get("pos", ""),
                "team": p.get("team", ""),
            }

    print(f"  Loaded {len(slate)} players from data.json slate")
    return slate


def merge_slate_into_profiles(profiles, slate):
    """Override currentProjMin/price/pos in profiles with fresh data.json values."""
    updated = 0
    for p in profiles:
        key = normalize_name(p["name"])
        if key in slate:
            s = slate[key]
            if s["min"] > 0:
                p["currentProjMin"] = s["min"]
                p["price"] = s["price"]
                if s["pos"]:
                    p["pos"] = s["pos"]
                updated += 1
    print(f"  Merged fresh slate data for {updated}/{len(profiles)} players")
    return profiles


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: QUERY CLAUDE FOR MINUTES PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════

CLAUDE_SYSTEM = """You are the world's most accurate NBA minutes allocation analyst for DFS (daily fantasy sports).
You must distribute exactly 240 team minutes across active players for tonight's game (5 players × 48 min = 240). You may add up to 2 minutes for overtime probability (target: 240-242 total). Do not go beneath 240.

ALLOCATION PROCESS — follow these steps in order:
1. Start with each player's data source projection as the baseline. These projections already account for injury status, rest days, and lineup changes.
2. Adjust projections up or down based on recent trends (L5 avg vs data source), but stay within ~10 minutes of the data source projection for each player.
3. Sum all projected minutes. If the total is not between 240 and 242, adjust bench players (lowest-minutes players) up or down by 1-2 min each until the total equals 240-242. Do NOT reduce starters to fix the total.
4. When a player returns from injury (when multiple games in L3, L5, or L10 show recent 0s but data source is giving them minutes), defer to the data source's minutes.

CRITICAL RULES:
- Every active player's projection must be within 10 minutes of their data source projection. Do NOT drastically reduce or inflate any player.
- If a player is listed as ACTIVE with a data source projection > 0, they WILL play. Never project an active player below 5 minutes.
- Players with 0 projected minutes from the data source are OUT — do not include them.
- No player should exceed 40 minutes unless their data source projection is 38+.
- Weight recent games: L3 > L5 > L10, but the data source projection is the anchor.
- Consider when teams are playing games on the second of back-to-back days — downgrade minutes for starters slightly unless player typically plays the same minutes historically on back-to-back days.

MANDATORY: Your projected minutes MUST sum to between 240 and 242. Add them up before responding. Include the sum as "totalMinutes" in your JSON output.

Respond ONLY with valid JSON. No markdown fences, no commentary."""


def build_team_prompt(team, opponent, team_players):
    """Build a Claude prompt for one team's minutes projections."""
    active = [p for p in team_players if p.get("currentProjMin", 0) > 0]
    out = [p for p in team_players if p.get("currentProjMin", 0) == 0 and p.get("l5Avg", 0) > 0]

    if not active:
        return None

    lines = [f"Project minutes for {team} vs {opponent} tonight.\n"]
    lines.append("ACTIVE PLAYERS (sorted by L5 avg):")

    active_sorted = sorted(active, key=lambda x: x.get("l5Avg", 0), reverse=True)
    for p in active_sorted:
        games_str = "  ".join(
            f"{g['minutes']:.0f}{'*' if g.get('started') else ''}"
            for g in p.get("last10Games", [])
            if g.get("date")
        )
        lines.append(
            f"- {p['name']} ({p.get('pos', '?')}, ${p.get('price', 0)}): "
            f"L5={p.get('l5Avg', 0)}, L10={p.get('l10Avg', 0)}"
        )
        lines.append(f"  Last 10: {games_str}")
        lines.append(f"  (* = started that game)")

    if out:
        lines.append("\nOUT PLAYERS:")
        for p in out:
            lines.append(f"- {p['name']}: L5={p.get('l5Avg', 0)} when active")

        # Substitution chains for out players
        has_chains = False
        for p in out:
            chain = p.get("substitutionChain", [])
            if chain:
                if not has_chains:
                    lines.append("\nSUBSTITUTION PATTERNS (when player goes OUT):")
                    has_chains = True
                chain_str = ", ".join(
                    f"{s['replacement']} ({s['pct']*100:.0f}%)"
                    for s in chain
                )
                lines.append(f"- When {p['name']} sits: {chain_str}")

    proj_str = ", ".join(
        f"{p['name']}={p.get('currentProjMin', 0)}"
        for p in active_sorted
    )
    lines.append(f"\nCurrent projected minutes from data source: {proj_str}")

    total_proj = sum(p.get("currentProjMin", 0) for p in active_sorted)
    lines.append(f"Data source total: {total_proj:.0f} min for {len(active_sorted)} active players")
    lines.append(f"\nYou MUST distribute exactly 240-241 total minutes across these {len(active_sorted)} players. Add up your projections before responding to verify.")

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
                # Strip markdown fences if present
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\n?", "", text)
                    text = re.sub(r"\n?```$", "", text)

                result = json.loads(text)
                players_list = result.get("players", [])
                team_total = sum(p.get("projectedMinutes", 0) for p in players_list)

                if team_total > 0 and not (239 <= team_total <= 242):
                    diff = 241.0 - team_total
                    if abs(diff) <= 5:
                        # Small fix: spread across bench players only (< 25 min)
                        bench = [p for p in players_list if p.get("projectedMinutes", 0) < 25]
                        if bench:
                            adj = diff / len(bench)
                            for p in bench:
                                p["projectedMinutes"] = round(p["projectedMinutes"] + adj, 1)
                            new_total = sum(p.get("projectedMinutes", 0) for p in players_list)
                            print(f"    Adjusted {team} bench: {team_total:.0f} → {new_total:.1f} min")
                        break  # close enough, done
                    elif attempt < 2:
                        # Off by >5: retry with correction message
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
                        continue  # retry with correction
                    else:
                        print(f"    WARNING: {team} still at {team_total:.0f} after correction — accepting")
                        break
                else:
                    break  # within range, done

            except json.JSONDecodeError as e:
                print(f"    JSON parse error for {team}: {e}")
                if attempt < 2:
                    time.sleep(2)
            except Exception as e:
                print(f"    Claude API error for {team}: {e}")
                if attempt < 2:
                    time.sleep(5)

        # Store results
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

        time.sleep(1)  # rate limit between teams

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
            "l5Avg": p.get("l5Avg", 0),
            "l10Avg": p.get("l10Avg", 0),
            "currentProjMin": p.get("currentProjMin", 0),
            "last10Games": p.get("last10Games", []),
        })

    # Sort by team, then projected minutes descending
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

CSV_FILE = "minutes-export.csv"


def write_csv(output):
    """Write a flat CSV of all minutes data for the Minutes tab."""
    players = output.get("players", [])
    if not players:
        print("  No players — skipping CSV")
        return

    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Player", "Team", "Proj Min", "Confidence",
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
                p.get("l5Avg", 0),
                p.get("l10Avg", 0),
            ]
            for g in games[:10]:
                row.append(g.get("minutes", 0) if g else 0)
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

    # Step 1b: Merge fresh Stokastic minutes from data.json
    try:
        slate = load_slate_data()
        if slate:
            profiles = merge_slate_into_profiles(profiles, slate)
    except Exception as e:
        print(f"WARNING: Could not load data.json: {e}")
        print("  Falling back to currentProjMin from player-profiles.json")

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
    print(f"\nWrote {OUTPUT_FILE}: {len(output['players'])} players, {proj_count} with Claude projections")
    print("Done!")


if __name__ == "__main__":
    main()




