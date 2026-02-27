#!/usr/bin/env python3
"""
NBA Minutes Agent — Claude-Only Pipeline
Reads player-profiles.json (pushed by Colab notebook), sends per-team data
to Claude Haiku for minutes projections, and writes minutes-data.json.

No NBA API calls — all game data is pre-built by the Colab notebook.
"""

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
#  STEP 2: QUERY CLAUDE FOR MINUTES PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════

CLAUDE_SYSTEM = """You are an NBA minutes allocation analyst for DFS (daily fantasy sports).
You must distribute exactly 240 team minutes across active players for tonight's game (5 players × 48 min = 240). You may add up to 1 minute for overtime probability (target: 240-241 total).

ALLOCATION PROCESS — follow these steps in order:
1. Identify the 5 starters (highest recent minutes, marked with * in game logs). Allocate 28-36 min each.
2. Allocate remaining minutes to bench players based on recent usage (L5 avg heavily weighted).
3. Sum all projected minutes. If the total is not between 240 and 241, adjust bench players up or down by 1-2 min each until the total equals 240-241. Do NOT inflate starters beyond their recent averages.

GUIDELINES:
- Weight recent games heavily: L3 > L5 > L10
- Account for minutes redistribution when players are OUT
- Consider blowout risk: high spreads may reduce starter minutes in Q4
- Players with 0 projected minutes from the data source are OUT — do not include them
- No player should exceed 40 minutes unless they averaged 38+ over their L5
- The data source total is provided as a baseline — use it as a reference point

MANDATORY: Your projected minutes MUST sum to between 240 and 241. Add them up before responding. Include the sum as "totalMinutes" in your JSON output.

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

        for attempt in range(2):
            try:
                response = client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=2048,
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
                players_list = result.get("players", [])
                # Lightweight safety net: bench-only adjustment for small deviations
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
                    else:
                        print(f"    WARNING: {team} total is {team_total:.0f} min (off by {abs(diff):.0f}) — needs prompt tuning")
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

    # Step 2: Query Claude
    print("\nQuerying Claude for projections...")
    claude_projections = query_claude(profiles)

    # Step 3: Build output
    print("\nBuilding output...")
    output = build_output(profiles, claude_projections, season)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    proj_count = sum(1 for p in output["players"] if p.get("projectedMinutes"))
    print(f"\nWrote {OUTPUT_FILE}: {len(output['players'])} players, {proj_count} with Claude projections")
    print("Done!")


if __name__ == "__main__":
    main()

