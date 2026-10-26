"""Download draft history, ADP and prior-season stats from the Sleeper API.

Builds three CSVs in data/:
  - draft_picks.csv: every pick from a user's completed drafts (who took whom)
  - adp_rankings.csv: Sleeper's pre-draft ADP for each of those seasons
  - prior_season_stats.csv: the top ADP players' stats from the season before

NFL teams are always the team a player began that season on (week 1).

Other managers' names and Sleeper IDs are replaced with labels like
"Manager A" unless --keep-names is passed.

Usage: python fetch_data.py <sleeper_username> [--keep-names]
"""

import argparse
import csv
import json
import urllib.request
from functools import lru_cache
from pathlib import Path

API = "https://api.sleeper.app/v1"
# Undocumented endpoints behind the Sleeper app's projections/ADP and stats.
PROJECTIONS_API = "https://api.sleeper.app/projections/nfl"
STATS_API = "https://api.sleeper.app/stats/nfl"

DEFAULT_USERNAME = "jdemps46"
SEASONS = range(2018, 2027)
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
POSITION_QUERY = "&".join(f"position[]={p}" for p in POSITIONS)
START_WEEKS = range(1, 19)  # first week a player appears sets their team

DATA_DIR = Path(__file__).resolve().parent / "data"
PICKS_CSV = DATA_DIR / "draft_picks.csv"
FIELDS = [
    "season", "draft_name", "draft_id", "draft_type", "round", "pick_no",
    "draft_slot", "owner", "owner_id", "team_name", "player", "position", "nfl_team",
    "player_id", "is_keeper",
]

ADP_CSV = DATA_DIR / "adp_rankings.csv"
ADP_COLUMNS = ["adp_ppr", "adp_half_ppr", "adp_std", "adp_2qb"]
ADP_MISSING = 999  # Sleeper's placeholder for "no ADP"
ADP_FIELDS = [
    "season", "rank_ppr", "player", "position", "nfl_team", "player_id",
    *ADP_COLUMNS,
]

STATS_CSV = DATA_DIR / "prior_season_stats.csv"
STATS_TOP_N = 200
STAT_COLUMNS = [
    "gp", "gs",
    # passing
    "pass_att", "pass_cmp", "pass_yd", "pass_td", "pass_int", "pass_2pt",
    # rushing
    "rush_att", "rush_yd", "rush_td", "rush_2pt",
    # receiving
    "rec_tgt", "rec", "rec_yd", "rec_td", "rec_2pt",
    "fum", "fum_lost",
    # kicking
    "fga", "fgm", "xpa", "xpm",
    # team defense
    "sack", "int", "fum_rec", "ff", "def_td", "safe", "pts_allow", "yds_allow",
    # fantasy points
    "pts_std", "pts_half_ppr", "pts_ppr",
]
STATS_FIELDS = [
    "adp_season", "stats_season", "rank_ppr", "adp_ppr", "player", "position",
    "nfl_team", "player_id", "has_stats", *STAT_COLUMNS,
]


@lru_cache(maxsize=None)
def fetch(url):
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def get(path):
    return fetch(f"{API}{path}")


def player_name(player):
    return f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()


@lru_cache(maxsize=None)
def season_start_teams(season):
    """Map player_id -> NFL team in the first week they appeared that season."""
    teams = {}
    for week in START_WEEKS:
        url = f"{STATS_API}/{season}/{week}?season_type=regular&{POSITION_QUERY}"
        for s in fetch(url) or []:
            if s.get("team"):
                teams.setdefault(s["player_id"], s["team"])
    return teams


def league_members(league_id):
    """Map user_id -> (display_name, team_name) for a league."""
    if not league_id:
        return {}
    members = {}
    for u in get(f"/league/{league_id}/users") or []:
        team = (u.get("metadata") or {}).get("team_name") or ""
        members[u["user_id"]] = (u.get("display_name", ""), team)
    return members


def draft_rows(draft):
    members = league_members(draft.get("league_id"))
    teams = season_start_teams(draft["season"])
    for p in get(f"/draft/{draft['draft_id']}/picks"):
        m = p.get("metadata") or {}
        owner, team = members.get(p.get("picked_by"), (p.get("picked_by", ""), ""))
        yield {
            "season": draft["season"],
            "draft_name": (draft.get("metadata") or {}).get("name", ""),
            "draft_id": draft["draft_id"],
            "draft_type": draft.get("type", ""),
            "round": p["round"],
            "pick_no": p["pick_no"],
            "draft_slot": p["draft_slot"],
            "owner": owner,
            "owner_id": p.get("picked_by", ""),
            "team_name": team or owner,
            "player": player_name(m),
            "position": m.get("position", ""),
            "nfl_team": teams.get(p.get("player_id")) or m.get("team", ""),
            "player_id": p.get("player_id", ""),
            "is_keeper": bool(p.get("is_keeper")),
        }


def adp_rows(season):
    """Sleeper's pre-draft ADP for a season (the same ADP shown in the app)."""
    url = f"{PROJECTIONS_API}/{season}?season_type=regular&{POSITION_QUERY}&order_by=adp_ppr"
    teams = season_start_teams(season)
    rows = []
    for p in fetch(url):
        stats = p.get("stats") or {}
        if stats.get("adp_ppr", ADP_MISSING) >= ADP_MISSING:
            continue
        player = p.get("player") or {}
        rows.append({
            "season": season,
            "player": player_name(player),
            "position": player.get("position", ""),
            "nfl_team": teams.get(p["player_id"]) or p.get("team") or "",
            "player_id": p["player_id"],
            **{col: stats.get(col, "") for col in ADP_COLUMNS},
        })
    rows.sort(key=lambda r: r["adp_ppr"])
    for rank, r in enumerate(rows, 1):
        r["rank_ppr"] = rank
    return rows


def prior_season_stats_rows(adp, adp_season):
    """Previous-season totals for the top STATS_TOP_N players in ADP."""
    stats_season = int(adp_season) - 1
    url = f"{STATS_API}/{stats_season}?season_type=regular&{POSITION_QUERY}"
    season_stats = {s["player_id"]: s.get("stats") or {} for s in fetch(url)}
    teams = season_start_teams(stats_season)

    for r in adp[:STATS_TOP_N]:
        stats = season_stats.get(r["player_id"])
        played = bool(stats and stats.get("gp"))
        yield {
            "adp_season": adp_season,
            "stats_season": stats_season,
            "rank_ppr": r["rank_ppr"],
            "adp_ppr": r["adp_ppr"],
            "player": r["player"],
            "position": r["position"],
            "nfl_team": teams.get(r["player_id"], ""),
            "player_id": r["player_id"],
            "has_stats": played,
            **{col: stats.get(col, 0) if played else "" for col in STAT_COLUMNS},
        }


def anonymize(picks, user_id):
    """Replace every other manager's name, team name and Sleeper user ID with a
    stable label, and draft IDs with "<season>-<n>". Sleeper's API is public,
    so the raw IDs alone would let anyone look up the real usernames."""
    others = sorted({r["owner_id"] for r in picks} - {user_id})
    labels = {oid: f"Manager {chr(ord('A') + i)}" for i, oid in enumerate(others)}
    draft_ids = {}
    for r in sorted(picks, key=lambda r: (int(r["season"]), r["draft_id"])):
        if r["draft_id"] not in draft_ids:
            n = sum(k[0] == r["season"] for k in draft_ids.values()) + 1
            draft_ids[r["draft_id"]] = (r["season"], n)
    for r in picks:
        season, n = draft_ids[r["draft_id"]]
        r["draft_id"] = f"{season}-{n}"
        if r["owner_id"] in labels:
            r["owner"] = r["team_name"] = r["owner_id"] = labels[r["owner_id"]]


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def main(username=DEFAULT_USERNAME, keep_names=False):
    user = get(f"/user/{username}")
    picks = []
    for season in SEASONS:
        for draft in get(f"/user/{user['user_id']}/drafts/nfl/{season}") or []:
            if draft.get("status") != "complete":
                continue
            rows = list(draft_rows(draft))
            name = (draft.get("metadata") or {}).get("name", "")
            print(f"{season} {name}: {len(rows)} picks")
            picks.extend(rows)
    picks.sort(key=lambda r: (-int(r["season"]), r["draft_id"], r["pick_no"]))

    # Managers rename themselves; label every pick with the user's latest name.
    latest_name = {}
    for r in sorted(picks, key=lambda r: int(r["season"])):
        latest_name[r["owner_id"]] = r["owner"]
    for r in picks:
        r["owner"] = latest_name[r["owner_id"]]
    if not keep_names:
        anonymize(picks, user["user_id"])
    write_csv(PICKS_CSV, FIELDS, picks)

    adp, stats = [], []
    for season in sorted({r["season"] for r in picks}, reverse=True):
        season_adp = adp_rows(season)
        print(f"{season} ADP: {len(season_adp)} players")
        adp.extend(season_adp)
        stats.extend(prior_season_stats_rows(season_adp, season))
    write_csv(ADP_CSV, ADP_FIELDS, adp)
    write_csv(STATS_CSV, STATS_FIELDS, stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("username", nargs="?", default=DEFAULT_USERNAME,
                        help="Sleeper username whose drafts to download")
    parser.add_argument("--keep-names", action="store_true",
                        help="keep other managers' real names and Sleeper IDs")
    args = parser.parse_args()
    main(args.username, args.keep_names)
