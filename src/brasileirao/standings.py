import pandas as pd

from brasileirao.models import Match

FIELDS = ["team_id", "tla", "team", "played", "won", "drawn", "lost", "gf", "ga", "gd", "points"]


def _sort_key(row: dict) -> tuple:
    return (-row["points"], -row["won"], -row["gd"], -row["gf"])


def _head_to_head_points(played: list[Match], team_ids: list[int]) -> dict[int, int]:
    members = set(team_ids)
    points = {team_id: 0 for team_id in team_ids}
    for m in played:
        if m.home_id not in members or m.away_id not in members:
            continue
        if m.home_goals > m.away_goals:
            points[m.home_id] += 3
        elif m.home_goals < m.away_goals:
            points[m.away_id] += 3
        else:
            points[m.home_id] += 1
            points[m.away_id] += 1
    return points


def _break_ties(rows: list[dict], played: list[Match]) -> list[dict]:
    ordered: list[dict] = []
    i = 0
    while i < len(rows):
        j = i
        while j + 1 < len(rows) and _sort_key(rows[j + 1]) == _sort_key(rows[i]):
            j += 1
        group = rows[i : j + 1]
        if len(group) > 1:
            h2h = _head_to_head_points(played, [r["team_id"] for r in group])
            group = sorted(group, key=lambda r: -h2h[r["team_id"]])
        ordered.extend(group)
        i = j + 1
    return ordered


def standings(matches: list[Match]) -> pd.DataFrame:
    """Pontos corridos table. Tiebreaks: points, wins, goal difference, goals scored,
    head-to-head points. Remaining ties keep alphabetical order. Card tiebreakers and
    the official draw are not modelled."""
    rows: dict[int, dict] = {}
    for m in matches:
        for team_id, name, tla in ((m.home_id, m.home_name, m.home_tla), (m.away_id, m.away_name, m.away_tla)):
            rows.setdefault(
                team_id,
                {"team_id": team_id, "tla": tla or "", "team": name,
                 "played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0},
            )

    played = [m for m in matches if m.is_played]
    for m in played:
        home, away = rows[m.home_id], rows[m.away_id]
        home["played"] += 1
        away["played"] += 1
        home["gf"] += m.home_goals
        home["ga"] += m.away_goals
        away["gf"] += m.away_goals
        away["ga"] += m.home_goals
        if m.home_goals > m.away_goals:
            home["won"] += 1
            away["lost"] += 1
        elif m.home_goals < m.away_goals:
            away["won"] += 1
            home["lost"] += 1
        else:
            home["drawn"] += 1
            away["drawn"] += 1

    table = []
    for row in sorted(rows.values(), key=lambda r: r["team"]):
        row["gd"] = row["gf"] - row["ga"]
        row["points"] = 3 * row["won"] + row["drawn"]
        table.append(row)

    ordered = _break_ties(sorted(table, key=_sort_key), played)
    df = pd.DataFrame(ordered, columns=FIELDS)
    df.insert(0, "position", range(1, len(df) + 1))
    return df
