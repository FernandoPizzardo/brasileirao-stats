from datetime import datetime, timezone

from brasileirao.models import Match
from brasileirao.standings import standings


def played(id, home, away, hg, ag, names=None):
    names = names or {}
    return Match(
        id=id,
        season=2026,
        matchday=1,
        utc_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status="FINISHED",
        home_id=home,
        home_name=names.get(home, f"T{home}"),
        away_id=away,
        away_name=names.get(away, f"T{away}"),
        home_goals=hg,
        away_goals=ag,
    )


def test_points_and_order():
    matches = [played(1, 1, 2, 2, 0), played(2, 2, 3, 1, 1), played(3, 3, 1, 0, 0)]
    df = standings(matches)
    assert df["team_id"].tolist() == [1, 3, 2]
    assert df["points"].tolist() == [4, 2, 1]


def test_head_to_head_breaks_ties():
    names = {1: "Zeta", 2: "Alpha"}
    matches = [
        played(1, 1, 2, 1, 0, names),
        played(2, 2, 3, 1, 0, names),
        played(3, 3, 1, 1, 0, names),
        played(4, 1, 4, 2, 0, names),
        played(5, 2, 4, 2, 0, names),
    ]
    df = standings(matches)
    assert df["team_id"].tolist()[:2] == [1, 2]
    assert df["points"].tolist() == [6, 6, 3, 0]


def test_unplayed_matches_do_not_count():
    matches = [played(1, 1, 2, 1, 0)]
    future = matches[0].model_copy(update={"id": 2, "status": "SCHEDULED", "home_goals": None, "away_goals": None})
    df = standings(matches + [future])
    assert df.set_index("team_id").loc[1, "points"] == 3
    assert df.set_index("team_id").loc[2, "played"] == 1
