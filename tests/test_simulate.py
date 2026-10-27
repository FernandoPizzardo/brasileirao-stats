from datetime import datetime, timezone

import numpy as np

from brasileirao.models import Match
from brasileirao.simulate import Fixture, Zones, simulate_season, summarize
from brasileirao.standings import standings


def match(id, home, away, hg=None, ag=None, status="FINISHED"):
    return Match(
        id=id,
        season=2026,
        matchday=1,
        utc_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        status=status,
        home_id=home,
        home_name=f"T{home}",
        away_id=away,
        away_name=f"T{away}",
        home_goals=hg,
        away_goals=ag,
    )


def test_forced_home_win_is_applied_in_every_simulation():
    table = standings([match(1, 1, 2, 0, 0), match(2, 1, 2, status="SCHEDULED")])
    fixtures = [Fixture(2, 1, 2, 1.4, 1.0)]
    result = simulate_season(
        table, fixtures, rho=0.0, n_sims=500, rng=np.random.default_rng(1), constraints={2: {"H"}}
    )
    assert (result.points[:, 0] == 4).all()
    assert (result.points[:, 1] == 1).all()
    assert (result.positions[:, 0] == 1).all()


def test_positions_are_permutations_and_title_probs_sum_to_one():
    matches = [
        match(1, 1, 2, 2, 1),
        match(2, 2, 3, 0, 0),
        match(3, 3, 1, 1, 3),
        match(4, 1, 3, status="SCHEDULED"),
        match(5, 2, 1, status="SCHEDULED"),
        match(6, 3, 2, status="SCHEDULED"),
    ]
    table = standings(matches)
    fixtures = [Fixture(4, 1, 3, 1.5, 1.0), Fixture(5, 2, 1, 1.2, 1.2), Fixture(6, 3, 2, 1.0, 1.1)]
    result = simulate_season(table, fixtures, rho=-0.05, n_sims=2000, rng=np.random.default_rng(7))

    n_teams = len(result.team_ids)
    assert (np.sort(result.positions, axis=1) == np.arange(1, n_teams + 1)).all()

    summary = summarize(result, table, Zones(libertadores=1, sula=1, relegation=1))
    assert abs(summary["title"].sum() - 1) < 1e-9
