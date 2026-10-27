from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from brasileirao.poisson import scoreline_pmf

Outcome = Literal["H", "D", "A"]
MAX_GOALS = 10


@dataclass(frozen=True)
class Fixture:
    match_id: int
    home_id: int
    away_id: int
    lam_home: float
    lam_away: float


@dataclass(frozen=True, eq=False)
class SimulationResult:
    team_ids: list[int]
    points: np.ndarray
    positions: np.ndarray


@dataclass(frozen=True)
class Zones:
    libertadores: int
    sula: int
    relegation: int


def _outcome_mask(shape: tuple[int, int], allowed: set[str]) -> np.ndarray:
    home_goals, away_goals = np.indices(shape)
    mask = np.zeros(shape)
    if "H" in allowed:
        mask[home_goals > away_goals] = 1
    if "D" in allowed:
        mask[home_goals == away_goals] = 1
    if "A" in allowed:
        mask[home_goals < away_goals] = 1
    return mask


def simulate_season(
    table,
    fixtures: list[Fixture],
    rho: float,
    n_sims: int,
    rng: np.random.Generator,
    constraints: dict[int, set[str]] | None = None,
) -> SimulationResult:
    """Simulate the rest of the season. `constraints` maps match_id to the allowed outcomes
    from the home side's perspective, e.g. {"H"} forces a home win.

    Ties are ranked by points, wins, goal difference, goals scored, then a random draw.
    Head-to-head is not applied in simulations, which is a known approximation."""
    team_ids = table["team_id"].tolist()
    index = {team_id: i for i, team_id in enumerate(team_ids)}
    n_teams = len(team_ids)

    points = np.tile(table["points"].to_numpy(dtype=np.int64), (n_sims, 1))
    wins = np.tile(table["won"].to_numpy(dtype=np.int64), (n_sims, 1))
    gf = np.tile(table["gf"].to_numpy(dtype=np.int64), (n_sims, 1))
    ga = np.tile(table["ga"].to_numpy(dtype=np.int64), (n_sims, 1))

    for fixture in fixtures:
        pmf = scoreline_pmf(fixture.lam_home, fixture.lam_away, rho, MAX_GOALS)
        allowed = (constraints or {}).get(fixture.match_id)
        if allowed:
            pmf = pmf * _outcome_mask(pmf.shape, allowed)
            pmf = pmf / pmf.sum()
        hg, ag = np.divmod(rng.choice(pmf.size, size=n_sims, p=pmf.ravel()), pmf.shape[1])

        h = index[fixture.home_id]
        a = index[fixture.away_id]
        home_win = hg > ag
        away_win = ag > hg
        draw = hg == ag
        points[:, h] += 3 * home_win + draw
        points[:, a] += 3 * away_win + draw
        wins[:, h] += home_win
        wins[:, a] += away_win
        gf[:, h] += hg
        ga[:, h] += ag
        gf[:, a] += ag
        ga[:, a] += hg

    gd = gf - ga
    tiebreak = rng.integers(0, 100, size=(n_sims, n_teams))
    key = (((points * 39 + wins) * 2001 + (gd + 1000)) * 1001 + gf) * 100 + tiebreak
    order = np.argsort(-key, axis=1, kind="stable")
    positions = np.empty_like(order)
    np.put_along_axis(positions, order, np.tile(np.arange(1, n_teams + 1), (n_sims, 1)), axis=1)

    return SimulationResult(team_ids=team_ids, points=points, positions=positions)


def position_probabilities(result: SimulationResult) -> np.ndarray:
    """Matrix [team, final position - 1] of probabilities."""
    n_teams = len(result.team_ids)
    probs = np.zeros((n_teams, n_teams))
    for k in range(1, n_teams + 1):
        probs[:, k - 1] = (result.positions == k).mean(axis=0)
    return probs


def summarize(result: SimulationResult, table: pd.DataFrame, zones: Zones) -> pd.DataFrame:
    n_teams = len(result.team_ids)
    info = table.set_index("team_id")
    rows = []
    for i, team_id in enumerate(result.team_ids):
        pos = result.positions[:, i]
        rows.append(
            {
                "team": info.at[team_id, "team"],
                "tla": info.at[team_id, "tla"],
                "exp_points": float(result.points[:, i].mean()),
                "exp_position": float(pos.mean()),
                "title": float(np.mean(pos == 1)),
                "libertadores": float(np.mean(pos <= zones.libertadores)),
                "sula": float(np.mean((pos > zones.libertadores) & (pos <= zones.libertadores + zones.sula))),
                "relegation": float(np.mean(pos > n_teams - zones.relegation)),
            }
        )
    return pd.DataFrame(rows).sort_values("exp_points", ascending=False, ignore_index=True)
