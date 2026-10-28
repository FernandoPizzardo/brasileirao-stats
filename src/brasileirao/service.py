import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import numpy as np
import pandas as pd

from brasileirao.config import Settings
from brasileirao.models import Match
from brasileirao.poisson import Model, fit
from brasileirao.simulate import Fixture, SimulationResult, Zones, simulate_season
from brasileirao.sources.football_data import FootballDataClient
from brasileirao.standings import standings

log = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)
class Context:
    as_of: datetime
    model: Model
    table: pd.DataFrame
    open_matches: list[Match]
    fixtures: list[Fixture]


def load_matches(settings: Settings, refresh: bool = False) -> list[Match]:
    client = FootballDataClient(settings.football_data_token, settings.data_dir / "raw")
    matches: list[Match] = []
    for season in range(settings.season - settings.history_seasons, settings.season + 1):
        is_current = season == settings.season
        try:
            matches += client.season_matches(settings.competition_code, season, refresh=refresh and is_current)
        except httpx.HTTPError as exc:
            if is_current:
                raise
            log.warning("Pulando a temporada %s: %s", season, exc)
    return matches


def verify_season(settings: Settings, season: int, matches: list[Match]) -> list[str]:
    """Compare our table with the official one. Returns a list of mismatch descriptions."""
    client = FootballDataClient(settings.football_data_token, settings.data_dir / "raw")
    official = client.official_standings(settings.competition_code, season)
    ours = standings([m for m in matches if m.season == season]).set_index("team_id")
    problems = []
    for team_id, row in ours.iterrows():
        o = official.get(team_id)
        if o is None:
            problems.append(f"{row['team']}: ausente na tabela oficial")
        elif (o["points"], o["goalDifference"], o["position"]) != (row["points"], row["gd"], row["position"]):
            problems.append(
                f"{row['team']}: nossa posição {row['position']} ({row['points']} pts, SG {row['gd']}) "
                f"x oficial {o['position']} ({o['points']} pts, SG {o['goalDifference']})"
            )
    return problems


def refresh_cache(settings: Settings) -> None:
    load_matches(settings, refresh=True)


def zones_from(settings: Settings) -> Zones:
    return Zones(
        libertadores=settings.libertadores_spots,
        sula=settings.sula_spots,
        relegation=settings.relegation_spots,
    )


def build_context(settings: Settings, refresh: bool = False) -> Context:
    matches = load_matches(settings, refresh=refresh)
    as_of = datetime.now(timezone.utc)
    model = fit(matches, as_of, settings.half_life_days, settings.ridge)

    season = [m for m in matches if m.season == settings.season]
    table = standings(season)
    open_matches = sorted(
        (m for m in season if not m.is_played and m.status != "CANCELLED"),
        key=lambda m: (m.matchday, m.utc_date),
    )
    fixtures = [
        Fixture(m.id, m.home_id, m.away_id, *model.expected_goals(m.home_id, m.away_id))
        for m in open_matches
    ]
    return Context(as_of=as_of, model=model, table=table, open_matches=open_matches, fixtures=fixtures)


def find_focus_team(table: pd.DataFrame, name: str) -> pd.DataFrame:
    return table[table["team"].str.contains(name, case=False, regex=False)]


def focus_matches(ctx: Context, focus_id: int) -> list[Match]:
    return [m for m in ctx.open_matches if focus_id in (m.home_id, m.away_id)]


SCENARIO_PRESETS: dict[str, tuple[str, str]] = {
    "Modelo": ("Model", "Model"),
    "Vence todos os jogos restantes": ("Win", "Win"),
    "Empata todos os jogos restantes": ("Draw", "Draw"),
    "Perde todos os jogos restantes": ("Loss", "Loss"),
    "Vence em casa e perde fora": ("Win", "Loss"),
    "Perde em casa e vence fora": ("Loss", "Win"),
}


def preset_choices(ctx: Context, focus_id: int, preset: str) -> dict[int, str]:
    """Choice for every remaining match of the focus team: home results and away results may differ."""
    home_choice, away_choice = SCENARIO_PRESETS[preset]
    return {m.id: (home_choice if m.home_id == focus_id else away_choice) for m in focus_matches(ctx, focus_id)}


def focus_constraints(ctx: Context, focus_id: int, choices: dict[int, str]) -> dict[int, set[str]]:
    """choices maps match_id to "Win", "Draw", "Loss" or "Model" (no constraint) from the focus team's perspective."""
    by_id = {m.id: m for m in ctx.open_matches}
    constraints: dict[int, set[str]] = {}
    for match_id, choice in choices.items():
        if choice == "Model":
            continue
        is_home = by_id[match_id].home_id == focus_id
        if choice == "Draw":
            constraints[match_id] = {"D"}
        elif (choice == "Win") == is_home:
            constraints[match_id] = {"H"}
        else:
            constraints[match_id] = {"A"}
    return constraints


def simulate(
    ctx: Context,
    settings: Settings,
    constraints: dict[int, set[str]] | None = None,
) -> SimulationResult:
    rng = np.random.default_rng(settings.random_seed)
    return simulate_season(
        ctx.table,
        ctx.fixtures,
        ctx.model.rho,
        settings.n_simulations,
        rng,
        constraints,
    )
