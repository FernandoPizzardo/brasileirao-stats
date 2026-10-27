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
            log.warning("Skipping season %s: %s", season, exc)
    return matches


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
