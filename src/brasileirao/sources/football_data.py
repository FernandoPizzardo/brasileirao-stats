import json
from pathlib import Path

import httpx

from brasileirao.models import Match

BASE_URL = "https://api.football-data.org/v4"


class FootballDataError(RuntimeError):
    pass


def parse_matches(payload: dict, season: int) -> list[Match]:
    matches = []
    for raw in payload.get("matches", []):
        full_time = (raw.get("score") or {}).get("fullTime") or {}
        home, away = raw["homeTeam"], raw["awayTeam"]
        matches.append(
            Match(
                id=raw["id"],
                season=season,
                matchday=raw.get("matchday") or 0,
                utc_date=raw["utcDate"],
                status=raw["status"],
                home_id=home["id"],
                home_name=home["name"],
                home_tla=home.get("tla"),
                away_id=away["id"],
                away_name=away["name"],
                away_tla=away.get("tla"),
                home_goals=full_time.get("home"),
                away_goals=full_time.get("away"),
            )
        )
    return matches


class FootballDataClient:
    def __init__(self, token: str, cache_dir: Path, transport: httpx.BaseTransport | None = None):
        if not token:
            raise FootballDataError("FOOTBALL_DATA_TOKEN is not set. Add it to your .env file.")
        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={"X-Auth-Token": token},
            timeout=30,
            transport=transport,
        )
        self._cache_dir = cache_dir

    def season_matches(self, competition: str, season: int, refresh: bool = False) -> list[Match]:
        path = self._cache_dir / f"{competition}_{season}.json"
        if refresh or not path.exists():
            response = self._http.get(f"/competitions/{competition}/matches", params={"season": season})
            response.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response.text, encoding="utf-8")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return parse_matches(payload, season)
