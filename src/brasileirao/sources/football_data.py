import json
import time
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
            raise FootballDataError("FOOTBALL_DATA_TOKEN não configurado. Adicione-o ao arquivo .env.")
        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={"X-Auth-Token": token},
            timeout=30,
            transport=transport,
        )
        self._cache_dir = cache_dir

    def _get(self, url: str, params: dict) -> httpx.Response:
        # The free tier allows about 10 requests per minute. Wait and retry on 429.
        for attempt in range(4):
            response = self._http.get(url, params=params)
            if response.status_code != 429:
                response.raise_for_status()
                return response
            time.sleep(int(response.headers.get("Retry-After", 65)) if attempt < 3 else 0)
        response.raise_for_status()
        return response

    def official_standings(self, competition: str, season: int) -> dict[int, dict]:
        response = self._get(f"/competitions/{competition}/standings", {"season": season})
        table = response.json()["standings"][0]["table"]
        return {row["team"]["id"]: row for row in table}

    def season_matches(self, competition: str, season: int, refresh: bool = False) -> list[Match]:
        path = self._cache_dir / f"{competition}_{season}.json"
        if refresh or not path.exists():
            response = self._get(f"/competitions/{competition}/matches", {"season": season})
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(response.text, encoding="utf-8")
        payload = json.loads(path.read_text(encoding="utf-8"))
        return parse_matches(payload, season)
