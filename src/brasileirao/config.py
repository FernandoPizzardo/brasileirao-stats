from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    football_data_token: str = ""
    competition_code: str = "BSA"
    season: int = 2026
    history_seasons: int = 4
    data_dir: Path = Path("data")
    focus_team: str = "Corinthians"
    n_simulations: int = 20000
    random_seed: int = 42
    half_life_days: float = 365.0
    ridge: float = 0.1
    libertadores_spots: int = 6
    sula_spots: int = 6
    relegation_spots: int = 4


@lru_cache
def get_settings() -> Settings:
    return Settings()
