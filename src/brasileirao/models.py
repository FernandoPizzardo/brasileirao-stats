from datetime import datetime

from pydantic import BaseModel


class Match(BaseModel):
    id: int
    season: int
    matchday: int
    utc_date: datetime
    status: str
    home_id: int
    home_name: str
    home_tla: str | None = None
    away_id: int
    away_name: str
    away_tla: str | None = None
    home_goals: int | None = None
    away_goals: int | None = None

    @property
    def is_played(self) -> bool:
        return self.status == "FINISHED" and self.home_goals is not None and self.away_goals is not None
