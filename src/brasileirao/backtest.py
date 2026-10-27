from datetime import timedelta

import numpy as np
import pandas as pd

from brasileirao.models import Match
from brasileirao.poisson import fit, outcome_probs, scoreline_pmf

OUTCOMES = ("H", "D", "A")


def _actual(m: Match) -> str:
    if m.home_goals > m.away_goals:
        return "H"
    if m.home_goals < m.away_goals:
        return "A"
    return "D"


def backtest(
    matches: list[Match],
    half_life_days: float,
    ridge: float,
    min_matchday: int = 8,
) -> pd.DataFrame:
    """For each matchday, fit only on matches finished before its first kickoff, then
    predict every match on that matchday. Compares the model with a frequency baseline."""
    rows = []
    for season in sorted({m.season for m in matches}):
        played = [m for m in matches if m.season == season and m.is_played]
        for matchday in sorted({m.matchday for m in played}):
            if matchday < min_matchday:
                continue
            targets = [m for m in played if m.matchday == matchday]
            cutoff = min(m.utc_date for m in targets) - timedelta(seconds=1)

            model = fit(matches, cutoff, half_life_days, ridge)
            past = [m for m in matches if m.is_played and m.utc_date <= cutoff]
            counts = {o: sum(_actual(m) == o for m in past) for o in OUTCOMES}
            total = sum(counts.values())
            baseline = {o: counts[o] / total for o in OUTCOMES}

            for m in targets:
                lam_h, lam_a = model.expected_goals(m.home_id, m.away_id)
                p_home, p_draw, p_away = outcome_probs(scoreline_pmf(lam_h, lam_a, model.rho))
                model_p = {"H": p_home, "D": p_draw, "A": p_away}
                actual = _actual(m)
                rows.append(
                    {
                        "season": season,
                        "matchday": matchday,
                        "actual": actual,
                        **{f"model_{o}": model_p[o] for o in OUTCOMES},
                        **{f"base_{o}": baseline[o] for o in OUTCOMES},
                    }
                )
    return pd.DataFrame(rows)


def score(df: pd.DataFrame, prefix: str) -> dict[str, float]:
    probs = np.column_stack([df[f"{prefix}_{o}"] for o in OUTCOMES])
    onehot = np.column_stack([(df["actual"] == o).astype(float) for o in OUTCOMES])
    p_actual = (probs * onehot).sum(axis=1)
    return {
        "log_loss": float(-np.log(np.clip(p_actual, 1e-12, None)).mean()),
        "brier": float(((probs - onehot) ** 2).sum(axis=1).mean()),
        "accuracy": float((probs.argmax(axis=1) == onehot.argmax(axis=1)).mean()),
    }
